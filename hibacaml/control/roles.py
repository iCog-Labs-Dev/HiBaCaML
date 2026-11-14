"""Per-unit shell tier labels (roles), separate from physical storage slots.

A unit's label is its tier: 0 kernel, 1-3 the shells. Labels start at the
physical layout; kernel units stay 0 and every node keeps its physical tier
sizes. Labels set write rates and certificate tier statistics, and changing
one never touches a parameter or moment.

The structural controller's host arithmetic lives here too (guide §9): unit
usage, same-label redundancy, multi-context promotion evidence, adjacent
relabelling, labelled inhibition and the pruning trigger. Their float32/float64
order follows the retained reference exactly.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.certificates import shell_slices
from hibacaml.graph.builder import shell_node_positions
from hibacaml.reporting.integrity import fingerprint

REGIONS = ("kernel", "tier1", "tier2", "tier3")  # physical storage order of a shell node's units
# Write scale per label after T0: kernel frozen, shells increasingly plastic.
REFERENCE_LABEL_SCALES = (0.0, 0.04140854626893997, 0.12422563880681992, 0.37267690896987915)

Roles = Dict[str, np.ndarray]


def physical_labels(cfg: HiBaCaMLConfig) -> np.ndarray:
    labels = np.zeros(cfg.column_pool.shell_output_dim, dtype=np.int8)
    for label, region in enumerate(shell_slices(cfg).values()):
        labels[region] = label
    return labels


def initial_roles(structure, cfg: HiBaCaMLConfig) -> Roles:
    base = physical_labels(cfg)
    return {name: base.copy() for name in shell_node_positions(structure)}


def validate_roles(structure, cfg: HiBaCaMLConfig, roles: Roles) -> Roles:
    """Every shell node labelled; kernel units stay 0; per-node tier sizes stay physical."""
    if not isinstance(roles, dict) or set(roles) != set(shell_node_positions(structure)):
        raise ValueError("shell roles must cover every shell node exactly")
    base = physical_labels(cfg)
    counts = np.bincount(base, minlength=4)
    for name, labels in roles.items():
        labels = np.asarray(labels)
        if labels.dtype != np.int8 or labels.shape != base.shape:
            raise ValueError(f"invalid role array: {name}")
        if not np.array_equal(labels == 0, base == 0):
            raise ValueError(f"kernel units must keep label 0: {name}")
        if labels.min() < 0 or labels.max() > 3 or not np.array_equal(np.bincount(labels, minlength=4), counts):
            raise ValueError(f"role labels must keep the physical tier sizes: {name}")
    return roles


def roles_digest(roles: Roles) -> str:
    return fingerprint({name: np.asarray(roles[name]) for name in sorted(roles)})


def unit_scale_table(
    structure,
    cfg: HiBaCaMLConfig,
    roles: Roles,
    scaled_columns: Iterable[int],
    label_scales: Sequence[float] = REFERENCE_LABEL_SCALES,
) -> np.ndarray:
    """`[columns, positions, units]` write scales: label scales on `scaled_columns`, 1 elsewhere."""
    scales = np.asarray(label_scales, dtype=np.float32)
    pool = cfg.column_pool
    table = np.ones((pool.total_columns, 2 + max(1, pool.kernel_depth), pool.shell_output_dim), np.float32)
    scaled = {int(c) for c in scaled_columns}
    for name, (column, position) in shell_node_positions(structure).items():
        if column in scaled:
            table[column, position] = scales[np.asarray(roles[name])]
    return table


# ---- structural-control arithmetic (host NumPy) ------------------------------------------------


def slot_region(cfg: HiBaCaMLConfig, slot: int) -> Tuple[str, int]:
    """`(region, index within it)` of a physical unit slot."""
    for region, bounds in shell_slices(cfg).items():
        if bounds.start <= slot < bounds.stop:
            return region, int(slot - bounds.start)
    raise ValueError(f"slot {slot} lies outside the shell units")


def node_arrays(node) -> Tuple[np.ndarray, np.ndarray]:
    """Feed-forward incoming weights `[in, units]` and log precisions `[units]`, float32, by slot."""
    weights = np.concatenate([np.asarray(node.weights[r], dtype=np.float32) for r in REGIONS], axis=-1)
    log_precision = np.concatenate(
        [np.asarray(node.biases["log_precision_" + r], dtype=np.float32).reshape(-1) for r in REGIONS])
    return weights, log_precision


def unit_usage(activations) -> np.ndarray:
    """Mean absolute activation of each unit over every leading axis, in float32."""
    values = np.abs(np.asarray(activations, dtype=np.float32))
    return values.reshape(-1, values.shape[-1]).mean(axis=0)


def redundancy(weights) -> np.ndarray:
    """Mean positive cosine overlap of each incoming column with the others (denominator n)."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] <= 1:
        return np.zeros((0 if weights.ndim != 2 else weights.shape[1],), dtype=np.float32)
    normed = weights / (np.linalg.norm(weights, axis=0, keepdims=True) + np.float32(1e-8))
    similarity = normed.T @ normed - np.eye(weights.shape[1], dtype=np.float32)
    return np.mean(np.maximum(similarity, np.float32(0.0)), axis=1)


def label_scores(node, labels) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per unit `(gate * incoming norm - same-label redundancy, live, redundancy)`."""
    weights, log_precision = node_arrays(node)
    labels = np.asarray(labels)
    gate = 1.0 / (1.0 + np.exp(-log_precision))
    norms = np.linalg.norm(weights, axis=0)
    shared = np.zeros_like(gate)
    for label in (1, 2, 3):
        units = np.flatnonzero(labels == label)
        shared[units] = redundancy(weights[:, units])
    return gate * norms - shared, (norms > 1e-6) & (gate > 0.2), shared


def evidence_pairs(roles: Roles, usage, nodes: Sequence[str], outer_label: int, *, min_contexts: int = 2,
                   frozen=frozenset()) -> List[Dict[str, object]]:
    """At most one adjacent promotion pair per node from multi-context usage evidence.

    A unit is used in a context when it reaches that context's median over the
    inner-label units. The outer unit must be used in `min_contexts` contexts and
    strictly more than its inner partner; `frozen` holds `(node, slot)` that already moved.
    """
    inner_label = outer_label - 1
    if inner_label < 1:
        raise ValueError("promotions never enter the kernel")
    pairs = []
    for node in nodes:
        contexts = sorted(usage[node])
        if len(contexts) < min_contexts:
            continue
        labels = np.asarray(roles[node])
        table = np.stack([np.asarray(usage[node][j], dtype=np.float64) for j in contexts])
        inner = np.flatnonzero(labels == inner_label)
        outer = np.flatnonzero(labels == outer_label)
        thresholds = np.median(table[:, inner], axis=1, keepdims=True)
        used = (table >= thresholds).sum(axis=0)
        mean = table.mean(axis=0)
        candidates = [int(u) for u in outer if (node, int(u)) not in frozen and used[u] >= min_contexts]
        partners = [int(u) for u in inner if (node, int(u)) not in frozen]
        if not candidates or not partners:
            continue
        promoted = min(candidates, key=lambda u: (-used[u], -mean[u], u))
        demoted = min(partners, key=lambda u: (used[u], mean[u], u))
        if used[demoted] >= used[promoted]:
            continue
        pairs.append(dict(
            node=node, outer_label=int(outer_label), inner_label=int(inner_label),
            promoted_slot=promoted, demoted_slot=demoted, contexts=[int(j) for j in contexts],
            promoted_contexts_used=int(used[promoted]), demoted_contexts_used=int(used[demoted]),
            promoted_mean_usage=float(mean[promoted]), demoted_mean_usage=float(mean[demoted]),
        ))
    return pairs


def relabel(roles: Roles, node: str, promoted_slot: int, demoted_slot: int) -> Roles:
    """New roles with two adjacent shell labels exchanged; `roles` is left untouched."""
    labels = np.asarray(roles[node])
    outer, inner = int(labels[promoted_slot]), int(labels[demoted_slot])
    if promoted_slot == demoted_slot or not (inner >= 1 and outer == inner + 1):
        raise ValueError("a relabel exchanges two distinct slots of adjacent shell tiers")
    updated = labels.copy()
    updated[promoted_slot], updated[demoted_slot] = inner, outer
    return {**roles, node: updated}


def inhibited_log_precision(weights, log_precision, labels, *, strengths=(0.30, 0.18, 0.08),
                            dose: float = 10.0, floor: float = 0.25) -> np.ndarray:
    """Same-tier inhibition of labelled shell gates; a gate at or below the floor never rises."""
    weights = np.asarray(weights, dtype=np.float32)
    after = np.asarray(log_precision, dtype=np.float32).copy()
    labels = np.asarray(labels)
    floor_logit = np.float32(math.log(floor / (1.0 - floor)))
    for tier in (1, 2, 3):
        units = np.flatnonzero(labels == tier)
        r = redundancy(weights[:, units])  # from the pre-edit weights
        adjustment = r
        for _ in range(2):
            adjustment = r + np.float32(0.5) * adjustment
        adjustment = np.float32(dose) * (np.float32(strengths[tier - 1]) * adjustment)
        for slot, value in zip(units, adjustment):
            before = after[slot]
            after[slot] = np.maximum(before - value, np.minimum(before, floor_logit))
    return after


def prune_trigger(named_nodes, roles: Roles, usage, label: int, *, weak_fraction: float = 0.10,
                  score_quantile: float = 0.125) -> Optional[Dict[str, object]]:
    """The node with the most weak live units of `label`, or None.

    Weak: live, usage below `weak_fraction` x the label's median usage, and role score
    at or below its `score_quantile` (both over every unit of the label). Ties go to
    higher mean label redundancy, then the earlier node.
    """
    best = None
    for order, (name, node) in enumerate(named_nodes):
        labels = np.asarray(roles[name])
        units = np.flatnonzero(labels == label)
        if units.size == 0:
            continue
        score, live, shared = label_scores(node, labels)
        values = np.asarray(usage[name], dtype=np.float64)
        weak = units[
            live[units]
            & (values[units] < weak_fraction * np.median(values[units]))
            & (score[units] <= np.quantile(score[units], score_quantile))
        ]
        if weak.size == 0:
            continue
        key = (int(weak.size), float(shared[units].mean()), -order)
        if best is None or key > best[0]:
            best = (key, name, tuple(int(u) for u in weak))
    if best is None:
        return None
    return dict(node=best[1], slots=best[2], weak_units=best[0][0], tier_redundancy=best[0][1])
