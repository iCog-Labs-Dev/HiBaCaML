"""Per-unit shell tier labels (roles), separate from physical storage slots.

A unit's label is its tier: 0 kernel, 1-3 the shells. Labels start at the
physical layout; kernel units stay 0 and every node keeps its physical tier
sizes. Labels set write rates and certificate tier statistics, and changing
one never touches a parameter or moment.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.certificates import shell_slices
from hibacaml.graph.builder import shell_node_positions
from hibacaml.reporting.integrity import fingerprint

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
