"""Post-T0 write protection: which parameter coordinates an update may change, and by how much.

A policy never changes the forward graph or its gradients; it masks gradients
before clipping, masks and scales the post-AdamW updates, and finally restores
the original bytes of every protected coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.certificates import shell_slices
from hibacaml.graph.builder import shell_node_positions

# Composer matrices that stop learning once a policy is installed.
FROZEN_COMPOSER_WEIGHTS = ("feature_proj", "out_proj")


@dataclass(frozen=True, eq=False)
class WritePolicy:
    """Write permission for one training condition.

    Installing any policy also freezes the stem convs and the composer's
    feature/output maps. `unit_scales[column, position, unit]` scales the
    post-AdamW update of each shell unit (positions B, K0.., L); 0 freezes it.
    """

    trainable_columns: Tuple[int, ...]
    frozen_feature_pools: Tuple[int, ...]
    unit_scales: np.ndarray


@dataclass(frozen=True)
class PolicyLayout:
    """Parameter ownership resolved once from the graph metadata."""

    column_of_node: Dict[str, int]
    stem: Tuple[str, ...]
    composer: str
    feature_pools: Dict[str, int]
    shells: Dict[str, Tuple[int, int]]
    regions: Dict[str, slice]

    @property
    def feature_pools_by_column(self) -> Dict[int, str]:
        return {column: name for name, column in self.feature_pools.items()}

    @classmethod
    def from_structure(cls, structure, cfg: HiBaCaMLConfig) -> "PolicyLayout":
        meta = structure.config["hibacaml"]
        return cls(
            # Column ownership is the `col<N>/` namespace, as for support masking.
            column_of_node={
                name: int(name.split("/")[0][3:]) for name in structure.nodes if name.startswith("col")
            },
            stem=tuple(meta.get("stem_nodes", ())),
            composer=meta["composer_node"],
            feature_pools={nodes["feature_pool"]: c for c, nodes in enumerate(meta["column_nodes"])},
            shells=shell_node_positions(structure),
            regions=shell_slices(cfg),
        )


def validate(policy: WritePolicy, cfg: HiBaCaMLConfig) -> None:
    pool = cfg.column_pool
    for name in ("trainable_columns", "frozen_feature_pools"):
        columns = getattr(policy, name)
        if len(set(columns)) != len(columns) or any(
            isinstance(c, bool) or not isinstance(c, (int, np.integer)) or not 0 <= c < pool.total_columns
            for c in columns
        ):
            raise ValueError(f"{name} must be distinct column indices below {pool.total_columns}")
    table = np.asarray(policy.unit_scales)
    expected = (pool.total_columns, 2 + max(1, pool.kernel_depth), pool.shell_output_dim)
    if table.shape != expected:
        raise ValueError(f"unit_scales must have shape {expected}, got {table.shape}")
    if not (np.all(np.isfinite(table)) and np.all((table >= 0) & (table <= 1))):
        raise ValueError("unit_scales must be finite and within [0, 1]")


def policy_inputs(policy: WritePolicy, cfg: HiBaCaMLConfig) -> Dict[str, jax.Array]:
    """Device inputs with fixed keys and shapes, so changing a policy never retraces."""
    total = cfg.column_pool.total_columns
    columns = np.arange(total)
    return {
        "write_mask": jnp.asarray(np.isin(columns, policy.trainable_columns)),
        "frozen_pools": jnp.asarray(np.isin(columns, policy.frozen_feature_pools)),
        "unit_scales": jnp.asarray(policy.unit_scales, dtype=jnp.float32),
    }


def _region(key: str, regions) -> str:
    for prefix in ("recur_", "b_", "log_precision_", ""):
        if key.startswith(prefix) and key[len(prefix):] in regions:
            return key[len(prefix):]
    raise ValueError(f"shell leaf {key!r} has no destination region")


def _frozen(tree, inputs, support, layout: PolicyLayout):
    """Per-leaf booleans, broadcastable to each leaf, marking coordinates the policy protects."""
    writable = (support > 0) & inputs["write_mask"]
    nodes = {}
    for name, node in tree.nodes.items():
        column = layout.column_of_node.get(name)
        if name in layout.stem:
            whole = jnp.bool_(True)
        elif column is None:
            whole = jnp.bool_(False)
        else:
            whole = ~writable[column]
            if name in layout.feature_pools:
                whole = whole | inputs["frozen_pools"][column]
        weights = {key: whole for key in node.weights}
        biases = {key: whole for key in node.biases}
        if name == layout.composer:
            weights.update({key: jnp.bool_(True) for key in FROZEN_COMPOSER_WEIGHTS})
        if name in layout.shells:
            column, position = layout.shells[name]
            for group in (weights, biases):
                for key in group:
                    units = inputs["unit_scales"][column, position, layout.regions[_region(key, layout.regions)]]
                    group[key] = whole | (units == 0)
        nodes[name] = node._replace(weights=weights, biases=biases)
    return tree._replace(nodes=nodes)


def mask(tree, inputs, support, layout: PolicyLayout):
    """Zero every protected gradient or update coordinate."""
    frozen = _frozen(tree, inputs, support, layout)
    return jax.tree_util.tree_map(lambda f, x: jnp.where(f, jnp.zeros_like(x), x), frozen, tree)


def scale(updates, inputs, layout: PolicyLayout):
    """Apply per-unit write scales to the complete post-AdamW update of each shell node."""
    nodes = dict(updates.nodes)
    for name, (column, position) in layout.shells.items():
        node = nodes[name]

        def scaled(group):
            out = {}
            for key, value in group.items():
                s = inputs["unit_scales"][column, position, layout.regions[_region(key, layout.regions)]]
                out[key] = jnp.where(s == 0, jnp.zeros_like(value), jnp.where(s == 1, value, s * value))
            return out

        nodes[name] = node._replace(weights=scaled(node.weights), biases=scaled(node.biases))
    return updates._replace(nodes=nodes)


def restore(new_params, params, inputs, support, layout: PolicyLayout):
    """Select the original bytes (signed zeros included) for every protected coordinate."""
    frozen = _frozen(params, inputs, support, layout)
    return jax.tree_util.tree_map(lambda f, new, old: jnp.where(f, old, new), frozen, new_params, params)


def frozen_leaves(params, policy: WritePolicy, layout: PolicyLayout):
    """`(node, field, key)` of every leaf a policy freezes outright, whatever the support.

    Stem convs, the composer maps, frozen feature pools, and shell leaves whose
    destination units all have scale 0 (the kernel leaves of scaled columns).
    """
    table = np.asarray(policy.unit_scales)
    pools = {layout.feature_pools_by_column[c] for c in policy.frozen_feature_pools}
    leaves = []
    for name, node in sorted(params.nodes.items()):
        for field in ("weights", "biases"):
            for key in sorted(getattr(node, field)):
                if name in layout.stem or name in pools or (
                    name == layout.composer and key in FROZEN_COMPOSER_WEIGHTS
                ):
                    leaves.append((name, field, key))
                elif name in layout.shells:
                    column, position = layout.shells[name]
                    region = layout.regions[_region(key, layout.regions)]
                    if not np.any(table[column, position, region]):
                        leaves.append((name, field, key))
    return tuple(leaves)
