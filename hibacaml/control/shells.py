"""Structural shell editing for HiBaCaML."""

from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphState, NodeParams
from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.certificates import (
    CertificateController,
    effective_precision,
    shell_node_names,
    shell_slices,
)
from hibacaml.types import PersistentHiBaCaMLState


_PROMOTION_PAIRS = (
    ("tier3", "tier2"),
    ("tier2", "tier1"),
    ("tier1", "kernel"),
)

_DEMOTION_PAIRS = (
    ("kernel", "tier1"),
    ("tier1", "tier2"),
    ("tier2", "tier3"),
)

_OCCUPANCY_SHELLS = ("tier3", "tier2")

_PRUNE_SHELLS = (
    ("tier3", "outer_quantile"),
    ("tier2", "middle_quantile"),
)


def _shell_occupancy(node_params: NodeParams, shell_name: str) -> jnp.ndarray:
    log_precision = node_params.biases[f"log_precision_{shell_name}"]
    weight = node_params.weights[shell_name]
    live = (jnp.linalg.norm(weight, axis=0) > 1e-6) & (
        effective_precision(log_precision.squeeze(0).squeeze(0)) > 0.2
    )
    return jnp.mean(live.astype(jnp.float32))


class ShellController:
    """Structural editing of column shells."""

    def __init__(self, cfg: HiBaCaMLConfig, certificates: CertificateController):
        self.cfg = cfg
        # Editing ends by refreshing what it changed, so the observer is a
        # collaborator rather than a caller's responsibility.
        self.certificates = certificates
        self.column_nodes = certificates.column_nodes
        self.shell_slices = shell_slices(cfg)
        self._shell_widths = {
            name: int(shell_slice.stop - shell_slice.start)
            for name, shell_slice in self.shell_slices.items()
        }
        self.inhibition_strengths = {
            "tier1": 0.30,
            "tier2": 0.18,
            "tier3": 0.08,
        }

    def _active_node_entries(self, active_columns: Sequence[int]):
        columns = tuple(int(column) for column in active_columns)
        if len(set(columns)) != len(columns):
            raise ValueError("active_columns must not contain duplicates")
        return tuple(
            (column, node_name)
            for column in columns
            for node_name in shell_node_names(self.column_nodes[column])
        )

    def _project_structural_decisions(
        self,
        params,
        graph_state,
        node_names,
        phi,
    ):
        if not node_names:
            decisions = (
                jnp.zeros((0, len(_OCCUPANCY_SHELLS)), dtype=jnp.float32),
                tuple(
                    jnp.zeros((0, self._shell_widths[shell]), dtype=bool)
                    for shell, _ in _PRUNE_SHELLS
                ),
                jnp.zeros((0, len(_PROMOTION_PAIRS)), dtype=jnp.int32),
                jnp.zeros((0, len(_PROMOTION_PAIRS)), dtype=jnp.int32),
                jnp.zeros((0, len(_PROMOTION_PAIRS)), dtype=jnp.float32),
            )
            return decisions, ()

        occupancies = []
        prune_masks = {shell: [] for shell, _ in _PRUNE_SHELLS}
        outer_indices = []
        inner_indices = []
        promotion_gaps = []
        inhibitions = []

        for node_name in node_names:
            node_params = params.nodes[node_name]
            metrics = self._unit_metrics(
                node_params,
                graph_state.nodes[node_name],
            )
            inhibition = self._inhibition_adjustment(
                node_params,
                {
                    shell: entry["redundancy"]
                    for shell, entry in metrics.items()
                    if "redundancy" in entry
                },
            )
            inhibitions.append(inhibition)
            inhibited = self._apply_inhibition(node_params, inhibition)
            occupancies.append(
                jnp.stack(
                    [
                        _shell_occupancy(inhibited, shell)
                        for shell in _OCCUPANCY_SHELLS
                    ]
                )
            )

            for shell, quantile_field in _PRUNE_SHELLS:
                scores = metrics[shell]["score"]
                if scores.size == 0:
                    mask = jnp.zeros((0,), dtype=bool)
                else:
                    mask = scores <= jnp.quantile(
                        scores,
                        getattr(phi, quantile_field),
                    )
                prune_masks[shell].append(mask)

            node_outer_indices = []
            node_inner_indices = []
            node_gaps = []
            for outer_shell, inner_shell in _PROMOTION_PAIRS:
                outer_scores = metrics[outer_shell]["score"]
                inner_scores = metrics[inner_shell]["score"]
                if outer_scores.size == 0 or inner_scores.size == 0:
                    node_outer_indices.append(jnp.int32(-1))
                    node_inner_indices.append(jnp.int32(-1))
                    node_gaps.append(jnp.float32(0.0))
                    continue
                outer_index = jnp.argmax(outer_scores).astype(jnp.int32)
                inner_index = jnp.argmin(inner_scores).astype(jnp.int32)
                node_outer_indices.append(outer_index)
                node_inner_indices.append(inner_index)
                node_gaps.append(
                    outer_scores[outer_index] - inner_scores[inner_index]
                )
            outer_indices.append(jnp.stack(node_outer_indices))
            inner_indices.append(jnp.stack(node_inner_indices))
            promotion_gaps.append(jnp.stack(node_gaps))

        decisions = (
            jnp.stack(occupancies),
            tuple(
                jnp.stack(prune_masks[shell]) for shell, _ in _PRUNE_SHELLS
            ),
            jnp.stack(outer_indices),
            jnp.stack(inner_indices),
            jnp.stack(promotion_gaps),
        )
        return decisions, tuple(inhibitions)

    def _project_demotion_decisions(
        self,
        params,
        graph_state,
        node_names,
    ):
        if not node_names:
            shape = (0, len(_DEMOTION_PAIRS))
            return (
                jnp.zeros(shape, dtype=jnp.int32),
                jnp.zeros(shape, dtype=jnp.float32),
                jnp.zeros(shape, dtype=jnp.int32),
            )

        inner_indices = []
        inner_specificities = []
        outer_indices = []
        for node_name in node_names:
            metrics = self._unit_metrics(
                params.nodes[node_name],
                graph_state.nodes[node_name],
            )
            node_inner_indices = []
            node_specificities = []
            node_outer_indices = []
            for inner_shell, outer_shell in _DEMOTION_PAIRS:
                specificity = metrics[inner_shell]["specificity"]
                outer_scores = metrics[outer_shell]["score"]
                if specificity.size == 0 or outer_scores.size == 0:
                    node_inner_indices.append(jnp.int32(-1))
                    node_specificities.append(jnp.float32(0.0))
                    node_outer_indices.append(jnp.int32(-1))
                    continue
                inner_index = jnp.argmax(specificity).astype(jnp.int32)
                node_inner_indices.append(inner_index)
                node_specificities.append(specificity[inner_index])
                node_outer_indices.append(
                    jnp.argmin(outer_scores).astype(jnp.int32)
                )
            inner_indices.append(jnp.stack(node_inner_indices))
            inner_specificities.append(jnp.stack(node_specificities))
            outer_indices.append(jnp.stack(node_outer_indices))
        return (
            jnp.stack(inner_indices),
            jnp.stack(inner_specificities),
            jnp.stack(outer_indices),
        )

    def apply_structural_edits(
        self,
        params: GraphParams,
        graph_state: GraphState,
        persistent_state: PersistentHiBaCaMLState,
        active_columns: Sequence[int],
        phi,
    ) -> GraphParams:
        """Apply inhibition, pruning, and promotion to active columns."""
        
        if not self.cfg.exact_search.enable_structural_edits:
            return params

        entries = self._active_node_entries(active_columns)
        node_names = tuple(node_name for _, node_name in entries)
        decisions, inhibitions = self._project_structural_decisions(
            params,
            graph_state,
            node_names,
            phi,
        )
        (
            occupancies,
            prune_masks,
            outer_indices,
            inner_indices,
            promotion_gaps,
        ) = jax.device_get(decisions)

        updated_nodes = dict(params.nodes)
        targets = self.cfg.exact_search.semantic_targets
        for node_index, (_, node_name) in enumerate(entries):
            updated_node = self._apply_inhibition(
                updated_nodes[node_name],
                inhibitions[node_index],
            )
            for shell_index, ((shell_name, _), target) in enumerate(
                zip(_PRUNE_SHELLS, (targets[2], targets[1]))
            ):
                if float(occupancies[node_index, shell_index]) > target:
                    indices = tuple(
                        int(index)
                        for index in np.flatnonzero(
                            prune_masks[shell_index][node_index]
                        )
                    )
                    updated_node = self._prune_units(
                        updated_node,
                        shell_name,
                        indices,
                    )

            for pair_index, (outer_shell, inner_shell) in enumerate(
                _PROMOTION_PAIRS
            ):
                outer_index = int(outer_indices[node_index, pair_index])
                inner_index = int(inner_indices[node_index, pair_index])
                if outer_index < 0 or inner_index < 0:
                    continue
                if (
                    float(promotion_gaps[node_index, pair_index])
                    > phi.replacement_margin_base
                ):
                    updated_node = self._swap_units(
                        updated_node,
                        inner_shell,
                        inner_index,
                        outer_shell,
                        outer_index,
                    )
            updated_nodes[node_name] = updated_node

        params = params._replace(nodes=updated_nodes)
        self.certificates.refresh_certificates(params, graph_state, persistent_state)
        return params

    def demotion_swap_candidates(
        self,
        params: GraphParams,
        graph_state: GraphState,
        active_columns: Sequence[int],
        phi,
        *,
        max_candidates: int,
    ) -> list[dict[str, object]]:
        """Return conservative demotion-swap candidates ranked by specificity excess."""
        entries = self._active_node_entries(active_columns)
        node_names = tuple(node_name for _, node_name in entries)
        inner_indices, specificities, outer_indices = jax.device_get(
            self._project_demotion_decisions(
                params,
                graph_state,
                node_names,
            )
        )

        gain = float(phi.demotion_min_role_gain)
        candidates = []
        for node_index, (column_index, node_name) in enumerate(entries):
            for pair_index, (inner_shell, outer_shell) in enumerate(
                _DEMOTION_PAIRS
            ):
                inner_index = int(inner_indices[node_index, pair_index])
                outer_index = int(outer_indices[node_index, pair_index])
                if inner_index < 0 or outer_index < 0:
                    continue
                specificity = float(specificities[node_index, pair_index])
                if specificity <= gain:
                    continue
                candidates.append(
                    {
                        "score": specificity - gain,
                        "column_index": int(column_index),
                        "node_name": node_name,
                        "inner_shell": inner_shell,
                        "outer_shell": outer_shell,
                        "inner_index": inner_index,
                        "outer_index": outer_index,
                    }
                )
        candidates.sort(key=lambda row: float(row["score"]), reverse=True)
        return candidates[: max(0, int(max_candidates))]

    def apply_demotion_swap(
        self,
        params: GraphParams,
        *,
        node_name: str,
        inner_shell: str,
        outer_shell: str,
        inner_index: int,
        outer_index: int,
    ) -> GraphParams:
        """Apply one pre-audited demotion swap."""
        updated_nodes = dict(params.nodes)
        updated_nodes[node_name] = self._swap_units(
            updated_nodes[node_name],
            inner_shell,
            int(inner_index),
            outer_shell,
            int(outer_index),
        )
        return params._replace(nodes=updated_nodes)

    def _unit_metrics(self, node_params: NodeParams, node_state) -> Dict[str, Dict[str, jnp.ndarray]]:
        metrics = {}
        for shell_name, shell_slice in self.shell_slices.items():
            weight = node_params.weights[shell_name]
            precision = effective_precision(node_params.biases[f"log_precision_{shell_name}"].reshape(-1))
            shell_values = node_state.z_mu[..., shell_slice]
            if weight.shape[1] == 0:
                metrics[shell_name] = {
                    "score": jnp.zeros((0,)),
                    "specificity": jnp.zeros((0,)),
                }
                continue
            unit_norm = jnp.linalg.norm(weight, axis=0)
            redundancy = self._redundancy_penalty(weight)
            score = precision * unit_norm - redundancy
            specificity = jnp.var(shell_values, axis=tuple(range(shell_values.ndim - 1)))
            metrics[shell_name] = {
                "score": score,
                "specificity": specificity.reshape(-1),
                "redundancy": redundancy,
            }
        return metrics

    def _redundancy_penalty(self, weight: jnp.ndarray) -> jnp.ndarray:
        if weight.shape[1] <= 1:
            return jnp.zeros((weight.shape[1],), dtype=weight.dtype)
        normed = weight / (jnp.linalg.norm(weight, axis=0, keepdims=True) + 1e-8)
        sim = jnp.matmul(normed.T, normed)
        sim = sim - jnp.eye(sim.shape[0], dtype=sim.dtype)
        return jnp.mean(jnp.maximum(sim, 0.0), axis=1)

    def _inhibition_adjustment(
        self,
        node_params: NodeParams,
        redundancy_by_shell: Dict[str, jnp.ndarray] | None = None,
    ) -> Dict[str, jnp.ndarray]:
        adjustment = {}
        redundancy_by_shell = redundancy_by_shell or {}
        for shell_name, strength in self.inhibition_strengths.items():
            # Reuse the value _unit_metrics already computed from these same params.
            # Empty shells are skipped there, so fall back when it is absent.
            redundancy = redundancy_by_shell.get(shell_name)
            if redundancy is None:
                redundancy = self._redundancy_penalty(node_params.weights[shell_name])
            adjust = redundancy
            for _ in range(2):
                adjust = redundancy + 0.5 * adjust
            adjustment[shell_name] = strength * adjust
        return adjustment

    def _apply_inhibition(self, node_params: NodeParams, inhibition: Dict[str, jnp.ndarray]) -> NodeParams:
        biases = dict(node_params.biases)
        for shell_name, adjust in inhibition.items():
            key = f"log_precision_{shell_name}"
            biases[key] = biases[key] - adjust.reshape((1, 1, -1))
        return NodeParams(weights=dict(node_params.weights), biases=biases)

    def _prune_units(self, node_params: NodeParams, shell_name: str, indices: Sequence[int]) -> NodeParams:
        if not indices:
            return node_params
        weights = dict(node_params.weights)
        biases = dict(node_params.biases)
        keep = np.ones((weights[shell_name].shape[1],), dtype=bool)
        keep[list(indices)] = False
        keep = jnp.asarray(keep)
        weights[shell_name] = jnp.where(
            keep[None, :],
            weights[shell_name],
            0.0,
        )
        bias_key = f"b_{shell_name}"
        precision_key = f"log_precision_{shell_name}"
        biases[bias_key] = jnp.where(keep, biases[bias_key], 0.0)
        biases[precision_key] = jnp.where(
            keep,
            biases[precision_key],
            -8.0,
        )
        return NodeParams(weights=weights, biases=biases)

    def _swap_units(
        self,
        node_params: NodeParams,
        inner_shell: str,
        inner_idx: int,
        outer_shell: str,
        outer_idx: int,
    ) -> NodeParams:
        weights = dict(node_params.weights)
        biases = dict(node_params.biases)
        inner_col = weights[inner_shell][:, inner_idx]
        outer_col = weights[outer_shell][:, outer_idx]
        inner_selection = jnp.arange(weights[inner_shell].shape[1]) == inner_idx
        outer_selection = jnp.arange(weights[outer_shell].shape[1]) == outer_idx
        weights[inner_shell] = jnp.where(
            inner_selection[None, :],
            outer_col[:, None],
            weights[inner_shell],
        )
        weights[outer_shell] = jnp.where(
            outer_selection[None, :],
            inner_col[:, None],
            weights[outer_shell],
        )
        for prefix in ("b_", "log_precision_"):
            inner_key = f"{prefix}{inner_shell}"
            outer_key = f"{prefix}{outer_shell}"
            inner_value = biases[inner_key][..., inner_idx]
            outer_value = biases[outer_key][..., outer_idx]
            biases[inner_key] = jnp.where(
                inner_selection,
                outer_value[..., None],
                biases[inner_key],
            )
            biases[outer_key] = jnp.where(
                outer_selection,
                inner_value[..., None],
                biases[outer_key],
            )
        return NodeParams(weights=weights, biases=biases)


def precision_weight_gradients(
    grads: GraphParams,
    params: GraphParams,
    *,
    shell_names: Sequence[str],
    enabled: bool,
    strength: float,
    floor: float,
) -> GraphParams:
    """Precondition shell gradients by inverse effective precision."""

    if not enabled:
        return grads
    strength = max(float(strength), 0.0)
    floor = max(float(floor), 0.0)
    updated_nodes = {}
    for node_name, node_grads in grads.nodes.items():
        node_params = params.nodes.get(node_name)
        if node_params is None:
            updated_nodes[node_name] = node_grads
            continue
        weights = dict(node_grads.weights)
        biases = dict(node_grads.biases)
        for shell_name in shell_names:
            precision_key = f"log_precision_{shell_name}"
            if shell_name not in weights or precision_key not in node_params.biases:
                continue
            precision = effective_precision(node_params.biases[precision_key])
            factor = jnp.maximum(floor, 1.0 / (1.0 + strength * precision))
            weights[shell_name] = weights[shell_name] * factor.reshape((1, -1))
            recurrent_key = f"recur_{shell_name}"
            if recurrent_key in weights:
                weights[recurrent_key] = weights[recurrent_key] * factor.reshape((1, -1))
            for bias_key in (f"b_{shell_name}", precision_key):
                if bias_key in biases:
                    biases[bias_key] = biases[bias_key] * factor
        updated_nodes[node_name] = node_grads._replace(weights=weights, biases=biases)
    return grads._replace(nodes=updated_nodes)
