"""Structural shell editing for HiBaCaML."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphState, NodeParams
from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.certificates import (
    CertificateController,
    effective_precision,
    shell_node_names,
    shell_occupancy,
    shell_slices,
)
from hibacaml.types import PersistentHiBaCaMLState


class ShellController:
    """Structural editing of column shells."""

    def __init__(self, cfg: HiBaCaMLConfig, certificates: CertificateController):
        self.cfg = cfg
        # Editing ends by refreshing what it changed, so the observer is a
        # collaborator rather than a caller's responsibility.
        self.certificates = certificates
        self.column_nodes = certificates.column_nodes
        self.shell_slices = shell_slices(cfg)
        self.inhibition_strengths = {
            "tier1": 0.30,
            "tier2": 0.18,
            "tier3": 0.08,
        }

    def apply_structural_edits(
        self,
        params: GraphParams,
        graph_state: GraphState,
        persistent_state: PersistentHiBaCaMLState,
        active_columns: Sequence[int],
        phi,
    ) -> GraphParams:
        """Apply inhibition, prune, promote, and demote to active columns."""
        if not self.cfg.exact_search.enable_structural_edits:
            return params
        updated_nodes = dict(params.nodes)
        for column_index in active_columns:
            meta = self.column_nodes[column_index]
            for node_name in shell_node_names(meta):
                node_params = updated_nodes[node_name]
                node_state = graph_state.nodes[node_name]
                metrics = self._unit_metrics(node_params, node_state)
                updated_node = self._apply_inhibition(
                    node_params,
                    self._inhibition_adjustment(
                        node_params,
                        {
                            shell: entry["redundancy"]
                            for shell, entry in metrics.items()
                            if "redundancy" in entry
                        },
                    ),
                )

                if shell_occupancy(updated_node, "tier3") > self.cfg.exact_search.semantic_targets[2]:
                    prune_idx = self._low_score_indices(metrics["tier3"]["score"], phi.outer_quantile)
                    updated_node = self._prune_units(updated_node, "tier3", prune_idx)

                if shell_occupancy(updated_node, "tier2") > self.cfg.exact_search.semantic_targets[1]:
                    prune_idx = self._low_score_indices(metrics["tier2"]["score"], phi.middle_quantile)
                    updated_node = self._prune_units(updated_node, "tier2", prune_idx)

                for outer_shell, inner_shell in (
                    ("tier3", "tier2"),
                    ("tier2", "tier1"),
                    ("tier1", "kernel"),
                ):
                    outer_scores = metrics[outer_shell]["score"]
                    inner_scores = metrics[inner_shell]["score"]
                    if outer_scores.size == 0 or inner_scores.size == 0:
                        continue
                    outer_idx = int(jnp.argmax(outer_scores))
                    inner_idx = int(jnp.argmin(inner_scores))
                    if float(outer_scores[outer_idx] - inner_scores[inner_idx]) > phi.replacement_margin_base:
                        updated_node = self._swap_units(
                            updated_node,
                            inner_shell,
                            inner_idx,
                            outer_shell,
                            outer_idx,
                        )

                # V20.2b: demotion swaps go through the audited
                # demotion_swap_audit path; no unaudited demotion here.
                updated_nodes[node_name] = updated_node

        params = params._replace(nodes=updated_nodes)
        # Editing returns with certificates already describing the edited graph.
        # The disabled path above returns before this, which is why refresh
        # cadence differs between the two branches.
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
        candidates = []
        for column_index in active_columns:
            meta = self.column_nodes[column_index]
            for node_name in shell_node_names(meta):
                node_params = params.nodes[node_name]
                node_state = graph_state.nodes[node_name]
                metrics = self._unit_metrics(node_params, node_state)
                for inner_shell, outer_shell in (
                    ("kernel", "tier1"),
                    ("tier1", "tier2"),
                    ("tier2", "tier3"),
                ):
                    inner_specificity = metrics[inner_shell]["specificity"]
                    outer_scores = metrics[outer_shell]["score"]
                    if inner_specificity.size == 0 or outer_scores.size == 0:
                        continue
                    inner_idx = int(jnp.argmax(inner_specificity))
                    specificity = float(inner_specificity[inner_idx])
                    if specificity <= phi.demotion_min_role_gain:
                        continue
                    outer_idx = int(jnp.argmin(outer_scores))
                    candidates.append(
                        {
                            "score": specificity - float(phi.demotion_min_role_gain),
                            "column_index": int(column_index),
                            "node_name": node_name,
                            "inner_shell": inner_shell,
                            "outer_shell": outer_shell,
                            "inner_index": int(inner_idx),
                            "outer_index": int(outer_idx),
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

    def _low_score_indices(self, scores: jnp.ndarray, quantile: float) -> Tuple[int, ...]:
        if scores.size == 0:
            return ()
        threshold = jnp.quantile(scores, quantile)
        return tuple(int(i) for i in jnp.where(scores <= threshold)[0].tolist())

    def _prune_units(self, node_params: NodeParams, shell_name: str, indices: Sequence[int]) -> NodeParams:
        if not indices:
            return node_params
        weights = dict(node_params.weights)
        biases = dict(node_params.biases)
        arr_idx = jnp.asarray(indices)
        weights[shell_name] = weights[shell_name].at[:, arr_idx].set(0.0)
        biases[f"b_{shell_name}"] = biases[f"b_{shell_name}"].at[..., arr_idx].set(0.0)
        biases[f"log_precision_{shell_name}"] = biases[f"log_precision_{shell_name}"].at[..., arr_idx].set(-8.0)
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
        weights[inner_shell] = weights[inner_shell].at[:, inner_idx].set(outer_col)
        weights[outer_shell] = weights[outer_shell].at[:, outer_idx].set(inner_col)
        for prefix in ("b_", "log_precision_"):
            inner_key = f"{prefix}{inner_shell}"
            outer_key = f"{prefix}{outer_shell}"
            inner_value = biases[inner_key][..., inner_idx]
            outer_value = biases[outer_key][..., outer_idx]
            biases[inner_key] = biases[inner_key].at[..., inner_idx].set(outer_value)
            biases[outer_key] = biases[outer_key].at[..., outer_idx].set(inner_value)
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
    """Precondition shell gradients by inverse effective precision.

    Module-level rather than a controller method so the training update can close
    over plain configuration values instead of a controller object.
    """
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
