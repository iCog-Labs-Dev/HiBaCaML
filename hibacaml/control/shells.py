"""Shell certificates and structural edits for HiBaCaML."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphState, NodeParams
from hibacaml.config import HiBaCaMLConfig
from hibacaml.types import ColumnCertificate, PersistentHiBaCaMLState, ShellStats


class ShellController:
    """External shell statistics, certificate, and edit service."""

    def __init__(self, cfg: HiBaCaMLConfig, structure):
        self.cfg = cfg
        self.structure = structure
        self.column_nodes = structure.config["hibacaml"]["column_nodes"]
        kernel_dim = cfg.column_pool.memory_dim
        s1, s2, s3 = cfg.column_pool.shell_sizes
        self.shell_slices = {
            "kernel": slice(0, kernel_dim),
            "tier1": slice(kernel_dim, kernel_dim + s1),
            "tier2": slice(kernel_dim + s1, kernel_dim + s1 + s2),
            "tier3": slice(kernel_dim + s1 + s2, kernel_dim + s1 + s2 + s3),
        }
        self.inhibition_strengths = {
            "tier1": 0.08,
            "tier2": 0.18,
            "tier3": 0.30,
        }

    def update_shell_stats(
        self,
        persistent_state: PersistentHiBaCaMLState,
        graph_state: GraphState,
    ) -> None:
        """Update shell EMAs from the latest graph state."""
        for column_index, meta in enumerate(self.column_nodes):
            stats = persistent_state.shell_stats.setdefault(column_index, ShellStats())
            for shell_name, shell_slice in self.shell_slices.items():
                shell_values = [
                    graph_state.nodes[node_name].z_mu[..., shell_slice]
                    for node_name in self._shell_node_names(meta)
                ]
                act = float(
                    jnp.mean(
                        jnp.asarray(
                            [jnp.mean(jnp.abs(values)) for values in shell_values],
                            dtype=jnp.float32,
                        )
                    )
                )
                var = float(
                    jnp.mean(
                        jnp.asarray(
                            [jnp.mean(jnp.var(values, axis=0)) for values in shell_values],
                            dtype=jnp.float32,
                        )
                    )
                )
                old_act = stats.activation_ema.get(shell_name, 0.0)
                old_var = stats.task_variance_ema.get(shell_name, 0.0)
                stats.activation_ema[shell_name] = 0.9 * old_act + 0.1 * act
                stats.task_variance_ema[shell_name] = 0.9 * old_var + 0.1 * var
            stats.reuse_ema = 0.9 * stats.reuse_ema + 0.1 * (
                stats.activation_ema.get("kernel", 0.0) + stats.activation_ema.get("tier1", 0.0)
            ) / 2.0
            stats.specificity_ema = 0.9 * stats.specificity_ema + 0.1 * stats.task_variance_ema.get(
                "tier3", 0.0
            )

    def refresh_certificates(
        self,
        params: GraphParams,
        graph_state: GraphState,
        persistent_state: PersistentHiBaCaMLState,
    ) -> Dict[int, ColumnCertificate]:
        """Compute and store fresh column certificates."""
        self.update_shell_stats(persistent_state, graph_state)
        certificates: Dict[int, ColumnCertificate] = {}
        for column_index, meta in enumerate(self.column_nodes):
            certificates[column_index] = self._compute_column_certificate(
                column_index,
                meta,
                params,
                graph_state,
                persistent_state.shell_stats[column_index],
            )
        cert_vectors = {
            idx: jnp.asarray(
                [
                    cert.q_mean,
                    cert.prec_mean,
                    cert.pred_mean,
                    cert.live_frac,
                    cert.tier_occ[0],
                    cert.tier_occ[1],
                    cert.tier_occ[2],
                    cert.tier_q[0],
                    cert.tier_q[1],
                    cert.tier_q[2],
                    cert.shared_abstraction_mass,
                    cert.specificity_load,
                    cert.demotion_pressure,
                    cert.saturation,
                ],
                dtype=jnp.float32,
            )
            for idx, cert in certificates.items()
        }
        for idx, cert in certificates.items():
            signature = []
            base = cert_vectors[idx]
            base_norm = jnp.linalg.norm(base) + 1e-8
            for other_idx in range(self.cfg.column_pool.total_columns):
                other = cert_vectors[other_idx]
                sim = float(jnp.dot(base, other) / (base_norm * (jnp.linalg.norm(other) + 1e-8)))
                signature.append(sim)
            cert.similarity_signature = tuple(signature)
        persistent_state.certificates = certificates
        return certificates

    def semantic_penalty(
        self,
        persistent_state: PersistentHiBaCaMLState,
        active_columns: Sequence[int] | None = None,
    ) -> float:
        """Penalty for shell occupancy drift or broken q ordering."""
        occ_targets = self.cfg.exact_search.semantic_targets
        occ1 = []
        occ2 = []
        occ3 = []
        q1 = []
        q2 = []
        q3 = []
        if active_columns is None:
            certificates = list(persistent_state.certificates.values())
        else:
            certificates = [
                persistent_state.certificates[idx]
                for idx in active_columns
                if idx in persistent_state.certificates
            ]
        if not certificates:
            return 0.0
        for cert in certificates:
            occ1.append(cert.tier_occ[0])
            occ2.append(cert.tier_occ[1])
            occ3.append(cert.tier_occ[2])
            q1.append(cert.tier_q[0])
            q2.append(cert.tier_q[1])
            q3.append(cert.tier_q[2])
        occ_penalty = (
            abs(float(jnp.mean(jnp.asarray(occ1))) - occ_targets[0])
            + abs(float(jnp.mean(jnp.asarray(occ2))) - occ_targets[1])
            + abs(float(jnp.mean(jnp.asarray(occ3))) - occ_targets[2])
        )
        q_order_penalty = float(
            jnp.maximum(0.0, jnp.mean(jnp.asarray(q2)) - jnp.mean(jnp.asarray(q1)))
            + jnp.maximum(0.0, jnp.mean(jnp.asarray(q3)) - jnp.mean(jnp.asarray(q2)))
        )
        return occ_penalty + q_order_penalty

    def apply_structural_edits(
        self,
        params: GraphParams,
        graph_state: GraphState,
        persistent_state: PersistentHiBaCaMLState,
        active_columns: Sequence[int],
        phi,
    ) -> GraphParams:
        """Apply inhibition, prune, promote, and demote to active columns."""
        updated_nodes = dict(params.nodes)
        for column_index in active_columns:
            meta = self.column_nodes[column_index]
            for node_name in self._shell_node_names(meta):
                node_params = updated_nodes[node_name]
                node_state = graph_state.nodes[node_name]
                metrics = self._unit_metrics(node_params, node_state)
                updated_node = self._apply_inhibition(
                    node_params,
                    self._inhibition_adjustment(node_params),
                )

                if self._occupancy(updated_node, "tier3") > self.cfg.exact_search.semantic_targets[2]:
                    prune_idx = self._low_score_indices(metrics["tier3"]["score"], phi.outer_quantile)
                    updated_node = self._prune_units(updated_node, "tier3", prune_idx)

                if self._occupancy(updated_node, "tier2") > self.cfg.exact_search.semantic_targets[1]:
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

                if self.cfg.exact_search.allow_unaudited_demotion:
                    for inner_shell, outer_shell in (
                        ("kernel", "tier1"),
                        ("tier1", "tier2"),
                        ("tier2", "tier3"),
                    ):
                        inner_specificity = metrics[inner_shell]["specificity"]
                        if inner_specificity.size == 0:
                            continue
                        inner_idx = int(jnp.argmax(inner_specificity))
                        if float(inner_specificity[inner_idx]) <= phi.demotion_min_role_gain:
                            continue
                        outer_scores = metrics[outer_shell]["score"]
                        if outer_scores.size == 0:
                            continue
                        outer_idx = int(jnp.argmin(outer_scores))
                        updated_node = self._swap_units(
                            updated_node,
                            inner_shell,
                            inner_idx,
                            outer_shell,
                            outer_idx,
                        )

                updated_nodes[node_name] = updated_node

        params = params._replace(nodes=updated_nodes)
        self.refresh_certificates(params, graph_state, persistent_state)
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
            for node_name in self._shell_node_names(meta):
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

    def precision_weight_gradients(
        self,
        grads: GraphParams,
        params: GraphParams,
    ) -> GraphParams:
        """Precondition shell gradients by inverse effective precision."""
        if not self.cfg.exact_search.enable_precision_update_resistance:
            return grads
        strength = max(float(self.cfg.exact_search.precision_update_strength), 0.0)
        floor = max(float(self.cfg.exact_search.precision_update_floor), 0.0)
        updated_nodes = {}
        for node_name, node_grads in grads.nodes.items():
            node_params = params.nodes.get(node_name)
            if node_params is None:
                updated_nodes[node_name] = node_grads
                continue
            weights = dict(node_grads.weights)
            biases = dict(node_grads.biases)
            for shell_name in self.shell_slices:
                precision_key = f"log_precision_{shell_name}"
                if shell_name not in weights or precision_key not in node_params.biases:
                    continue
                precision = jax_nn_sigmoid(node_params.biases[precision_key])
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

    def _compute_column_certificate(
        self,
        column_index: int,
        meta: Dict[str, str],
        params: GraphParams,
        graph_state: GraphState,
        shell_stats: ShellStats,
    ) -> ColumnCertificate:
        q_means = []
        prec_means = []
        pred_means = []
        live_fracs = []
        tier_q = {"kernel": [], "tier1": [], "tier2": [], "tier3": []}
        tier_occ = {"kernel": [], "tier1": [], "tier2": [], "tier3": []}
        for node_name in self._shell_node_names(meta):
            node_params = params.nodes[node_name]
            node_state = graph_state.nodes[node_name]
            pred_means.append(float(jnp.mean(jnp.abs(node_state.z_mu))))
            live_fracs.append(
                np_mean(
                    [
                        self._occupancy(node_params, shell_name)
                        for shell_name in self.shell_slices
                    ]
                )
            )
            for shell_name in self.shell_slices:
                log_precision = node_params.biases[f"log_precision_{shell_name}"]
                q_val = float(jnp.mean(jax_nn_sigmoid(log_precision)))
                q_means.append(q_val)
                prec_means.append(float(jnp.mean(jnp.exp(log_precision))))
                tier_q[shell_name].append(q_val)
                tier_occ[shell_name].append(self._occupancy(node_params, shell_name))

        kernel_q_mean = float(np_mean(tier_q["kernel"]))
        tier1_mean = float(np_mean(tier_q["tier1"]))
        tier2_mean = float(np_mean(tier_q["tier2"]))
        tier3_mean = float(np_mean(tier_q["tier3"]))
        occ_tier1 = float(np_mean(tier_occ["tier1"]))
        occ_tier2 = float(np_mean(tier_occ["tier2"]))
        occ_tier3 = float(np_mean(tier_occ["tier3"]))
        shared_abstraction_mass = float(np_mean([kernel_q_mean, tier1_mean])) * shell_stats.reuse_ema
        specificity_load = tier3_mean * shell_stats.specificity_ema
        demotion_pressure = max(
            0.0,
            shell_stats.task_variance_ema.get("tier1", 0.0)
            - shell_stats.task_variance_ema.get("tier3", 0.0),
        )
        saturation = float(np_mean(live_fracs))
        return ColumnCertificate(
            column_index=column_index,
            q_mean=float(np_mean(q_means)),
            prec_mean=float(np_mean(prec_means)),
            pred_mean=float(np_mean(pred_means)),
            live_frac=float(np_mean(tier_occ["kernel"])),
            tier_q=(tier1_mean, tier2_mean, tier3_mean),
            tier_occ=(occ_tier1, occ_tier2, occ_tier3),
            shared_abstraction_mass=shared_abstraction_mass,
            specificity_load=specificity_load,
            demotion_pressure=demotion_pressure,
            saturation=saturation,
        )

    def _shell_node_names(self, meta: Dict[str, str]) -> List[str]:
        if "b_micro" in meta:
            kernel_names = list(meta.get("k_micro_depth", (meta["k_micro"],)))
            return [meta["b_micro"], *kernel_names, meta["l_micro"]]
        return [meta["bridge"], meta["kernel_0"], meta["kernel_1"], meta["lateral"]]

    def _occupancy(self, node_params: NodeParams, shell_name: str) -> float:
        log_precision = node_params.biases[f"log_precision_{shell_name}"]
        weight = node_params.weights[shell_name]
        live = (jnp.linalg.norm(weight, axis=0) > 1e-6) & (jax_nn_sigmoid(log_precision.squeeze(0).squeeze(0)) > 0.2)
        return float(jnp.mean(live.astype(jnp.float32)))

    def _unit_metrics(self, node_params: NodeParams, node_state) -> Dict[str, Dict[str, jnp.ndarray]]:
        metrics = {}
        for shell_name, shell_slice in self.shell_slices.items():
            weight = node_params.weights[shell_name]
            precision = jax_nn_sigmoid(node_params.biases[f"log_precision_{shell_name}"].reshape(-1))
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
            }
        return metrics

    def _redundancy_penalty(self, weight: jnp.ndarray) -> jnp.ndarray:
        if weight.shape[1] <= 1:
            return jnp.zeros((weight.shape[1],), dtype=weight.dtype)
        normed = weight / (jnp.linalg.norm(weight, axis=0, keepdims=True) + 1e-8)
        sim = jnp.matmul(normed.T, normed)
        sim = sim - jnp.eye(sim.shape[0], dtype=sim.dtype)
        return jnp.mean(jnp.maximum(sim, 0.0), axis=1)

    def _inhibition_adjustment(self, node_params: NodeParams) -> Dict[str, jnp.ndarray]:
        adjustment = {}
        for shell_name, strength in self.inhibition_strengths.items():
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


def jax_nn_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Small helper to avoid importing jax.nn in multiple places."""
    return 1.0 / (1.0 + jnp.exp(-x))


def np_mean(values: Sequence[float]) -> float:
    """Mean over Python sequences with an empty fallback."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))
