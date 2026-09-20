"""Collect shell statistics and generate column certificates for HiBaCaML."""

from __future__ import annotations

from typing import Dict, List, Sequence

import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphState, NodeParams
from hibacaml.config import HiBaCaMLConfig
from hibacaml.types import ColumnCertificate, PersistentHiBaCaMLState, ShellStats


class CertificateController:
    """Shell statistics, column certificates, and semantic scoring."""

    def __init__(self, cfg: HiBaCaMLConfig, structure):
        self.cfg = cfg
        self.structure = structure
        self.column_nodes = structure.config["hibacaml"]["column_nodes"]
        self.shell_slices = shell_slices(cfg)

    def refresh_certificates(
        self,
        params: GraphParams,
        graph_state: GraphState,
        persistent_state: PersistentHiBaCaMLState,
    ) -> Dict[int, ColumnCertificate]:
        """Compute and store fresh column certificates."""
        self._update_shell_stats(persistent_state, graph_state)
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

    def certificate_vectors(
        self,
        persistent_state: PersistentHiBaCaMLState,
    ) -> Dict[int, jnp.ndarray]:
        """Return the unmasked per-column certificate vectors.

        Split out so callers that score many supports against one certificate
        state build these once instead of once per support mask.
        """
        vectors = {}
        for idx in range(self.cfg.column_pool.total_columns):
            cert = persistent_state.certificates.get(idx)
            if cert is None:
                vectors[idx] = jnp.zeros((self.cfg.composer_cert_dim,), dtype=jnp.float32)
                continue
            vectors[idx] = jnp.asarray(
                [
                    cert.q_mean,
                    cert.prec_mean,
                    cert.pred_mean,
                    cert.live_frac,
                    cert.tier_q[0],
                    cert.tier_q[1],
                    cert.tier_q[2],
                    cert.shared_abstraction_mass,
                    cert.specificity_load,
                    cert.demotion_pressure,
                ],
                dtype=jnp.float32,
            )
        return vectors

    @staticmethod
    def mask_certificate_vectors(
        vectors: Dict[int, jnp.ndarray],
        support_mask: jnp.ndarray,
    ) -> Dict[int, jnp.ndarray]:
        """Apply one support mask to precomputed certificate vectors."""
        return {idx: vec * support_mask[idx] for idx, vec in vectors.items()}

    def certificate_matrix(
        self,
        persistent_state: PersistentHiBaCaMLState,
        support_mask: jnp.ndarray,
    ) -> Dict[int, jnp.ndarray]:
        """Return per-column certificate vectors masked by active support."""
        return self.mask_certificate_vectors(
            self.certificate_vectors(persistent_state),
            support_mask,
        )

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

    def _update_shell_stats(
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
                    for node_name in shell_node_names(meta)
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
        mean_occupancies = []
        tier_q = {"kernel": [], "tier1": [], "tier2": [], "tier3": []}
        tier_occ = {"kernel": [], "tier1": [], "tier2": [], "tier3": []}
        for node_name in shell_node_names(meta):
            node_params = params.nodes[node_name]
            node_state = graph_state.nodes[node_name]
            pred_means.append(float(jnp.mean(jnp.abs(node_state.z_mu))))
            occupancies = {
                shell_name: shell_occupancy(node_params, shell_name)
                for shell_name in self.shell_slices
            }
            mean_occupancies.append(mean_or_zero(list(occupancies.values())))
            for shell_name in self.shell_slices:
                log_precision = node_params.biases[f"log_precision_{shell_name}"]
                q_val = float(jnp.mean(effective_precision(log_precision)))
                q_means.append(q_val)
                prec_means.append(float(jnp.mean(jnp.exp(log_precision))))
                tier_q[shell_name].append(q_val)
                tier_occ[shell_name].append(occupancies[shell_name])

        kernel_q_mean = float(mean_or_zero(tier_q["kernel"]))
        tier1_mean = float(mean_or_zero(tier_q["tier1"]))
        tier2_mean = float(mean_or_zero(tier_q["tier2"]))
        tier3_mean = float(mean_or_zero(tier_q["tier3"]))
        occ_tier1 = float(mean_or_zero(tier_occ["tier1"]))
        occ_tier2 = float(mean_or_zero(tier_occ["tier2"]))
        occ_tier3 = float(mean_or_zero(tier_occ["tier3"]))
        shared_abstraction_mass = float(mean_or_zero([kernel_q_mean, tier1_mean])) * shell_stats.reuse_ema
        specificity_load = tier3_mean * shell_stats.specificity_ema
        demotion_pressure = max(
            0.0,
            shell_stats.task_variance_ema.get("tier1", 0.0)
            - shell_stats.task_variance_ema.get("tier3", 0.0),
        )
        saturation = float(mean_or_zero(mean_occupancies))
        return ColumnCertificate(
            column_index=column_index,
            q_mean=float(mean_or_zero(q_means)),
            prec_mean=float(mean_or_zero(prec_means)),
            pred_mean=float(mean_or_zero(pred_means)),
            live_frac=float(mean_or_zero(tier_occ["kernel"])),
            tier_q=(tier1_mean, tier2_mean, tier3_mean),
            tier_occ=(occ_tier1, occ_tier2, occ_tier3),
            shared_abstraction_mass=shared_abstraction_mass,
            specificity_load=specificity_load,
            demotion_pressure=demotion_pressure,
            saturation=saturation,
        )


# ----------------------------------------------------------------------
# Shell geometry, shared with shells.py
# ----------------------------------------------------------------------
def shell_slices(cfg: HiBaCaMLConfig) -> Dict[str, slice]:
    """Where each shell lives inside a shell-bank node's feature axis."""
    kernel_dim = cfg.column_pool.memory_dim
    s1, s2, s3 = cfg.column_pool.shell_sizes
    return {
        "kernel": slice(0, kernel_dim),
        "tier1": slice(kernel_dim, kernel_dim + s1),
        "tier2": slice(kernel_dim + s1, kernel_dim + s1 + s2),
        "tier3": slice(kernel_dim + s1 + s2, kernel_dim + s1 + s2 + s3),
    }


def shell_node_names(meta: Dict[str, str]) -> List[str]:
    if "b_micro" in meta:
        kernel_names = list(meta.get("k_micro_depth", (meta["k_micro"],)))
        return [meta["b_micro"], *kernel_names, meta["l_micro"]]
    return [meta["bridge"], meta["kernel_0"], meta["kernel_1"], meta["lateral"]]


def shell_occupancy(node_params: NodeParams, shell_name: str) -> float:
    log_precision = node_params.biases[f"log_precision_{shell_name}"]
    weight = node_params.weights[shell_name]
    live = (jnp.linalg.norm(weight, axis=0) > 1e-6) & (
        effective_precision(log_precision.squeeze(0).squeeze(0)) > 0.2
    )
    return float(jnp.mean(live.astype(jnp.float32)))


def effective_precision(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid of a log-precision bias, kept in one place."""
    return 1.0 / (1.0 + jnp.exp(-x))


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean over Python sequences with an empty fallback."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))
