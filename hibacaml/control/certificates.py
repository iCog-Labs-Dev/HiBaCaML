"""Collect shell statistics and generate column certificates for HiBaCaML."""

from __future__ import annotations

from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.types import GraphParams, GraphState
from hibacaml.config import HiBaCaMLConfig
from hibacaml.types import ColumnCertificate, PersistentHiBaCaMLState, ShellStats


_CERTIFICATE_FIELDS = (
    "q_mean",
    "prec_mean",
    "pred_mean",
    "live_frac",
    "tier_occ0",
    "tier_occ1",
    "tier_occ2",
    "tier_q0",
    "tier_q1",
    "tier_q2",
    "shared_abstraction_mass",
    "specificity_load",
    "demotion_pressure",
    "saturation",
)

_COMPOSER_FIELDS = (
    "q_mean",
    "prec_mean",
    "pred_mean",
    "live_frac",
    "tier_q0",
    "tier_q1",
    "tier_q2",
    "shared_abstraction_mass",
    "specificity_load",
    "demotion_pressure",
)

_COMPOSER_GATHER = tuple(
    _CERTIFICATE_FIELDS.index(name) for name in _COMPOSER_FIELDS
)


def _project_observations(
    params,
    graph_state,
    column_node_names,
    shell_names,
    shell_bounds,
):
    gate = []
    scale = []
    occupancy = []
    prediction = []
    activation = []
    variance = []

    for node_names in column_node_names:
        column_gate = []
        column_scale = []
        column_occupancy = []
        column_prediction = []
        column_activation = []
        column_variance = []
        for node_name in node_names:
            node_params = params.nodes[node_name]
            z_mu = graph_state.nodes[node_name].z_mu
            column_prediction.append(jnp.mean(jnp.abs(z_mu)))

            node_gate = []
            node_scale = []
            node_occupancy = []
            node_activation = []
            node_variance = []
            for shell_name, (start, stop) in zip(shell_names, shell_bounds):
                log_precision = node_params.biases[f"log_precision_{shell_name}"]
                weight = node_params.weights[shell_name]
                shell_values = z_mu[..., start:stop]
                precision = 1.0 / (1.0 + jnp.exp(-log_precision))

                node_gate.append(jnp.mean(precision))
                node_scale.append(jnp.mean(jnp.exp(log_precision)))
                live = (jnp.linalg.norm(weight, axis=0) > 1e-6) & (
                    precision.squeeze(0).squeeze(0) > 0.2
                )
                node_occupancy.append(jnp.mean(live.astype(jnp.float32)))
                node_activation.append(jnp.mean(jnp.abs(shell_values)))
                node_variance.append(jnp.mean(jnp.var(shell_values, axis=0)))

            column_gate.append(jnp.stack(node_gate))
            column_scale.append(jnp.stack(node_scale))
            column_occupancy.append(jnp.stack(node_occupancy))
            column_activation.append(jnp.stack(node_activation))
            column_variance.append(jnp.stack(node_variance))

        gate.append(jnp.stack(column_gate))
        scale.append(jnp.stack(column_scale))
        occupancy.append(jnp.stack(column_occupancy))
        prediction.append(jnp.stack(column_prediction))
        activation.append(jnp.mean(jnp.stack(column_activation), axis=0))
        variance.append(jnp.mean(jnp.stack(column_variance), axis=0))

    return (
        jnp.stack(gate),
        jnp.stack(scale),
        jnp.stack(occupancy),
        jnp.stack(prediction),
        jnp.stack(activation),
        jnp.stack(variance),
    )


def _advance_shell_stats(observations, shell_stats, shell_names):
    activation = np.asarray(observations[4])
    variance = np.asarray(observations[5])

    for column_index in range(activation.shape[0]):
        stats = shell_stats[column_index]
        for shell_index, shell_name in enumerate(shell_names):
            old_activation = stats.activation_ema.get(shell_name, 0.0)
            old_variance = stats.task_variance_ema.get(shell_name, 0.0)
            stats.activation_ema[shell_name] = (
                0.9 * old_activation
                + 0.1 * float(activation[column_index, shell_index])
            )
            stats.task_variance_ema[shell_name] = (
                0.9 * old_variance
                + 0.1 * float(variance[column_index, shell_index])
            )
        stats.reuse_ema = 0.9 * stats.reuse_ema + 0.1 * (
            stats.activation_ema.get("kernel", 0.0)
            + stats.activation_ema.get("tier1", 0.0)
        ) / 2.0
        stats.specificity_ema = (
            0.9 * stats.specificity_ema
            + 0.1 * stats.task_variance_ema.get("tier3", 0.0)
        )


def _column_certificate(column_index, observations, stats, shell_names):
    gate = np.asarray(observations[0][column_index])
    scale = np.asarray(observations[1][column_index])
    occupancy = np.asarray(observations[2][column_index])
    prediction = np.asarray(observations[3][column_index])
    shell_index = {name: index for index, name in enumerate(shell_names)}

    def shell_values(values, shell_name):
        return [float(value) for value in values[:, shell_index[shell_name]]]

    q_means = [float(value) for value in gate.reshape(-1)]
    precision_means = [float(value) for value in scale.reshape(-1)]
    prediction_means = [float(value) for value in prediction]
    mean_occupancies = [
        mean_or_zero([float(value) for value in row]) for row in occupancy
    ]

    kernel_q = mean_or_zero(shell_values(gate, "kernel"))
    tier1_q = mean_or_zero(shell_values(gate, "tier1"))
    tier2_q = mean_or_zero(shell_values(gate, "tier2"))
    tier3_q = mean_or_zero(shell_values(gate, "tier3"))
    return ColumnCertificate(
        column_index=column_index,
        q_mean=mean_or_zero(q_means),
        prec_mean=mean_or_zero(precision_means),
        pred_mean=mean_or_zero(prediction_means),
        live_frac=mean_or_zero(shell_values(occupancy, "kernel")),
        tier_q=(tier1_q, tier2_q, tier3_q),
        tier_occ=(
            mean_or_zero(shell_values(occupancy, "tier1")),
            mean_or_zero(shell_values(occupancy, "tier2")),
            mean_or_zero(shell_values(occupancy, "tier3")),
        ),
        shared_abstraction_mass=(
            mean_or_zero([kernel_q, tier1_q]) * stats.reuse_ema
        ),
        specificity_load=tier3_q * stats.specificity_ema,
        demotion_pressure=max(
            0.0,
            stats.task_variance_ema.get("tier1", 0.0)
            - stats.task_variance_ema.get("tier3", 0.0),
        ),
        saturation=mean_or_zero(mean_occupancies),
    )


def _certificate_values(certificate):
    return (
        certificate.q_mean,
        certificate.prec_mean,
        certificate.pred_mean,
        certificate.live_frac,
        *certificate.tier_occ,
        *certificate.tier_q,
        certificate.shared_abstraction_mass,
        certificate.specificity_load,
        certificate.demotion_pressure,
        certificate.saturation,
    )


def _similarity_matrix(values):
    rows = np.asarray(values, dtype=np.float32)
    norms = (
        np.linalg.norm(rows, axis=1).astype(np.float32)
        + np.float32(1e-8)
    )
    return (rows @ rows.T) / np.outer(norms, norms)


class CertificateController:
    """Shell statistics, column certificates, and semantic scoring."""

    def __init__(self, cfg: HiBaCaMLConfig, structure):
        self.cfg = cfg
        self.structure = structure
        self.column_nodes = structure.config["hibacaml"]["column_nodes"]
        self.shell_slices = shell_slices(cfg)
        self._shell_names = tuple(self.shell_slices)
        self._shell_bounds = tuple(
            (int(shell_slice.start), int(shell_slice.stop))
            for shell_slice in self.shell_slices.values()
        )
        self._column_node_names = tuple(
            tuple(shell_node_names(meta)) for meta in self.column_nodes
        )
        self._certificate_stack_cache = None

    def refresh_certificates(
        self,
        params: GraphParams,
        graph_state: GraphState,
        persistent_state: PersistentHiBaCaMLState,
    ) -> Dict[int, ColumnCertificate]:
        """Compute and store fresh column certificates."""
        for column_index in range(len(self.column_nodes)):
            persistent_state.shell_stats.setdefault(column_index, ShellStats())

        observations = jax.device_get(
            _project_observations(
                params,
                graph_state,
                self._column_node_names,
                self._shell_names,
                self._shell_bounds,
            )
        )
        _advance_shell_stats(
            observations,
            persistent_state.shell_stats,
            self._shell_names,
        )
        certificates = {
            column_index: _column_certificate(
                column_index,
                observations,
                persistent_state.shell_stats[column_index],
                self._shell_names,
            )
            for column_index in range(len(self.column_nodes))
        }
        similarities = _similarity_matrix(
            [_certificate_values(certificates[index]) for index in certificates]
        )
        for column_index, certificate in certificates.items():
            certificate.similarity_signature = tuple(
                float(value) for value in similarities[column_index]
            )
        persistent_state.certificates = certificates
        persistent_state.certificates_revision = (
            int(getattr(persistent_state, "certificates_revision", 0)) + 1
        )
        self._certificate_stack_cache = None
        return certificates

    def _certificate_stack(
        self,
        persistent_state: PersistentHiBaCaMLState,
    ) -> jnp.ndarray:
        revision = int(getattr(persistent_state, "certificates_revision", 0))
        cached = self._certificate_stack_cache
        if (
            cached is not None
            and cached[0] is persistent_state
            and cached[1] == revision
        ):
            return cached[2]

        rows = []
        for column_index in range(self.cfg.column_pool.total_columns):
            certificate = persistent_state.certificates.get(column_index)
            if certificate is None:
                values = (0.0,) * len(_CERTIFICATE_FIELDS)
            else:
                values = _certificate_values(certificate)
            rows.append([values[index] for index in _COMPOSER_GATHER])
        stack = jnp.asarray(rows, dtype=jnp.float32)
        self._certificate_stack_cache = (persistent_state, revision, stack)
        return stack

    def certificate_vectors(
        self,
        persistent_state: PersistentHiBaCaMLState,
    ) -> Dict[int, jnp.ndarray]:
        """Return the unmasked per-column certificate vectors.

        Retained for the frozen control baseline capture.
        """
        stack = self._certificate_stack(persistent_state)
        return {
            index: stack[index]
            for index in range(self.cfg.column_pool.total_columns)
        }

    def certificate_stack_for_masks(
        self,
        persistent_state: PersistentHiBaCaMLState,
        support_masks: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return (supports, columns, fields) certificate inputs for stacked supports.

        Deliberately eager: inside a compiled program XLA may contract this
        multiply into a downstream FMA, moving values the controller compares.
        """
        stack = self._certificate_stack(persistent_state)
        return support_masks[:, :, None] * stack[None, :, :]

    def certificate_matrix(
        self,
        persistent_state: PersistentHiBaCaMLState,
        support_mask: jnp.ndarray,
    ) -> Dict[int, jnp.ndarray]:
        """Return per-column certificate vectors masked by active support."""
        masked = self._certificate_stack(persistent_state) * support_mask[:, None]
        return {
            index: masked[index]
            for index in range(self.cfg.column_pool.total_columns)
        }

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


def effective_precision(x: jnp.ndarray) -> jnp.ndarray:
    """Sigmoid of a log-precision bias, kept in one place."""
    return 1.0 / (1.0 + jnp.exp(-x))


def mean_or_zero(values: Sequence[float]) -> float:
    """Mean over Python sequences with an empty fallback."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))
