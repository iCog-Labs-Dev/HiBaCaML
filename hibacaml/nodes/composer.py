"""Column composition and final output integration."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec
from hibacaml.nodes.common import (
    apply_activation,
    finalize_state,
    slot_inputs,
    sum_inputs,
)


def _masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, *, axis: int = -1) -> jnp.ndarray:
    neg_inf = jnp.full_like(logits, -1e9)
    masked_logits = jnp.where(mask, logits, neg_inf)
    probs = jax.nn.softmax(masked_logits, axis=axis)
    probs = jnp.where(mask, probs, 0.0)
    denom = jnp.sum(probs, axis=axis, keepdims=True) + 1e-8
    return probs / denom


def _composer_gates(
    params: NodeParams,
    features: jnp.ndarray,
    certs: jnp.ndarray,
    query: jnp.ndarray,
    node_config: Dict[str, Any],
):
    """Shared composer core: gate probabilities, priors, and the correction vector."""
    active_mask = (jnp.linalg.norm(features, axis=-1) + jnp.linalg.norm(certs, axis=-1)) > 1e-8
    active_count = jnp.maximum(jnp.sum(active_mask.astype(features.dtype), axis=-1, keepdims=True), 1.0)
    uniform_prior = active_mask.astype(features.dtype) / active_count

    prior_raw = jnp.matmul(certs, params.weights["prior_proj"]).squeeze(-1)
    base_prior = _masked_softmax(prior_raw, active_mask)
    prior_mix_scale = float(node_config.get("prior_mix_scale", 0.0))
    prior_probs = (1.0 - prior_mix_scale) * base_prior + prior_mix_scale * uniform_prior
    prior_probs = prior_probs / (jnp.sum(prior_probs, axis=-1, keepdims=True) + 1e-8)
    prior_logits = jnp.log(prior_probs + 1e-8)

    q_context = (
        node_config.get("query_score_scale", 1.0)
        * jnp.matmul(query, params.weights["query_proj"])[:, None, :]
    )
    feature_hidden = jnp.tanh(jnp.matmul(features, params.weights["feature_proj"]) + q_context)
    residual_logits = jnp.matmul(feature_hidden, params.weights["attn_v"]).squeeze(-1)

    gate_logits = (
        node_config["prior_logit_scale"] * prior_logits
        + node_config["residual_gate_scale"] * residual_logits
    )
    topk = int(node_config.get("topk", 0))
    if 0 < topk < gate_logits.shape[-1]:
        topk_values, _ = jax.lax.top_k(jnp.where(active_mask, gate_logits, -1e9), topk)
        threshold = topk_values[..., -1:]
        active_mask = active_mask & (gate_logits >= threshold)
    gate_probs = _masked_softmax(gate_logits / node_config["gate_temp"], active_mask)

    weighted_features = jnp.sum(features * gate_probs[..., None], axis=1)
    correction = jnp.matmul(weighted_features, params.weights["out_proj"]) + params.biases["b_out"]
    return correction, gate_probs, prior_probs, active_mask


def composer_correction(
    params: NodeParams,
    features: jnp.ndarray,
    certs: jnp.ndarray,
    query: jnp.ndarray,
    node_config: Dict[str, Any],
) -> jnp.ndarray:
    """Correction vector only — the forward path needs nothing else."""
    correction, _, _, _ = _composer_gates(
        params, features, certs, query, node_config
    )
    return correction


def composer_details(
    params: NodeParams,
    features: jnp.ndarray,
    certs: jnp.ndarray,
    query: jnp.ndarray,
    node_config: Dict[str, Any],
) -> Dict[str, jnp.ndarray]:
    """Composer computation plus the auxiliary penalty and reported diagnostics."""
    correction, gate_probs, prior_probs, active_mask = _composer_gates(
        params, features, certs, query, node_config
    )

    gate_entropy = -jnp.sum(gate_probs * jnp.log(gate_probs + 1e-8), axis=-1)
    entropy_ceiling = (
        node_config.get("gate_entropy_ceiling_frac", 1.0)
        * jnp.log(jnp.maximum(jnp.sum(active_mask.astype(features.dtype), axis=-1), 1.0))
    )
    entropy_penalty = node_config.get("gate_entropy_ceiling_weight", 0.0) * jnp.maximum(
        0.0,
        gate_entropy - entropy_ceiling,
    )
    prior_kl = jnp.sum(
        gate_probs * (jnp.log(gate_probs + 1e-8) - jnp.log(prior_probs + 1e-8)),
        axis=-1,
    )
    prior_kl_penalty = node_config.get("prior_kl_weight", 0.0) * prior_kl
    gate_dev = jnp.mean(jnp.abs(gate_probs - prior_probs), axis=-1)
    gate_dev_penalty = node_config.get("gate_dev_weight", 0.0) * jnp.maximum(
        0.0,
        node_config.get("gate_dev_floor", 0.0) - gate_dev,
    )
    aux_penalty = prior_kl_penalty + entropy_penalty + gate_dev_penalty

    top1_mass = jnp.max(gate_probs, axis=-1)
    effective_k = jnp.exp(gate_entropy)

    return {
        "prior_probs": prior_probs,
        "gate_probs": gate_probs,
        "correction": correction,
        "aux_penalty": aux_penalty,
        "gate_entropy": gate_entropy,
        "prior_kl": prior_kl,
        "top1_mass": top1_mass,
        "effective_k": effective_k,
        "gate_dev": gate_dev,
    }


def _feature_edge_column_index(edge_key: str) -> int:
    match = re.search(r"col(\d+)/", edge_key)
    return int(match.group(1)) if match else 0


class ColumnComposerNode(NodeBase):
    """Attention-style composer over active column features."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(std=0.02),
        hidden_dim: int = 64,
        gate_temp: float = 0.52,
        prior_logit_scale: float = 0.68,
        prior_mix_scale: float = 0.16,
        residual_gate_scale: float = 2.35,
        query_score_scale: float = 1.0,
        prior_kl_weight: float = 0.0,
        gate_entropy_ceiling_frac: float = 1.0,
        gate_entropy_ceiling_weight: float = 0.0,
        gate_dev_floor: float = 0.0,
        gate_dev_weight: float = 0.0,
        topk: int = 0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            hidden_dim=hidden_dim,
            gate_temp=gate_temp,
            prior_logit_scale=prior_logit_scale,
            prior_mix_scale=prior_mix_scale,
            residual_gate_scale=residual_gate_scale,
            query_score_scale=query_score_scale,
            prior_kl_weight=prior_kl_weight,
            gate_entropy_ceiling_frac=gate_entropy_ceiling_frac,
            gate_entropy_ceiling_weight=gate_entropy_ceiling_weight,
            gate_dev_floor=gate_dev_floor,
            gate_dev_weight=gate_dev_weight,
            topk=topk,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "feature": SlotSpec(name="feature", is_multi_input=True),
            "cert": SlotSpec(name="cert", is_multi_input=True),
            "query": SlotSpec(name="query", is_multi_input=False, is_variance_scalable=False),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(std=0.02)

        feature_dim = None
        cert_dim = None
        query_dim = None
        for edge_key, shape in input_shapes.items():
            if ":feature" in edge_key and feature_dim is None:
                feature_dim = int(shape[-1])
            elif ":cert" in edge_key and cert_dim is None:
                cert_dim = int(shape[-1])
            elif ":query" in edge_key and query_dim is None:
                query_dim = int(shape[-1])

        if feature_dim is None or cert_dim is None or query_dim is None:
            raise ValueError("ColumnComposerNode requires feature, cert, and query inputs")

        hidden_dim = int(config["hidden_dim"])
        keys = jax.random.split(key, 5)
        weights = {
            "prior_proj": initialize(keys[0], (cert_dim, 1), weight_init),
            "feature_proj": initialize(keys[1], (feature_dim, hidden_dim), weight_init),
            "query_proj": initialize(keys[2], (query_dim, hidden_dim), weight_init),
            "attn_v": initialize(keys[3], (hidden_dim, 1), weight_init),
            "out_proj": initialize(keys[4], (feature_dim, node_shape[-1]), weight_init),
        }
        biases = {
            "b_out": jnp.zeros((1, node_shape[-1])),
        }
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        feature_edges = slot_inputs(inputs, "feature")
        cert_edges = slot_inputs(inputs, "cert")
        query = sum_inputs(slot_inputs(inputs, "query"))
        if not feature_edges or not cert_edges or query is None:
            raise ValueError(f"{node_info.name} requires feature, cert, and query inputs")

        feature_keys = sorted(feature_edges, key=_feature_edge_column_index)
        cert_keys = sorted(cert_edges, key=_feature_edge_column_index)
        features = jnp.stack([feature_edges[key] for key in feature_keys], axis=1)
        certs = jnp.stack([cert_edges[key] for key in cert_keys], axis=1)
        pre_activation = composer_correction(
            params,
            features,
            certs,
            query,
            node_info.node_config,
        )
        z_mu = apply_activation(node_info, pre_activation)
        return finalize_state(state, node_info, z_mu)


class ScaledAddNode(NodeBase):
    """Combine base logits with a scaled correction branch."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        correction_scale: float = 1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            correction_scale=correction_scale,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "base": SlotSpec(name="base", is_multi_input=True),
            "correction": SlotSpec(name="correction", is_multi_input=True),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        return NodeParams(weights={}, biases={})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        base = sum_inputs(slot_inputs(inputs, "base"))
        correction = sum_inputs(slot_inputs(inputs, "correction"))
        if base is None:
            raise ValueError(f"{node_info.name} requires at least one base input")
        if correction is None:
            correction = jnp.zeros_like(base)

        pre_activation = base + node_info.node_config["correction_scale"] * correction
        z_mu = apply_activation(node_info, pre_activation)
        return finalize_state(state, node_info, z_mu)
