"""Core HiBaCaML node implementations."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, ZerosInitializer, initialize
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec


def _apply_activation(node_info: NodeInfo, x: jnp.ndarray) -> jnp.ndarray:
    activation = node_info.activation
    return type(activation).forward(x, activation.config)


def _slot_inputs(inputs: Dict[str, jnp.ndarray], slot_name: str) -> Dict[str, jnp.ndarray]:
    token = f":{slot_name}"
    return {k: v for k, v in inputs.items() if token in k}


def _sum_inputs(inputs: Dict[str, jnp.ndarray]) -> Optional[jnp.ndarray]:
    out = None
    for value in inputs.values():
        out = value if out is None else out + value
    return out


def _shell_dims(config: Dict[str, Any]) -> Dict[str, int]:
    shell_sizes = tuple(config["shell_sizes"])
    return {
        "kernel": int(config["kernel_dim"]),
        "tier1": int(shell_sizes[0]),
        "tier2": int(shell_sizes[1]),
        "tier3": int(shell_sizes[2]),
    }


def _shell_slices(config: Dict[str, Any]) -> Dict[str, slice]:
    dims = _shell_dims(config)
    start = 0
    out: Dict[str, slice] = {}
    for shell_name, shell_dim in dims.items():
        out[shell_name] = slice(start, start + shell_dim)
        start += shell_dim
    return out


def _shell_concat(parts: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.concatenate(
        [parts["kernel"], parts["tier1"], parts["tier2"], parts["tier3"]], axis=-1
    )


def _precision_scale(params: NodeParams, shell_name: str) -> jnp.ndarray:
    key = f"log_precision_{shell_name}"
    return jax.nn.sigmoid(params.biases[key])


def _precision_mean(params: NodeParams, shell_name: str) -> jnp.ndarray:
    key = f"log_precision_{shell_name}"
    return jnp.exp(params.biases[key])


def _masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray, *, axis: int = -1) -> jnp.ndarray:
    neg_inf = jnp.full_like(logits, -1e9)
    masked_logits = jnp.where(mask, logits, neg_inf)
    probs = jax.nn.softmax(masked_logits, axis=axis)
    probs = jnp.where(mask, probs, 0.0)
    denom = jnp.sum(probs, axis=axis, keepdims=True) + 1e-8
    return probs / denom


def composer_stage2_details(
    params: NodeParams,
    features: jnp.ndarray,
    query: jnp.ndarray,
    node_config: Dict[str, Any],
    cert_prior: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """Attention-style composer computation."""
    active_mask = jnp.linalg.norm(features, axis=-1) > 1e-8
    q_context = (
        node_config.get("query_score_scale", 1.0)
        * jnp.matmul(query, params.weights["query_proj"])[:, None, :]
    )
    feature_hidden = jnp.tanh(jnp.matmul(features, params.weights["feature_proj"]) + q_context)
    gate_logits = jnp.matmul(feature_hidden, params.weights["attn_v"]).squeeze(-1)
    if cert_prior is not None:
        gate_logits = gate_logits + node_config.get("cert_prior_scale", 0.0) * cert_prior
    active_top_k = int(node_config.get("active_top_k", features.shape[1]))
    if 0 < active_top_k < features.shape[1]:
        masked_logits = jnp.where(active_mask, gate_logits, -1e9)
        _, top_indices = jax.lax.top_k(masked_logits, active_top_k)
        top_mask = jnp.zeros_like(active_mask, dtype=bool)
        row_indices = jnp.arange(features.shape[0])[:, None]
        top_mask = top_mask.at[row_indices, top_indices].set(True)
        active_mask = active_mask & top_mask
    gate_probs = _masked_softmax(gate_logits, active_mask)

    weighted_features = jnp.sum(features * gate_probs[..., None], axis=1)
    correction = jnp.matmul(weighted_features, params.weights["out_proj"]) + params.biases["b_out"]
    active_count = jnp.maximum(jnp.sum(active_mask.astype(jnp.float32), axis=-1), 1.0)
    uniform = jnp.where(active_mask, 1.0 / active_count[:, None], 0.0)
    gate_entropy = -jnp.sum(gate_probs * jnp.log(gate_probs + 1e-8), axis=-1)
    target_entropy = jnp.log(active_count + 1e-8)
    gate_deviation = jnp.sum(jnp.square(gate_probs - uniform), axis=-1)
    return {
        "gate_probs": gate_probs,
        "correction": correction,
        "gate_entropy": gate_entropy,
        "gate_entropy_deficit": jnp.maximum(0.0, target_entropy - gate_entropy),
        "gate_deviation": gate_deviation,
    }


def _feature_edge_column_index(edge_key: str) -> int:
    match = re.search(r"col(\d+)/", edge_key)
    return int(match.group(1)) if match else 0


class PatchTokenizerNode(NodeBase):
    """Convert an image into patch tokens with learned patch embeddings."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(std=0.02),
        patch_size: Tuple[int, int] = (7, 7),
        patch_embed_dim: int = 12,
        coord_dim: int = 2,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            patch_size=patch_size,
            patch_embed_dim=patch_embed_dim,
            coord_dim=coord_dim,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=False)}

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
        patch_h, patch_w = config["patch_size"]
        embed_dim = int(config["patch_embed_dim"])
        input_shape = next(iter(input_shapes.values()))
        in_dim = int(np.prod((patch_h, patch_w, input_shape[-1])))
        proj = initialize(key, (in_dim, embed_dim), weight_init)
        bias = jnp.zeros((1, 1, embed_dim))
        return NodeParams(weights={"proj": proj}, biases={"b_proj": bias})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = _sum_inputs(inputs)
        if x is None:
            raise ValueError(f"{node_info.name} requires an input image tensor")

        patch_h, patch_w = node_info.node_config["patch_size"]
        coord_dim = int(node_info.node_config["coord_dim"])
        batch, height, width, channels = x.shape
        nh = height // patch_h
        nw = width // patch_w

        patches = x.reshape(batch, nh, patch_h, nw, patch_w, channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5).reshape(
            batch, nh * nw, patch_h * patch_w * channels
        )
        embedded = jnp.matmul(patches, params.weights["proj"]) + params.biases["b_proj"]

        ys = jnp.linspace(0.0, 1.0, nh, dtype=embedded.dtype)
        xs = jnp.linspace(0.0, 1.0, nw, dtype=embedded.dtype)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        coords = jnp.stack([yy, xx], axis=-1).reshape(1, nh * nw, 2)
        coords = jnp.broadcast_to(coords, (batch, nh * nw, 2))
        if coord_dim != 2:
            coords = jnp.pad(coords, ((0, 0), (0, 0), (0, max(0, coord_dim - 2))))[
                :, :, :coord_dim
            ]

        pre_activation = jnp.concatenate([embedded, coords], axis=-1)
        z_mu = _apply_activation(node_info, pre_activation)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class ShellBankInputNode(NodeBase):
    """Project tokens into shell-aware column features."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(std=0.05),
        kernel_dim: int = 8,
        shell_sizes: Tuple[int, int, int] = (4, 6, 8),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_dim=kernel_dim,
            shell_sizes=shell_sizes,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

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
            weight_init = NormalInitializer(std=0.05)

        in_shape = next(iter(input_shapes.values()))
        in_dim = int(in_shape[-1])
        dims = _shell_dims(config)
        keys = jax.random.split(key, len(dims))
        weights = {
            shell_name: initialize(keys[idx], (in_dim, shell_dim), weight_init)
            for idx, (shell_name, shell_dim) in enumerate(dims.items())
        }
        biases = {}
        for shell_name, shell_dim in dims.items():
            biases[f"b_{shell_name}"] = jnp.zeros((1, 1, shell_dim))
            biases[f"log_precision_{shell_name}"] = jnp.zeros((1, 1, shell_dim))
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        x = _sum_inputs(_slot_inputs(inputs, "in"))
        if x is None:
            raise ValueError(f"{node_info.name} requires at least one 'in' input")

        shell_parts = {}
        pre_parts = {}
        for shell_name, shell_dim in _shell_dims(node_info.node_config).items():
            pre = jnp.matmul(x, params.weights[shell_name]) + params.biases[f"b_{shell_name}"]
            activated = _apply_activation(node_info, pre)
            shell_parts[shell_name] = activated * _precision_scale(params, shell_name)
            pre_parts[shell_name] = pre

        pre_activation = _shell_concat(pre_parts)
        z_mu = _shell_concat(shell_parts)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class ShellBankResidualNode(NodeBase):
    """Residual shell-bank node with separate transform and skip inputs."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(std=0.05),
        kernel_dim: int = 8,
        shell_sizes: Tuple[int, int, int] = (4, 6, 8),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_dim=kernel_dim,
            shell_sizes=shell_sizes,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(name="in", is_multi_input=True),
            "skip": SlotSpec(
                name="skip",
                is_multi_input=True,
                is_skip_connection=True,
                is_variance_scalable=False,
            ),
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
            weight_init = NormalInitializer(std=0.05)

        in_shape = None
        for edge_key, shape in input_shapes.items():
            if ":in" in edge_key:
                in_shape = shape
                break
        if in_shape is None:
            raise ValueError("ShellBankResidualNode requires an 'in' edge")
        in_dim = int(in_shape[-1])
        dims = _shell_dims(config)
        keys = jax.random.split(key, len(dims))
        weights = {
            shell_name: initialize(keys[idx], (in_dim, shell_dim), weight_init)
            for idx, (shell_name, shell_dim) in enumerate(dims.items())
        }
        biases = {}
        for shell_name, shell_dim in dims.items():
            biases[f"b_{shell_name}"] = jnp.zeros((1, 1, shell_dim))
            biases[f"log_precision_{shell_name}"] = jnp.zeros((1, 1, shell_dim))
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        transformed_input = _sum_inputs(_slot_inputs(inputs, "in"))
        if transformed_input is None:
            raise ValueError(f"{node_info.name} requires an 'in' input")

        skip_input = _sum_inputs(_slot_inputs(inputs, "skip"))
        shell_parts = {}
        pre_parts = {}
        for shell_name in _shell_dims(node_info.node_config):
            pre = jnp.matmul(transformed_input, params.weights[shell_name])
            pre = pre + params.biases[f"b_{shell_name}"]
            activated = _apply_activation(node_info, pre)
            shell_parts[shell_name] = activated * _precision_scale(params, shell_name)
            pre_parts[shell_name] = pre

        projected = _shell_concat(shell_parts)
        pre_activation = _shell_concat(pre_parts)
        z_mu = projected if skip_input is None else projected + skip_input
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class ShellBankRecurrentNode(NodeBase):
    """Shell-bank node with recurrent self-dynamics across settling steps."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=ZerosInitializer(),
        weight_init=NormalInitializer(std=0.05),
        kernel_dim: int = 8,
        shell_sizes: Tuple[int, int, int] = (4, 6, 8),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_dim=kernel_dim,
            shell_sizes=shell_sizes,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

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
            weight_init = NormalInitializer(std=0.05)

        in_shape = next(iter(input_shapes.values()))
        in_dim = int(in_shape[-1])
        dims = _shell_dims(config)
        keys = jax.random.split(key, len(dims) * 2)
        weights = {}
        for idx, (shell_name, shell_dim) in enumerate(dims.items()):
            weights[shell_name] = initialize(keys[idx], (in_dim, shell_dim), weight_init)
            weights[f"recur_{shell_name}"] = initialize(
                keys[len(dims) + idx],
                (shell_dim, shell_dim),
                weight_init,
            )
        biases = {}
        for shell_name, shell_dim in dims.items():
            biases[f"b_{shell_name}"] = jnp.zeros((1, 1, shell_dim))
            biases[f"log_precision_{shell_name}"] = jnp.zeros((1, 1, shell_dim))
        return NodeParams(weights=weights, biases=biases)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jax.Array, NodeState]:
        transformed_input = _sum_inputs(_slot_inputs(inputs, "in"))
        if transformed_input is None:
            raise ValueError(f"{node_info.name} requires an 'in' input")

        shell_parts = {}
        pre_parts = {}
        shell_slices = _shell_slices(node_info.node_config)
        for shell_name in _shell_dims(node_info.node_config):
            latent_slice = state.z_latent[..., shell_slices[shell_name]]
            pre = jnp.matmul(transformed_input, params.weights[shell_name])
            pre = pre + jnp.matmul(latent_slice, params.weights[f"recur_{shell_name}"])
            pre = pre + params.biases[f"b_{shell_name}"]
            activated = _apply_activation(node_info, pre)
            shell_parts[shell_name] = activated * _precision_scale(params, shell_name)
            pre_parts[shell_name] = pre

        pre_activation = _shell_concat(pre_parts)
        z_mu = _shell_concat(shell_parts)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class ElementwiseGateNode(NodeBase):
    """Multiply a value input by a selected scalar support gate."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        gate_index: int = 0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            gate_index=gate_index,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "value": SlotSpec(name="value", is_multi_input=False),
            "gate": SlotSpec(name="gate", is_multi_input=False, is_variance_scalable=False),
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
    ) -> Tuple[jax.Array, NodeState]:
        value = _sum_inputs(_slot_inputs(inputs, "value"))
        gate = _sum_inputs(_slot_inputs(inputs, "gate"))
        if value is None or gate is None:
            raise ValueError(f"{node_info.name} requires 'value' and 'gate' inputs")

        gate_index = int(node_info.node_config["gate_index"])
        gate_scalar = gate[..., gate_index : gate_index + 1]
        while gate_scalar.ndim < value.ndim:
            gate_scalar = gate_scalar[..., None]

        pre_activation = value * gate_scalar
        z_mu = _apply_activation(node_info, pre_activation)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state


class ComposerStage2Node(NodeBase):
    """Attention-style stage-2 composer over active column features."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(std=0.02),
        hidden_dim: int = 64,
        query_score_scale: float = 1.0,
        active_top_k: int = 5,
        cert_prior_scale: float = 0.0,
        gate_entropy_weight: float = 0.0,
        gate_deviation_weight: float = 0.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            hidden_dim=hidden_dim,
            query_score_scale=query_score_scale,
            active_top_k=active_top_k,
            cert_prior_scale=cert_prior_scale,
            gate_entropy_weight=gate_entropy_weight,
            gate_deviation_weight=gate_deviation_weight,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "feature": SlotSpec(name="feature", is_multi_input=True),
            "query": SlotSpec(name="query", is_multi_input=False, is_variance_scalable=False),
            "cert_prior": SlotSpec(name="cert_prior", is_multi_input=False, is_variance_scalable=False),
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
        query_dim = None
        for edge_key, shape in input_shapes.items():
            if ":feature" in edge_key and feature_dim is None:
                feature_dim = int(shape[-1])
            elif ":query" in edge_key and query_dim is None:
                query_dim = int(shape[-1])

        if feature_dim is None or query_dim is None:
            raise ValueError("ComposerStage2Node requires feature and query inputs")

        hidden_dim = int(config["hidden_dim"])
        keys = jax.random.split(key, 4)
        weights = {
            "feature_proj": initialize(keys[0], (feature_dim, hidden_dim), weight_init),
            "query_proj": initialize(keys[1], (query_dim, hidden_dim), weight_init),
            "attn_v": initialize(keys[2], (hidden_dim, 1), weight_init),
            "out_proj": initialize(keys[3], (feature_dim, node_shape[-1]), weight_init),
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
    ) -> Tuple[jax.Array, NodeState]:
        feature_edges = _slot_inputs(inputs, "feature")
        query = _sum_inputs(_slot_inputs(inputs, "query"))
        cert_prior = _sum_inputs(_slot_inputs(inputs, "cert_prior"))
        if not feature_edges or query is None:
            raise ValueError(f"{node_info.name} requires feature and query inputs")

        feature_keys = sorted(feature_edges, key=_feature_edge_column_index)
        features = jnp.stack([feature_edges[key] for key in feature_keys], axis=1)
        ordered_prior = None
        if cert_prior is not None:
            column_indices = jnp.asarray(
                [_feature_edge_column_index(key) for key in feature_keys],
                dtype=jnp.int32,
            )
            ordered_prior = cert_prior[:, column_indices]

        details = composer_stage2_details(
            params,
            features,
            query,
            node_info.node_config,
            ordered_prior,
        )
        pre_activation = details["correction"]
        z_mu = _apply_activation(node_info, pre_activation)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        aux_energy = (
            node_info.node_config.get("gate_entropy_weight", 0.0)
            * details["gate_entropy_deficit"]
            + node_info.node_config.get("gate_deviation_weight", 0.0)
            * details["gate_deviation"]
        )
        state = state._replace(energy=state.energy + aux_energy)
        return jnp.sum(state.energy), state


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
    ) -> Tuple[jax.Array, NodeState]:
        base = _sum_inputs(_slot_inputs(inputs, "base"))
        correction = _sum_inputs(_slot_inputs(inputs, "correction"))
        if base is None:
            raise ValueError(f"{node_info.name} requires at least one base input")
        if correction is None:
            correction = jnp.zeros_like(base)

        pre_activation = base + node_info.node_config["correction_scale"] * correction
        z_mu = _apply_activation(node_info, pre_activation)
        error = state.z_latent - z_mu
        state = state._replace(pre_activation=pre_activation, z_mu=z_mu, error=error)
        state = node_info.node_class.energy_functional(state, node_info)
        return jnp.sum(state.energy), state
