"""Shell-bank micro-column nodes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, ZerosInitializer, initialize
from fabricpc.core.types import NodeInfo, NodeParams, NodeState
from fabricpc.nodes.base import NodeBase, SlotSpec
from hibacaml.nodes.common import (
    apply_activation,
    finalize_state,
    slot_inputs,
    sum_inputs,
)


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


def _initialize_shell_params(
    key: jax.Array,
    in_dim: int,
    config: Dict[str, Any],
    weight_init,
    *,
    recurrent: bool = False,
) -> NodeParams:
    dims = _shell_dims(config)
    keys = jax.random.split(key, len(dims) * (2 if recurrent else 1))
    weights = {}
    for idx, (shell_name, shell_dim) in enumerate(dims.items()):
        weights[shell_name] = initialize(keys[idx], (in_dim, shell_dim), weight_init)
        if recurrent:
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


def _project_shell_bank(
    params: NodeParams,
    x: jnp.ndarray,
    node_info: NodeInfo,
) -> jnp.ndarray:
    shell_parts = {}
    for shell_name in _shell_dims(node_info.node_config):
        pre = jnp.matmul(x, params.weights[shell_name])
        pre = pre + params.biases[f"b_{shell_name}"]
        activated = apply_activation(node_info, pre)
        shell_parts[shell_name] = activated * _precision_scale(params, shell_name)
    return _shell_concat(shell_parts)


class ShellBankInputNode(NodeBase):
    """B micro-column projecting gated patch tokens into shell-aware features."""

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
        return _initialize_shell_params(key, in_dim, config, weight_init)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        x = sum_inputs(slot_inputs(inputs, "in"))
        if x is None:
            raise ValueError(f"{node_info.name} requires at least one 'in' input")

        z_mu = _project_shell_bank(params, x, node_info)
        return finalize_state(state, node_info, z_mu)


class ShellBankResidualNode(NodeBase):
    """L micro-column integrating lateral inputs with local B/K residuals."""

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
        return _initialize_shell_params(key, in_dim, config, weight_init)

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        transformed_input = sum_inputs(slot_inputs(inputs, "in"))
        if transformed_input is None:
            raise ValueError(f"{node_info.name} requires an 'in' input")

        skip_input = sum_inputs(slot_inputs(inputs, "skip"))
        projected = _project_shell_bank(params, transformed_input, node_info)
        z_mu = projected if skip_input is None else projected + skip_input
        return finalize_state(state, node_info, z_mu)


class ShellBankRecurrentNode(NodeBase):
    """K micro-column applying recurrent shell dynamics during settling."""

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
        return _initialize_shell_params(
            key,
            in_dim,
            config,
            weight_init,
            recurrent=True,
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        transformed_input = sum_inputs(slot_inputs(inputs, "in"))
        if transformed_input is None:
            raise ValueError(f"{node_info.name} requires an 'in' input")

        shell_parts = {}
        shell_slices = _shell_slices(node_info.node_config)
        for shell_name in _shell_dims(node_info.node_config):
            latent_slice = state.z_latent[..., shell_slices[shell_name]]
            pre = jnp.matmul(transformed_input, params.weights[shell_name])
            pre = pre + jnp.matmul(latent_slice, params.weights[f"recur_{shell_name}"])
            pre = pre + params.biases[f"b_{shell_name}"]
            activated = apply_activation(node_info, pre)
            shell_parts[shell_name] = activated * _precision_scale(params, shell_name)

        z_mu = _shell_concat(shell_parts)
        return finalize_state(state, node_info, z_mu)
