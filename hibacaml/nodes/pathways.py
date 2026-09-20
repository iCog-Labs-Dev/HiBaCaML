"""Patch-token preparation and support-gated graph pathways."""

from __future__ import annotations

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
        in_dim = int(patch_h * patch_w * input_shape[-1])
        proj = initialize(key, (in_dim, embed_dim), weight_init)
        bias = jnp.zeros((1, 1, embed_dim))
        return NodeParams(weights={"proj": proj}, biases={"b_proj": bias})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        x = sum_inputs(inputs)
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
        z_mu = apply_activation(node_info, pre_activation)
        return finalize_state(state, node_info, z_mu)


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
    ) -> NodeState:
        value = sum_inputs(slot_inputs(inputs, "value"))
        gate = sum_inputs(slot_inputs(inputs, "gate"))
        if value is None or gate is None:
            raise ValueError(f"{node_info.name} requires 'value' and 'gate' inputs")

        gate_index = int(node_info.node_config["gate_index"])
        gate_scalar = gate[..., gate_index : gate_index + 1]
        while gate_scalar.ndim < value.ndim:
            gate_scalar = gate_scalar[..., None]

        pre_activation = value * gate_scalar
        z_mu = apply_activation(node_info, pre_activation)
        return finalize_state(state, node_info, z_mu)
