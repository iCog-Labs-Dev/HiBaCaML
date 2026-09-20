"""Shared helpers for custom HiBaCaML nodes."""

from __future__ import annotations

from typing import Dict, Optional
import jax.numpy as jnp

from fabricpc.core.types import NodeInfo, NodeState

def apply_activation(node_info: NodeInfo, x: jnp.ndarray) -> jnp.ndarray:
    activation = node_info.activation
    return type(activation).forward(x, activation.config)

def finalize_state(
    state: NodeState,
    node_info: NodeInfo,
    z_mu: jnp.ndarray,
) -> NodeState:
    state = state._replace(z_mu=z_mu, error=state.z_latent - z_mu)
    return node_info.node_class.energy_functional(state, node_info)

def slot_inputs(inputs: Dict[str, jnp.ndarray], slot_name: str) -> Dict[str, jnp.ndarray]:
    token = f":{slot_name}"
    return {k: v for k, v in inputs.items() if token in k}

def sum_inputs(inputs: Dict[str, jnp.ndarray]) -> Optional[jnp.ndarray]:
    out = None
    for value in inputs.values():
        out = value if out is None else out + value
    return out
