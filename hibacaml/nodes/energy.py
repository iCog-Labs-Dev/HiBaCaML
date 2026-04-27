"""Weighted energies used by the HiBaCaML subsystem."""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy


class WeightedCrossEntropyEnergy(CrossEntropyEnergy):
    """Cross-entropy energy scaled by a fixed loss weight."""

    def __init__(self, weight: float = 1.0, eps: float = 1e-7, axis: int = -1):
        super().__init__(eps=eps, axis=axis)
        self.config = dict(self.config)
        self.config["weight"] = weight

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] | None = None
    ) -> jnp.ndarray:
        base = CrossEntropyEnergy.energy(z_latent, z_mu, config)
        weight = config.get("weight", 1.0) if config else 1.0
        return weight * base

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] | None = None
    ) -> jnp.ndarray:
        base = CrossEntropyEnergy.grad_latent(z_latent, z_mu, config)
        weight = config.get("weight", 1.0) if config else 1.0
        return weight * base


class WeightedGaussianEnergy(GaussianEnergy):
    """Gaussian energy scaled by a fixed loss weight."""

    def __init__(self, weight: float = 1.0, precision: float = 1.0):
        super().__init__(precision=precision)
        self.config = dict(self.config)
        self.config["weight"] = weight

    @staticmethod
    def energy(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] | None = None
    ) -> jnp.ndarray:
        base = GaussianEnergy.energy(z_latent, z_mu, config)
        weight = config.get("weight", 1.0) if config else 1.0
        return weight * base

    @staticmethod
    def grad_latent(
        z_latent: jnp.ndarray, z_mu: jnp.ndarray, config: Dict[str, Any] | None = None
    ) -> jnp.ndarray:
        base = GaussianEnergy.grad_latent(z_latent, z_mu, config)
        weight = config.get("weight", 1.0) if config else 1.0
        return weight * base
