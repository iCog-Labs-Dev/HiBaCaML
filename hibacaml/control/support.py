"""Support-state helpers for HiBaCaML."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence, Tuple

import jax.numpy as jnp

from hibacaml.config import HiBaCaMLConfig


def candidate_nonshared_pool(cfg: HiBaCaMLConfig) -> Tuple[int, ...]:
    """Nonshared pool for ordinary support search.

    V20.2b uses adaptive-only enumeration; reserves enter via the explicit
    reserve-recruitment path in `enumerate_reserve_recruitment_supports`.
    """
    return cfg.column_pool.adaptive_indices


def default_nonshared_support(cfg: HiBaCaMLConfig) -> Tuple[int, ...]:
    """Default nonshared support used before the first exact search."""
    return tuple(candidate_nonshared_pool(cfg)[: cfg.column_pool.topk_nonshared])


def build_full_support(cfg: HiBaCaMLConfig, nonshared: Sequence[int]) -> Tuple[int, ...]:
    """Return the sorted support including always-on shared columns."""
    return tuple(sorted(cfg.column_pool.shared_indices + tuple(nonshared)))


def support_mask_from_nonshared(
    cfg: HiBaCaMLConfig,
    nonshared: Sequence[int],
) -> jnp.ndarray:
    """Return a boolean-valued support vector with shared columns forced on."""
    mask = jnp.zeros((cfg.column_pool.total_columns,), dtype=jnp.float32)
    indices = build_full_support(cfg, nonshared)
    return mask.at[jnp.array(indices)].set(1.0)


def enumerate_nonshared_supports(cfg: HiBaCaMLConfig) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate all exact-search nonshared candidate supports."""
    return tuple(combinations(candidate_nonshared_pool(cfg), cfg.column_pool.topk_nonshared))


def enumerate_reserve_recruitment_supports(cfg: HiBaCaMLConfig) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate paper-style candidates with exactly one reserve column."""
    reserve_count = 1
    adaptive_count = cfg.column_pool.topk_nonshared - reserve_count
    if adaptive_count <= 0:
        return ()
    candidates = []
    for reserve in cfg.column_pool.reserve_indices:
        for adaptive in combinations(cfg.column_pool.adaptive_indices, adaptive_count):
            candidates.append(tuple(sorted((*adaptive, reserve))))
    return tuple(candidates)


def one_swap_neighbors(
    cfg: HiBaCaMLConfig,
    current_nonshared: Sequence[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate all one-swap neighbor supports."""
    current = tuple(current_nonshared)
    current_set = set(current)
    inactive = [col for col in candidate_nonshared_pool(cfg) if col not in current_set]
    neighbors = []
    for active in current:
        for replacement in inactive:
            updated = list(current)
            updated[updated.index(active)] = replacement
            neighbors.append(tuple(sorted(updated)))
    return tuple(neighbors)
