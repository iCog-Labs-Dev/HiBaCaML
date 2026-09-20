"""Stateless calculations used by exact support search."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable, List, Sequence, Tuple

from hibacaml.config import HiBaCaMLConfig, PhiConfig
from hibacaml.control.support import enumerate_reserve_recruitment_supports
from hibacaml.types import (
    ReserveRecruitmentRow,
    SupportPosteriorSummary,
    SupportSearchRow,
)

#: Bounded phi perturbation, one step per coordinate in each direction.
_PHI_STEPS = {
    "outer_quantile": 0.04,
    "middle_quantile": 0.02,
    "replacement_margin_base": 0.02,
    "demotion_min_role_gain": 0.01,
}


def rank_support_rows(
    cfg: HiBaCaMLConfig,
    rows: Sequence[SupportSearchRow],
) -> List[SupportSearchRow]:
    """Order candidates by posterior energy and attach the posterior fields."""
    if not rows:
        return []
    temperature = max(float(cfg.exact_search.support_posterior_temperature), 1e-8)
    ranked = sorted(rows, key=lambda row: row.posterior_energy)
    logits = [-temperature * float(row.posterior_energy) for row in ranked]
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    denom = max(sum(exp_logits), 1e-30)
    probs = [value / denom for value in exp_logits]
    entropy = -sum(prob * math.log(prob + 1e-30) for prob in probs)
    top1_prob = max(probs) if probs else 0.0
    return [
        replace(
            row,
            posterior_rank=rank,
            support_prob=float(probs[rank]),
            support_log_prob=float(math.log(probs[rank] + 1e-30)),
            posterior_entropy=float(entropy),
            top1_prob=float(top1_prob),
            reserve_recruitment_candidate=bool(row.reserve_recruitment_candidate),
        )
        for rank, row in enumerate(ranked)
    ]


def posterior_summary(
    task_id: int,
    rows: Sequence[SupportSearchRow],
    *,
    reserve_recruitment_triggered: bool,
) -> SupportPosteriorSummary:
    if not rows:
        return SupportPosteriorSummary(
            task_id=task_id,
            candidate_count=0,
            posterior_entropy=0.0,
            top1_prob=0.0,
            map_nonshared=tuple(),
            map_posterior_energy=0.0,
            reserve_recruitment_triggered=reserve_recruitment_triggered,
        )
    best = min(rows, key=lambda row: row.posterior_rank if row.posterior_rank >= 0 else 1_000_000)
    return SupportPosteriorSummary(
        task_id=task_id,
        candidate_count=len(rows),
        posterior_entropy=float(best.posterior_entropy),
        top1_prob=float(best.top1_prob),
        map_nonshared=best.nonshared,
        map_posterior_energy=float(best.posterior_energy),
        reserve_recruitment_triggered=reserve_recruitment_triggered,
    )


def reserve_recruitment_diagnostic(
    cfg: HiBaCaMLConfig,
    persistent_state,
    task_id: int,
    adaptive_rows: Sequence[SupportSearchRow],
) -> Tuple[bool, ReserveRecruitmentRow]:
    """Decide whether reserve candidates join the audit, and record why."""
    saturation = _adaptive_saturation(cfg, persistent_state)
    entropy = adaptive_rows[0].posterior_entropy if adaptive_rows else 0.0
    top1_prob = adaptive_rows[0].top1_prob if adaptive_rows else 0.0
    saturation_ok = saturation >= cfg.exact_search.reserve_saturation_threshold
    confidence_weak = (
        entropy >= cfg.exact_search.reserve_entropy_threshold
        or top1_prob <= cfg.exact_search.reserve_top1_prob_threshold
    )
    has_reserves = bool(cfg.column_pool.reserve_indices)
    triggered = has_reserves and saturation_ok and confidence_weak
    if triggered:
        reason = "saturation_and_weak_posterior"
    elif not has_reserves:
        reason = "no_reserve_columns"
    elif not saturation_ok:
        reason = "saturation_below_threshold"
    else:
        reason = "posterior_confident"
    reserve_candidates = enumerate_reserve_recruitment_supports(cfg)
    return triggered, ReserveRecruitmentRow(
        task_id=task_id,
        triggered=triggered,
        adaptive_candidate_count=len(adaptive_rows),
        reserve_candidate_count=len(reserve_candidates) if triggered else 0,
        saturation=float(saturation),
        posterior_entropy=float(entropy),
        top1_prob=float(top1_prob),
        saturation_threshold=float(cfg.exact_search.reserve_saturation_threshold),
        entropy_threshold=float(cfg.exact_search.reserve_entropy_threshold),
        top1_prob_threshold=float(cfg.exact_search.reserve_top1_prob_threshold),
        reason=reason,
    )


def phi_candidates(cfg: HiBaCaMLConfig, current_phi) -> Tuple[PhiConfig, ...]:
    """Deterministic bounded local controller neighborhood around current phi."""
    # PhiConfig is frozen, so the centre can be shared rather than copied.
    raw = [current_phi]
    for attr, step in _PHI_STEPS.items():
        for sign in (-1.0, 1.0):
            moved = replace(current_phi, **{attr: getattr(current_phi, attr) + sign * step})
            raw.append(_clip_phi(cfg, moved))
    dedup = {
        (
            phi.outer_quantile,
            phi.middle_quantile,
            phi.replacement_margin_base,
            phi.demotion_min_role_gain,
        ): phi
        for phi in raw
    }
    return tuple(dedup.values())


def _clip_phi(cfg: HiBaCaMLConfig, phi) -> PhiConfig:
    """Clamp every phi coordinate back inside its configured bounds."""
    search = cfg.exact_search
    return PhiConfig(
        outer_quantile=_clip(phi.outer_quantile, search.outer_quantile_bounds),
        middle_quantile=_clip(phi.middle_quantile, search.middle_quantile_bounds),
        replacement_margin_base=_clip(
            phi.replacement_margin_base, search.replacement_margin_bounds
        ),
        demotion_min_role_gain=_clip(
            phi.demotion_min_role_gain, search.demotion_gain_bounds
        ),
    )


def phi_l1_distance(left, right) -> float:
    return float(
        abs(left.outer_quantile - right.outer_quantile)
        + abs(left.middle_quantile - right.middle_quantile)
        + abs(left.replacement_margin_base - right.replacement_margin_base)
        + abs(left.demotion_min_role_gain - right.demotion_min_role_gain)
    )


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    ls = set(left)
    rs = set(right)
    if not ls and not rs:
        return 1.0
    return float(len(ls & rs) / max(len(ls | rs), 1))


def jump_size(left: Sequence[int], right: Sequence[int]) -> int:
    return int(len(set(left).symmetric_difference(set(right))))


def candidate_chunks(
    items: Sequence[Sequence[int]],
    size: int,
    *,
    pad: bool,
) -> Iterable[Tuple[Tuple[Tuple[int, ...], ...], int]]:
    """Yield `(chunk, real_count)`; a padded tail repeats its last entry."""
    size = max(1, int(size))
    normalized = tuple(tuple(sorted(item)) for item in items)
    for start in range(0, len(normalized), size):
        real = normalized[start : start + size]
        if not real:
            continue
        chunk = list(real)
        if pad and len(chunk) < size and size > 1:
            chunk.extend([chunk[-1]] * (size - len(chunk)))
        yield tuple(chunk), len(real)


def mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _adaptive_saturation(cfg: HiBaCaMLConfig, persistent_state) -> float:
    certificates = getattr(persistent_state, "certificates", {})
    values = [
        certificates[idx].saturation
        for idx in cfg.column_pool.adaptive_indices
        if idx in certificates
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clip(value: float, bounds: Tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))
