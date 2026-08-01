"""Exact support search, replay-bank reselection, and one-swap maintenance for HiBaCaML V20.2b."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax.numpy as jnp

from hibacaml.config import HiBaCaMLConfig, PhiConfig
from hibacaml.control.replay_bank import (
    ReplayRow,
    SelectorBank,
    build_context,
    make_replay_row,
)
from hibacaml.control.support import (
    build_full_support,
    enumerate_nonshared_supports,
    enumerate_reserve_recruitment_supports,
    one_swap_neighbors,
)
from hibacaml.types import (
    BoundaryBundle,
    ControllerSearchRow,
    DemotionSwapAuditRow,
    LocalSwapRow,
    ReplayProposalRow,
    ReserveRecruitmentRow,
    SupportPosteriorSummary,
    SupportSearchRow,
)


@dataclass(frozen=True)
class _ScoredCandidate:
    """One scored support candidate for the V20.2b reselection layer."""

    nonshared: Tuple[int, ...]
    base_total: float
    overlap_jaccard: float
    jump_size: int
    history_intersection: int
    penalised_total: float


@dataclass(frozen=True)
class ReselectionResult:
    """Outcome of the V20.2b boundary reselection pipeline."""

    accepted_nonshared: Tuple[int, ...]
    accepted_phi: PhiConfig
    accepted_source: str  # "original" | "local" | "replay"
    proposal_row: ReplayProposalRow


class ExactSearchService:
    """Exact support search service with V20.2b replay-bank reselection."""

    def __init__(
        self,
        cfg: HiBaCaMLConfig,
        trainer,
        *,
        selector_bank: SelectorBank,
        run_id: str = "",
    ):
        self.cfg = cfg
        self.trainer = trainer
        self.selector_bank = selector_bank
        self.run_id = run_id
        self._old_audit_cache: Dict[Tuple[int, int, int], Dict[str, float]] = {}

    def _normalize_data_batch_size(self, name: str, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} must be None or a positive integer, got {value!r}")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be None or a positive integer, got {value!r}")
        return value

    def _bundle_batch_caps(self, purpose: str) -> Dict[str, Optional[int] | bool]:
        cfg = self.cfg.exact_search
        if purpose == "boundary":
            return {
                "current": self._normalize_data_batch_size(
                    "boundary_current_data_batch_size",
                    cfg.boundary_current_data_batch_size,
                ),
                "rollout": self._normalize_data_batch_size(
                    "rollout_train_data_batch_size",
                    cfg.rollout_train_data_batch_size,
                ),
                "worst_old": self._normalize_data_batch_size(
                    "boundary_worst_old_data_batch_size",
                    cfg.boundary_worst_old_data_batch_size,
                ),
                "mixed_old": self._normalize_data_batch_size(
                    "boundary_mixed_old_data_batch_size",
                    cfg.boundary_mixed_old_data_batch_size,
                ),
                "include_rollout": True,
            }
        if purpose == "local_swap":
            cap = self._normalize_data_batch_size(
                "local_swap_audit_data_batch_size",
                cfg.local_swap_audit_data_batch_size,
            )
            return {
                "current": cap,
                "rollout": None,
                "worst_old": cap,
                "mixed_old": cap,
                "include_rollout": False,
            }
        if purpose == "demotion":
            cap = self._normalize_data_batch_size(
                "demotion_audit_data_batch_size",
                cfg.demotion_audit_data_batch_size,
            )
            return {
                "current": cap,
                "rollout": None,
                "worst_old": cap,
                "mixed_old": cap,
                "include_rollout": False,
            }
        raise ValueError(f"Unsupported boundary bundle purpose: {purpose!r}")

    @staticmethod
    def _slice_batch(batch: Dict[str, jnp.ndarray], max_examples: Optional[int]):
        if max_examples is None:
            return batch
        return {key: value[:max_examples] for key, value in batch.items()}

    def make_bundle(self, task_id: int, *, purpose: str = "boundary") -> BoundaryBundle:
        """Build the audit bundle: current eval batches + worst-old + mixed-old."""
        caps = self._bundle_batch_caps(purpose)
        task = self.trainer.task(task_id)
        current_eval = tuple(
            self._slice_batch(batch, caps["current"])
            for idx, batch in enumerate(task.test_loader)
            if idx < self.cfg.exact_search.boundary_current_batches
        )
        train_batches = ()
        if caps["include_rollout"]:
            train_batches = tuple(
                self._slice_batch(batch, caps["rollout"])
                for idx, batch in enumerate(task.train_loader)
                if idx < self.cfg.exact_search.boundary_rollout_steps
            )

        worst_old = None
        worst_old_eval = None
        mixed_old_eval: List[Tuple[int, Dict[str, jnp.ndarray]]] = []
        if task_id > 0:
            saved = self.trainer.evaluate_all_saved_supports()
            if saved:
                worst_old = min(saved, key=lambda key: saved[key]["accuracy"])
            if worst_old is not None:
                worst_old_task = self.trainer.task(worst_old)
                worst_old_eval = self._concat_loader_prefix(
                    worst_old_task.test_loader,
                    self.cfg.exact_search.boundary_old_batches,
                    batch_size=caps["worst_old"],
                )
                if worst_old_eval is not None:
                    worst_old_eval = (worst_old, worst_old_eval)
            mixed_old_eval = list(
                self._build_mixed_old_fragments(
                    task_id,
                    batch_size=caps["mixed_old"],
                )
            )
        return BoundaryBundle(
            current_eval=current_eval,
            train_batches=train_batches,
            worst_old_eval=worst_old_eval,
            mixed_old_eval=tuple(mixed_old_eval),
            worst_old=worst_old,
        )

    def _concat_loader_prefix(
        self,
        loader,
        max_batches: int,
        *,
        batch_size: Optional[int] = None,
    ):
        pieces = []
        remaining = batch_size
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            if remaining is not None:
                if remaining <= 0:
                    break
                batch = self._slice_batch(batch, remaining)
                remaining -= int(batch["x"].shape[0])
            pieces.append(batch)
        if not pieces:
            return None
        if len(pieces) == 1:
            return pieces[0]
        return {
            key: jnp.concatenate([piece[key] for piece in pieces], axis=0)
            for key in pieces[0]
        }

    def _build_mixed_old_fragments(
        self,
        task_id: int,
        *,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tuple[int, Dict[str, jnp.ndarray]], ...]:
        """Build one deterministic mixed-old batch from all prior tasks."""
        prior_batches: List[Tuple[int, Dict[str, jnp.ndarray]]] = []
        for prev_task_id in range(task_id):
            batch = next(iter(self.trainer.task(prev_task_id).test_loader), None)
            if batch is not None:
                prior_batches.append((prev_task_id, batch))
        if not prior_batches:
            return ()

        target_size = min(int(batch["x"].shape[0]) for _, batch in prior_batches)
        if batch_size is not None:
            target_size = min(target_size, int(batch_size))
        target_size = max(1, target_size)
        per_source = max(1, target_size // len(prior_batches))
        remainder = max(0, target_size - per_source * len(prior_batches))

        fragments = []
        for idx, (prev_task_id, _) in enumerate(prior_batches):
            take = per_source + (1 if idx < remainder else 0)
            remaining = take
            pieces = []
            for batch in self.trainer.task(prev_task_id).test_loader:
                take_now = min(remaining, int(batch["x"].shape[0]))
                if take_now <= 0:
                    continue
                pieces.append(
                    {key: jnp.asarray(value[:take_now]) for key, value in batch.items()}
                )
                remaining -= take_now
                if remaining <= 0:
                    break
            if not pieces:
                continue
            if len(pieces) == 1:
                fragment = pieces[0]
            else:
                fragment = {
                    key: jnp.concatenate([piece[key] for piece in pieces], axis=0)
                    for key in pieces[0]
                }
            fragments.append((prev_task_id, fragment))
        return tuple(fragments)

    def _switch_penalty(self, task_id: int, support_cols: Sequence[int]) -> float:
        if task_id == 0:
            return 0.0
        prev = self.trainer.persistent_state.task_support_snapshots.get(task_id - 1)
        prev_set = set(prev.nonshared) if prev is not None else set(self.trainer.current_nonshared)
        return self.cfg.exact_search.switch_penalty * float(
            len(set(support_cols).symmetric_difference(prev_set))
        )

    def _certificate_reuse_score(self, support_cols: Sequence[int]) -> float:
        certificates = getattr(self.trainer.persistent_state, "certificates", {})
        if not certificates:
            return 0.0
        values = []
        for idx in tuple(sorted(support_cols)):
            cert = certificates.get(idx)
            if cert is None:
                values.append(0.0)
                continue
            values.append(
                cert.q_mean
                + 0.5 * cert.shared_abstraction_mass
                - 0.25 * cert.specificity_load
                - 0.25 * cert.demotion_pressure
                - 0.10 * cert.saturation
            )
        return float(sum(values) / max(len(values), 1))

    def _support_row_from_objective(
        self,
        task_id: int,
        support_cols: Sequence[int],
        objective: Dict[str, float],
        *,
        reserve_recruitment_candidate: bool = False,
    ) -> SupportSearchRow:
        cert_score = self._certificate_reuse_score(support_cols)
        posterior_energy = (
            float(objective["total"])
            - self.cfg.exact_search.certificate_support_weight * cert_score
        )
        return SupportSearchRow(
            task_id=task_id,
            nonshared=tuple(sorted(support_cols)),
            static_total=objective["total"],
            current_first_loss=objective["current_first_loss"],
            current_remaining_loss=objective["current_remaining_loss"],
            old_worst_loss=objective["old_worst_loss"],
            old_mix_loss=objective["old_mix_loss"],
            switch_penalty=objective["switch_penalty"],
            posterior_energy=float(posterior_energy),
            certificate_reuse_score=float(cert_score),
            reserve_recruitment_candidate=bool(reserve_recruitment_candidate),
        )

    def _rank_support_rows(
        self,
        rows: Sequence[SupportSearchRow],
        *,
        reserve_recruitment_triggered: bool = False,
    ) -> List[SupportSearchRow]:
        if not rows:
            return []
        temperature = max(float(self.cfg.exact_search.support_posterior_temperature), 1e-8)
        ranked = sorted(rows, key=lambda row: row.posterior_energy)
        logits = [-temperature * float(row.posterior_energy) for row in ranked]
        max_logit = max(logits)
        exp_logits = [math.exp(logit - max_logit) for logit in logits]
        denom = max(sum(exp_logits), 1e-30)
        probs = [value / denom for value in exp_logits]
        entropy = -sum(prob * math.log(prob + 1e-30) for prob in probs)
        top1_prob = max(probs) if probs else 0.0
        return [
            SupportSearchRow(
                **{
                    **row.__dict__,
                    "posterior_rank": rank,
                    "support_prob": float(probs[rank]),
                    "support_log_prob": float(math.log(probs[rank] + 1e-30)),
                    "posterior_entropy": float(entropy),
                    "top1_prob": float(top1_prob),
                    "reserve_recruitment_candidate": bool(row.reserve_recruitment_candidate),
                }
            )
            for rank, row in enumerate(ranked)
        ]

    def _posterior_summary(
        self,
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

    def _adaptive_saturation(self) -> float:
        certificates = getattr(self.trainer.persistent_state, "certificates", {})
        values = [
            certificates[idx].saturation
            for idx in self.cfg.column_pool.adaptive_indices
            if idx in certificates
        ]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _reserve_recruitment_diagnostic(
        self,
        task_id: int,
        adaptive_rows: Sequence[SupportSearchRow],
    ) -> Tuple[bool, ReserveRecruitmentRow]:
        saturation = self._adaptive_saturation()
        entropy = adaptive_rows[0].posterior_entropy if adaptive_rows else 0.0
        top1_prob = adaptive_rows[0].top1_prob if adaptive_rows else 0.0
        saturation_ok = saturation >= self.cfg.exact_search.reserve_saturation_threshold
        confidence_weak = (
            entropy >= self.cfg.exact_search.reserve_entropy_threshold
            or top1_prob <= self.cfg.exact_search.reserve_top1_prob_threshold
        )
        has_reserves = bool(self.cfg.column_pool.reserve_indices)
        triggered = has_reserves and saturation_ok and confidence_weak
        if triggered:
            reason = "saturation_and_weak_posterior"
        elif not has_reserves:
            reason = "no_reserve_columns"
        elif not saturation_ok:
            reason = "saturation_below_threshold"
        else:
            reason = "posterior_confident"
        reserve_candidates = enumerate_reserve_recruitment_supports(self.cfg)
        return triggered, ReserveRecruitmentRow(
            task_id=task_id,
            triggered=triggered,
            adaptive_candidate_count=len(adaptive_rows),
            reserve_candidate_count=len(reserve_candidates) if triggered else 0,
            saturation=float(saturation),
            posterior_entropy=float(entropy),
            top1_prob=float(top1_prob),
            saturation_threshold=float(self.cfg.exact_search.reserve_saturation_threshold),
            entropy_threshold=float(self.cfg.exact_search.reserve_entropy_threshold),
            top1_prob_threshold=float(self.cfg.exact_search.reserve_top1_prob_threshold),
            reason=reason,
        )

    def boundary_objective(
        self,
        task_id: int,
        support_cols: Sequence[int],
        bundle: BoundaryBundle,
        *,
        trainer=None,
        refresh_certificates: bool,
    ) -> Dict[str, float]:
        """Boundary objective from Eq. (1)."""
        return self.boundary_objectives(
            task_id,
            [support_cols],
            bundle,
            trainer=trainer,
            refresh_certificates=refresh_certificates,
        )[0]

    def boundary_objectives(
        self,
        task_id: int,
        supports: Sequence[Sequence[int]],
        bundle: BoundaryBundle,
        *,
        trainer=None,
        refresh_certificates: bool,
    ) -> List[Dict[str, float]]:
        """Batched Eq. (1) boundary objective for support candidates."""
        trainer = trainer or self.trainer
        support_list = [tuple(sorted(support)) for support in supports]
        count = len(support_list)
        task = trainer.task(task_id)
        current_first = [0.0 for _ in range(count)]
        current_remaining = [0.0 for _ in range(count)]

        for idx, batch in enumerate(bundle.current_eval):
            losses = self._evaluate_batch_losses(
                trainer,
                task,
                batch,
                support_list,
                refresh_certificates=refresh_certificates and idx == 0,
            )
            target = current_first if idx == 0 else current_remaining
            for row_idx, loss in enumerate(losses):
                target[row_idx] += float(loss)

        old_terms = self._old_audit_terms(trainer, bundle)
        old_worst_loss = old_terms["old_worst_loss"]
        old_mix_loss = old_terms["old_mix_loss"]

        rows = []
        for idx, support_cols in enumerate(support_list):
            switch_penalty = self._switch_penalty(task_id, support_cols)
            total = (
                current_first[idx]
                + current_remaining[idx]
                + self.cfg.exact_search.exact_old_worst_weight * old_worst_loss
                + self.cfg.exact_search.exact_old_mix_weight * old_mix_loss
                + switch_penalty
            )
            rows.append(
                {
                    "current_first_loss": float(current_first[idx]),
                    "current_remaining_loss": float(current_remaining[idx]),
                    "old_worst_loss": float(old_worst_loss),
                    "old_mix_loss": float(old_mix_loss),
                    "switch_penalty": float(switch_penalty),
                    "total": float(total),
                }
            )
        return rows

    def _old_audit_terms(self, trainer, bundle: BoundaryBundle) -> Dict[str, float]:
        """Compute old-task audit losses under their frozen saved supports."""
        cache_key = (
            id(trainer),
            id(bundle),
            int(getattr(trainer.persistent_state, "params_revision", 0)),
        )
        cached = self._old_audit_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        if bundle.worst_old_eval is None and not bundle.mixed_old_eval:
            return {"old_worst_loss": 0.0, "old_mix_loss": 0.0}

        old_worst_loss = 0.0
        if bundle.worst_old_eval is not None:
            prev_task_id, batch = bundle.worst_old_eval
            snapshot = trainer.persistent_state.task_support_snapshots.get(prev_task_id)
            prev_support = snapshot.nonshared if snapshot is not None else trainer.current_nonshared
            old_worst_loss = trainer.evaluate_batch_loss(
                trainer.task(prev_task_id),
                batch,
                prev_support,
                refresh_certificates=False,
            )

        mix_losses = []
        for prev_task_id, batch in bundle.mixed_old_eval:
            snapshot = trainer.persistent_state.task_support_snapshots.get(prev_task_id)
            prev_support = snapshot.nonshared if snapshot is not None else trainer.current_nonshared
            mix_losses.append(
                trainer.evaluate_batch_loss(
                    trainer.task(prev_task_id),
                    batch,
                    prev_support,
                    refresh_certificates=False,
                )
            )

        terms = {
            "old_worst_loss": float(old_worst_loss),
            "old_mix_loss": float(sum(mix_losses) / max(len(mix_losses), 1)),
        }
        self._old_audit_cache[cache_key] = dict(terms)
        return terms

    def _evaluate_batch_losses(
        self,
        trainer,
        task,
        batch: Dict[str, jnp.ndarray],
        supports: Sequence[Sequence[int]],
        *,
        refresh_certificates: bool,
    ) -> List[float]:
        if hasattr(trainer, "evaluate_batch_losses"):
            return list(
                trainer.evaluate_batch_losses(
                    task,
                    batch,
                    supports,
                    refresh_certificates=refresh_certificates,
                )
            )
        return [
            trainer.evaluate_batch_loss(
                task,
                batch,
                support,
                refresh_certificates=refresh_certificates and idx == 0,
            )
            for idx, support in enumerate(supports)
        ]

    def static_support_score(
        self,
        task_id: int,
        support_cols: Sequence[int],
        bundle: BoundaryBundle,
        *,
        reserve_recruitment_candidate: bool = False,
    ) -> SupportSearchRow:
        objective = self.boundary_objective(
            task_id,
            support_cols,
            bundle,
            refresh_certificates=False,
        )
        return self._support_row_from_objective(
            task_id,
            support_cols,
            objective,
            reserve_recruitment_candidate=reserve_recruitment_candidate,
        )

    def static_support_scores_batched(
        self,
        task_id: int,
        support_candidates: Sequence[Sequence[int]],
        bundle: BoundaryBundle,
        *,
        reserve_recruitment_candidate: bool = False,
    ) -> List[SupportSearchRow]:
        """Score exact support candidates in JAX-friendly batches."""
        rows: List[SupportSearchRow] = []
        batch_size = max(1, int(self.cfg.exact_search.static_support_batch_size))
        for chunk, real_count in _candidate_chunks(
            tuple(support_candidates),
            batch_size,
            pad=True,
        ):
            objectives = self.boundary_objectives(
                task_id,
                chunk,
                bundle,
                refresh_certificates=False,
            )
            for support_cols, objective in zip(chunk[:real_count], objectives[:real_count]):
                rows.append(
                    self._support_row_from_objective(
                        task_id,
                        support_cols,
                        objective,
                        reserve_recruitment_candidate=reserve_recruitment_candidate,
                    )
                )
        return rows

    def phi_candidates(self) -> Tuple[PhiConfig, ...]:
        """Deterministic bounded local controller neighborhood around current phi."""
        base = self.trainer.current_phi
        steps = {
            "outer_quantile": 0.04,
            "middle_quantile": 0.02,
            "replacement_margin_base": 0.02,
            "demotion_min_role_gain": 0.01,
        }
        raw = [PhiConfig(**base.__dict__)]
        for attr, step in steps.items():
            for sign in (-1.0, 1.0):
                values = dict(base.__dict__)
                values[attr] = values[attr] + sign * step
                values["outer_quantile"] = _clip(values["outer_quantile"], self.cfg.exact_search.outer_quantile_bounds)
                values["middle_quantile"] = _clip(values["middle_quantile"], self.cfg.exact_search.middle_quantile_bounds)
                values["replacement_margin_base"] = _clip(
                    values["replacement_margin_base"],
                    self.cfg.exact_search.replacement_margin_bounds,
                )
                values["demotion_min_role_gain"] = _clip(
                    values["demotion_min_role_gain"],
                    self.cfg.exact_search.demotion_gain_bounds,
                )
                raw.append(PhiConfig(**values))
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

    def rollout_score(
        self,
        task_id: int,
        support_cols: Sequence[int],
        phi: PhiConfig,
        bundle: BoundaryBundle,
    ) -> ControllerSearchRow:
        clone = self.trainer.clone()
        clone.set_current_support(task_id, support_cols, phi)
        clone.current_phi = phi
        task = clone.task(task_id)
        final_state = None
        for batch in bundle.train_batches:
            if self.cfg.exact_search.rollout_gradient_mode == "trainer":
                grads, _, final_state = clone.compute_pc_gradients(
                    batch,
                    task,
                    support_cols,
                )
            elif self.cfg.exact_search.rollout_gradient_mode == "pc":
                from hibacaml.training.trainer import HiBaCaMLTrainer

                grads, _, final_state = HiBaCaMLTrainer.compute_pc_gradients(
                    clone,
                    batch,
                    task,
                    support_cols,
                )
            else:
                raise ValueError(
                    "rollout_gradient_mode must be 'trainer' or 'pc'"
                )
            grads = clone._mask_grads(grads, support_cols)
            clone._apply_grads(grads)
            clone.params = clone.shell_controller.apply_structural_edits(
                clone.params,
                final_state,
                clone.persistent_state,
                clone.active_full_support(support_cols),
                phi,
            )
            clone.persistent_state.params = clone.params
            clone._bump_params_revision()
        if final_state is None and bundle.current_eval:
            final_state, _ = clone.run_batch_evaluation_inference(
                bundle.current_eval[0],
                task,
                support_cols,
            )
        objective = self.boundary_objective(
            task_id,
            support_cols,
            bundle,
            trainer=clone,
            refresh_certificates=True,
        )
        semantic_regularizer = clone.shell_controller.semantic_penalty(
            clone.persistent_state,
            clone.active_full_support(support_cols),
        )
        phi_l1_penalty = self.cfg.exact_search.controller_l1_penalty * _phi_l1_distance(
            phi,
            self.cfg.phi,
        )
        if final_state is not None and hasattr(clone, "composer_diagnostics_from_state"):
            composer_diag = clone.composer_diagnostics_from_state(final_state, task, support_cols)
        else:
            composer_diag = {
                "gate_entropy": 0.0,
                "gate_deviation": 0.0,
                "cert_prior_mean": 0.0,
            }
        certs = list(clone.persistent_state.certificates.values())
        occ_tier1 = _mean_or_zero([cert.tier_occ[0] for cert in certs])
        occ_tier2 = _mean_or_zero([cert.tier_occ[1] for cert in certs])
        occ_tier3 = _mean_or_zero([cert.tier_occ[2] for cert in certs])
        q_tier1 = _mean_or_zero([cert.tier_q[0] for cert in certs])
        q_tier2 = _mean_or_zero([cert.tier_q[1] for cert in certs])
        q_tier3 = _mean_or_zero([cert.tier_q[2] for cert in certs])
        row = ControllerSearchRow(
            task_id=task_id,
            nonshared=tuple(sorted(support_cols)),
            phi=phi,
            rollout_total=float(objective["total"] + semantic_regularizer + phi_l1_penalty),
            boundary_total=objective["total"],
            current_loss=objective["current_first_loss"] + objective["current_remaining_loss"],
            old_worst_loss=objective["old_worst_loss"],
            old_mix_loss=objective["old_mix_loss"],
            switch_penalty=objective["switch_penalty"],
            phi_l1_penalty=float(phi_l1_penalty),
            semantic_regularizer=float(semantic_regularizer),
            gate_entropy=float(composer_diag.get("gate_entropy", 0.0)),
            gate_deviation=float(composer_diag.get("gate_deviation", 0.0)),
            cert_prior_mean=float(composer_diag.get("cert_prior_mean", 0.0)),
            occ_tier1=occ_tier1,
            occ_tier2=occ_tier2,
            occ_tier3=occ_tier3,
            q_tier1=q_tier1,
            q_tier2=q_tier2,
            q_tier3=q_tier3,
        )
        del clone
        gc.collect()
        return row

    # ------------------------------------------------------------------
    # V20.2b reselection helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
        ls = set(left)
        rs = set(right)
        if not ls and not rs:
            return 1.0
        return float(len(ls & rs) / max(len(ls | rs), 1))

    @staticmethod
    def _jump_size(left: Sequence[int], right: Sequence[int]) -> int:
        return int(len(set(left).symmetric_difference(set(right))))

    def _history_intersection(self, task_id: int, candidate: Sequence[int]) -> int:
        recent = getattr(self.trainer.persistent_state, "recently_demoted", {})
        if not recent:
            return 0
        window = max(0, int(self.cfg.exact_search.replay_history_window))
        if window == 0:
            return 0
        candidate_set = set(candidate)
        hits = 0
        for prev_task_id in range(max(0, task_id - window), task_id + 1):
            cols = recent.get(prev_task_id, ())
            hits += len(candidate_set.intersection(cols))
        return int(hits)

    def _penalise_total(
        self,
        *,
        base_total: float,
        candidate: Sequence[int],
        original: Sequence[int],
        task_id: int,
    ) -> _ScoredCandidate:
        cfg = self.cfg.exact_search
        overlap = self._jaccard(candidate, original)
        jump = self._jump_size(candidate, original)
        hist = self._history_intersection(task_id, candidate)
        penalty = (
            cfg.replay_overlap_penalty_alpha * (1.0 - overlap)
            + cfg.replay_jump_penalty_beta * float(jump)
            + cfg.replay_history_penalty_gamma * float(hist)
        )
        return _ScoredCandidate(
            nonshared=tuple(sorted(candidate)),
            base_total=float(base_total),
            overlap_jaccard=float(overlap),
            jump_size=int(jump),
            history_intersection=int(hist),
            penalised_total=float(base_total + penalty),
        )

    def _propose_from_bank(
        self,
        task_id: int,
        current_nonshared: Sequence[int],
        *,
        k: int,
        full_support: Sequence[int],
    ) -> List[Tuple[int, ...]]:
        if k <= 0 or self.selector_bank is None or len(self.selector_bank) == 0:
            return []
        context = build_context(
            task_id,
            self.trainer.persistent_state,
            self.cfg,
            full_support,
        )
        rows = self.selector_bank.query(
            context,
            k=k,
            current_nonshared=tuple(sorted(current_nonshared)),
        )
        adaptive_set = set(self.cfg.column_pool.adaptive_indices)
        reserve_set = set(self.cfg.column_pool.reserve_indices)
        allowed = adaptive_set | reserve_set
        topk = int(self.cfg.column_pool.topk_nonshared)
        candidates: List[Tuple[int, ...]] = []
        seen: set = set()
        for row in rows:
            cand = tuple(sorted(row.nonshared))
            if len(cand) != topk:
                continue
            if not all(col in allowed for col in cand):
                continue
            if cand in seen:
                continue
            seen.add(cand)
            candidates.append(cand)
        return candidates

    def _record_replay_row(
        self,
        *,
        provenance: str,
        task_id: int,
        nonshared: Sequence[int],
        phi: PhiConfig,
        objective: Dict[str, float],
    ) -> None:
        if self.selector_bank is None:
            return
        if not self.cfg.reporting.write_selector_state:
            return
        full_support = build_full_support(self.cfg, nonshared)
        context = build_context(
            task_id,
            self.trainer.persistent_state,
            self.cfg,
            full_support,
        )
        row = make_replay_row(
            run_id=self.run_id,
            task_id=task_id,
            global_step=int(self.trainer.persistent_state.global_step),
            context=context,
            nonshared=nonshared,
            full_support=full_support,
            phi=phi,
            observed_total=float(objective.get("total", 0.0)),
            observed_old_worst=float(objective.get("old_worst_loss", 0.0)),
            observed_old_mix=float(objective.get("old_mix_loss", 0.0)),
            provenance=provenance,
        )
        self.selector_bank.add(row)

    def _reselect_with_bank(
        self,
        task_id: int,
        *,
        original_nonshared: Tuple[int, ...],
        original_phi: PhiConfig,
        bundle: BoundaryBundle,
    ) -> ReselectionResult:
        """Three-stream V20.2b reselection at task boundary.

        Compares the exact-search winner ("original"), its best 1-hop
        refinement ("local"), and replay-bank candidates ("replay") under the
        penalised objective. Accepts the best with strict double-baseline
        gating: a replay candidate must beat both original and local; a local
        edit must beat original by `local_swap_margin`.
        """
        cfg = self.cfg.exact_search
        full_support = build_full_support(self.cfg, original_nonshared)

        # Score the original.
        original_objective = self.boundary_objective(
            task_id, original_nonshared, bundle, refresh_certificates=False
        )
        original_total = float(original_objective["total"])
        original_scored = self._penalise_total(
            base_total=original_total,
            candidate=original_nonshared,
            original=original_nonshared,
            task_id=task_id,
        )

        # Score 1-hop neighbours (local stream).
        local_neighbors = tuple(one_swap_neighbors(self.cfg, original_nonshared))
        local_scored: List[_ScoredCandidate] = []
        if local_neighbors:
            chunk_size = max(1, int(cfg.neighbor_support_batch_size))
            objectives: List[Dict[str, float]] = []
            for chunk, real_count in _candidate_chunks(
                local_neighbors,
                chunk_size,
                pad=True,
            ):
                objectives.extend(
                    self.boundary_objectives(
                        task_id, chunk, bundle, refresh_certificates=False
                    )[:real_count]
                )
            for neighbor, objective in zip(local_neighbors, objectives):
                local_scored.append(
                    self._penalise_total(
                        base_total=float(objective["total"]),
                        candidate=neighbor,
                        original=original_nonshared,
                        task_id=task_id,
                    )
                )
        local_best = min(local_scored, key=lambda c: c.penalised_total) if local_scored else None

        # Score replay-bank candidates.
        replay_candidates = self._propose_from_bank(
            task_id,
            original_nonshared,
            k=int(cfg.replay_topk),
            full_support=full_support,
        )
        replay_scored: List[_ScoredCandidate] = []
        if replay_candidates:
            chunk_size = max(1, int(cfg.neighbor_support_batch_size))
            objectives = []
            for chunk, real_count in _candidate_chunks(
                tuple(replay_candidates),
                chunk_size,
                pad=True,
            ):
                objectives.extend(
                    self.boundary_objectives(
                        task_id, chunk, bundle, refresh_certificates=False
                    )[:real_count]
                )
            for candidate, objective in zip(replay_candidates, objectives):
                replay_scored.append(
                    self._penalise_total(
                        base_total=float(objective["total"]),
                        candidate=candidate,
                        original=original_nonshared,
                        task_id=task_id,
                    )
                )
        replay_best = min(replay_scored, key=lambda c: c.penalised_total) if replay_scored else None

        # Double-baseline acceptance.
        accepted_source = "original"
        accepted_nonshared = tuple(original_nonshared)
        accepted_phi = original_phi
        reason = "original_wins"

        if (
            replay_best is not None
            and replay_best.penalised_total
            < original_scored.penalised_total - cfg.replay_min_gain_over_original
            and (
                local_best is None
                or replay_best.penalised_total
                < local_best.penalised_total - cfg.replay_min_gain_over_local
            )
        ):
            accepted_source = "replay"
            accepted_nonshared = replay_best.nonshared
            reason = "replay_beats_double_baseline"
        elif (
            cfg.prefer_high_overlap_tiebreak
            and replay_best is not None
            and local_best is not None
            and replay_best.penalised_total
            < original_scored.penalised_total - cfg.replay_min_gain_over_original
            and abs(replay_best.penalised_total - local_best.penalised_total)
            <= cfg.replay_min_gain_over_local
            and replay_best.overlap_jaccard > local_best.overlap_jaccard
        ):
            # Replay beats original but ties with local — prefer the higher-overlap repair.
            accepted_source = "replay"
            accepted_nonshared = replay_best.nonshared
            reason = "tiebreak_higher_overlap"
        elif (
            local_best is not None
            and local_best.penalised_total
            < original_scored.penalised_total - cfg.local_swap_margin
        ):
            accepted_source = "local"
            accepted_nonshared = local_best.nonshared
            reason = "local_beats_original_by_margin"

        proposal_row = ReplayProposalRow(
            task_id=int(task_id),
            global_step=int(self.trainer.persistent_state.global_step),
            provenance="boundary",
            original_nonshared=tuple(sorted(original_nonshared)),
            local_baseline_nonshared=local_best.nonshared if local_best is not None else (),
            replay_candidate_nonshared=replay_best.nonshared if replay_best is not None else (),
            original_total=original_scored.base_total,
            local_total=local_best.base_total if local_best is not None else 0.0,
            replay_total=replay_best.base_total if replay_best is not None else 0.0,
            original_penalised=original_scored.penalised_total,
            local_penalised=local_best.penalised_total if local_best is not None else 0.0,
            replay_penalised=replay_best.penalised_total if replay_best is not None else 0.0,
            overlap_jaccard=(
                replay_best.overlap_jaccard if replay_best is not None
                else (local_best.overlap_jaccard if local_best is not None else 1.0)
            ),
            jump_size=(
                replay_best.jump_size if accepted_source == "replay"
                else (local_best.jump_size if accepted_source == "local" else 0)
            ),
            history_intersection=(
                replay_best.history_intersection if replay_best is not None
                else (local_best.history_intersection if local_best is not None else 0)
            ),
            accepted_source=accepted_source,
            accepted_nonshared=accepted_nonshared,
            reason=reason,
        )
        return ReselectionResult(
            accepted_nonshared=accepted_nonshared,
            accepted_phi=accepted_phi,
            accepted_source=accepted_source,
            proposal_row=proposal_row,
        )

    def boundary_search(self, task_id: int) -> Tuple[Tuple[int, ...], PhiConfig]:
        search_started = time.perf_counter()
        bundle = self.make_bundle(task_id, purpose="boundary")
        support_candidates = tuple(enumerate_nonshared_supports(self.cfg))
        phi_candidates = self.phi_candidates()
        if getattr(self.trainer, "run_logger", None) is not None:
            self.trainer.run_logger.event(
                "boundary_search_start",
                task_id=task_id,
                candidate_count=len(support_candidates),
                full_audit_candidate_count=len(support_candidates),
                phi_candidate_count=len(phi_candidates),
            )
        support_rows = self.static_support_scores_batched(
            task_id,
            support_candidates,
            bundle,
        )
        support_rows = self._rank_support_rows(support_rows)
        reserve_triggered, reserve_diag = self._reserve_recruitment_diagnostic(
            task_id,
            support_rows,
        )
        if reserve_triggered:
            reserve_rows = self.static_support_scores_batched(
                task_id,
                enumerate_reserve_recruitment_supports(self.cfg),
                bundle,
                reserve_recruitment_candidate=True,
            )
            support_rows = self._rank_support_rows(
                [*support_rows, *reserve_rows],
                reserve_recruitment_triggered=True,
            )

        support_rows.sort(key=lambda row: row.posterior_energy)
        shortlisted = support_rows[: self.cfg.exact_search.boundary_shortlist]
        shortlisted = [
            SupportSearchRow(**{**row.__dict__, "shortlist_rank": rank})
            for rank, row in enumerate(shortlisted)
        ]
        shortlist_ranks = {row.nonshared: row.shortlist_rank for row in shortlisted}
        support_rows = [
            SupportSearchRow(
                **{
                    **row.__dict__,
                    "shortlist_rank": shortlist_ranks.get(row.nonshared, -1),
                }
            )
            for row in support_rows
        ]
        controller_rows: List[ControllerSearchRow] = []
        best_row = None
        for support_row in shortlisted:
            for phi in phi_candidates:
                row = self.rollout_score(task_id, support_row.nonshared, phi, bundle)
                controller_rows.append(row)
                if getattr(self.trainer, "run_logger", None) is not None:
                    self.trainer.run_logger.event(
                        "rollout_scored",
                        task_id=task_id,
                        support=row.nonshared,
                        rollout_total=row.rollout_total,
                        boundary_total=row.boundary_total,
                        phi=row.phi,
                    )
                if best_row is None or row.rollout_total < best_row.rollout_total:
                    best_row = row
        self.trainer._record_timing(
            task_id,
            "boundary_total_seconds",
            time.perf_counter() - search_started,
        )
        self._ensure_persistent_tables()
        self.trainer.persistent_state.support_tables[task_id] = support_rows
        self.trainer.persistent_state.support_posterior_tables[task_id] = self._posterior_summary(
            task_id,
            support_rows,
            reserve_recruitment_triggered=reserve_triggered,
        )
        self.trainer.persistent_state.reserve_recruitment_tables[task_id] = [reserve_diag]
        self.trainer.persistent_state.controller_tables[task_id] = controller_rows
        if best_row is None:
            fallback = tuple(sorted(self.trainer.current_nonshared))
            self.trainer.set_boundary_choice(task_id, fallback, self.trainer.current_phi)
            if getattr(self.trainer, "run_logger", None) is not None:
                self.trainer.run_logger.event(
                    "boundary_search_done",
                    task_id=task_id,
                    support_rows=len(support_rows),
                    controller_rows=len(controller_rows),
                    best_support=fallback,
                    best_total=None,
                    accepted_source="original",
                )
            return fallback, self.trainer.current_phi

        # V20.2b: route the exact-search winner through replay-bank reselection.
        reselection = self._reselect_with_bank(
            task_id,
            original_nonshared=tuple(best_row.nonshared),
            original_phi=best_row.phi,
            bundle=bundle,
        )
        self.trainer.persistent_state.replay_proposals.setdefault(task_id, []).append(
            reselection.proposal_row
        )
        self.trainer.set_boundary_choice(
            task_id, reselection.accepted_nonshared, reselection.accepted_phi
        )
        # Record the accepted support to the cross-run bank (boundary provenance).
        accepted_objective = self.boundary_objective(
            task_id,
            reselection.accepted_nonshared,
            bundle,
            refresh_certificates=False,
        )
        self._record_replay_row(
            provenance="boundary",
            task_id=task_id,
            nonshared=reselection.accepted_nonshared,
            phi=reselection.accepted_phi,
            objective=accepted_objective,
        )
        if getattr(self.trainer, "run_logger", None) is not None:
            self.trainer.run_logger.event(
                "boundary_search_done",
                task_id=task_id,
                support_rows=len(support_rows),
                controller_rows=len(controller_rows),
                best_support=reselection.accepted_nonshared,
                best_total=best_row.rollout_total,
                accepted_source=reselection.accepted_source,
                exact_winner=tuple(best_row.nonshared),
                reselection_reason=reselection.proposal_row.reason,
            )
        return reselection.accepted_nonshared, reselection.accepted_phi

    def demotion_swap_audit(
        self,
        task_id: int,
        final_state,
        nonshared: Sequence[int],
    ) -> List[DemotionSwapAuditRow]:
        """Audit a small set of internal demotion swaps before mutating shells."""
        if not self.cfg.exact_search.enable_demotion_swap_audit:
            return []
        self._ensure_persistent_tables()
        candidates = self.trainer.shell_controller.demotion_swap_candidates(
            self.trainer.params,
            final_state,
            self.trainer.active_full_support(nonshared),
            self.trainer.current_phi,
            max_candidates=self.cfg.exact_search.demotion_audit_max_candidates,
        )
        if not candidates:
            return []

        started_at = time.perf_counter()
        bundle = self.make_bundle(task_id, purpose="demotion")
        current_support = tuple(sorted(nonshared))
        round_index = 1 + len(
            {row.round_index for row in self.trainer.persistent_state.demotion_swap_tables.get(task_id, [])}
        )
        baseline_objective = self.boundary_objective(
            task_id,
            current_support,
            bundle,
            refresh_certificates=False,
        )
        baseline_semantic = (
            self.trainer.shell_controller.semantic_penalty(
                self.trainer.persistent_state,
                self.trainer.active_full_support(current_support),
            )
            if self.cfg.exact_search.log_semantic_penalty
            else 0.0
        )
        baseline_total = float(baseline_objective["total"] + baseline_semantic)
        rows: List[DemotionSwapAuditRow] = []
        best_row = None
        for candidate in candidates:
            clone = self.trainer.clone()
            clone.params = clone.shell_controller.apply_demotion_swap(
                clone.params,
                node_name=str(candidate["node_name"]),
                inner_shell=str(candidate["inner_shell"]),
                outer_shell=str(candidate["outer_shell"]),
                inner_index=int(candidate["inner_index"]),
                outer_index=int(candidate["outer_index"]),
            )
            clone.persistent_state.params = clone.params
            clone._bump_params_revision()
            candidate_objective = self.boundary_objective(
                task_id,
                current_support,
                bundle,
                trainer=clone,
                refresh_certificates=True,
            )
            candidate_semantic = (
                clone.shell_controller.semantic_penalty(
                    clone.persistent_state,
                    clone.active_full_support(current_support),
                )
                if self.cfg.exact_search.log_semantic_penalty
                else 0.0
            )
            candidate_total = float(candidate_objective["total"] + candidate_semantic)
            row = DemotionSwapAuditRow(
                task_id=task_id,
                round_index=round_index,
                global_step=self.trainer.persistent_state.global_step,
                column_index=int(candidate["column_index"]),
                node_name=str(candidate["node_name"]),
                inner_shell=str(candidate["inner_shell"]),
                outer_shell=str(candidate["outer_shell"]),
                inner_index=int(candidate["inner_index"]),
                outer_index=int(candidate["outer_index"]),
                baseline_total=baseline_total,
                candidate_total=candidate_total,
                gain=float(baseline_total - candidate_total),
                accepted=False,
            )
            rows.append(row)
            if best_row is None or row.gain > best_row.gain:
                best_row = row
            del clone

        if best_row is not None and best_row.gain > self.cfg.exact_search.demotion_swap_margin:
            self.trainer.params = self.trainer.shell_controller.apply_demotion_swap(
                self.trainer.params,
                node_name=best_row.node_name,
                inner_shell=best_row.inner_shell,
                outer_shell=best_row.outer_shell,
                inner_index=best_row.inner_index,
                outer_index=best_row.outer_index,
            )
            self.trainer.persistent_state.params = self.trainer.params
            self.trainer._bump_params_revision()
            rows = [
                DemotionSwapAuditRow(
                    **{
                        **row.__dict__,
                        "accepted": (
                            row.node_name == best_row.node_name
                            and row.inner_shell == best_row.inner_shell
                            and row.outer_shell == best_row.outer_shell
                            and row.inner_index == best_row.inner_index
                            and row.outer_index == best_row.outer_index
                        ),
                    }
                )
                for row in rows
            ]

        history = list(self.trainer.persistent_state.demotion_swap_tables.get(task_id, []))
        self.trainer.persistent_state.demotion_swap_tables[task_id] = history + rows
        self.trainer._record_timing(
            task_id,
            "demotion_swap_seconds",
            time.perf_counter() - started_at,
            accumulate=True,
        )
        if getattr(self.trainer, "run_logger", None) is not None:
            self.trainer.run_logger.event(
                "demotion_swap_audit_done",
                task_id=task_id,
                round_index=round_index,
                candidate_count=len(rows),
                accepted=any(row.accepted for row in rows),
                best_gain=best_row.gain if best_row is not None else None,
            )
        gc.collect()
        return rows

    def _ensure_persistent_tables(self) -> None:
        defaults = {
            "support_posterior_tables": {},
            "reserve_recruitment_tables": {},
            "controller_tables": {},
            "local_swap_tables": {},
            "demotion_swap_tables": {},
            "last_demotion_audit_step": {},
            "replay_proposals": {},
            "recently_demoted": {},
        }
        for name, value in defaults.items():
            if not hasattr(self.trainer.persistent_state, name):
                setattr(self.trainer.persistent_state, name, value)

    def local_one_swap(self, task_id: int) -> Tuple[int, ...]:
        """V20.2b in-task support refinement.

        Merges V18 1-hop neighbours with replay-bank candidates filtered by
        `replay_overlap_floor`, scores the union under the penalised objective,
        and accepts the best candidate if it beats current by `local_swap_margin`.
        """
        self._ensure_persistent_tables()
        if task_id == 0:
            return self.trainer.current_nonshared

        bundle = self.make_bundle(task_id, purpose="local_swap")
        current_nonshared = tuple(
            sorted(
                self.trainer.persistent_state.current_support.get(
                    task_id, self.trainer.current_nonshared
                )
            )
        )
        local_neighbors = tuple(one_swap_neighbors(self.cfg, current_nonshared))

        # Bank candidates filtered by overlap floor.
        full_support = build_full_support(self.cfg, current_nonshared)
        bank_proposals = self._propose_from_bank(
            task_id,
            current_nonshared,
            k=int(self.cfg.exact_search.replay_topk),
            full_support=full_support,
        )
        overlap_floor = float(self.cfg.exact_search.replay_overlap_floor)
        bank_candidates: List[Tuple[int, ...]] = []
        bank_overlap: Dict[Tuple[int, ...], float] = {}
        seen = set(local_neighbors) | {current_nonshared}
        for cand in bank_proposals:
            if cand in seen:
                continue
            overlap = self._jaccard(cand, current_nonshared)
            bank_overlap[cand] = overlap
            if overlap < overlap_floor:
                continue
            bank_candidates.append(cand)
            seen.add(cand)

        round_index = 1 + len(
            {
                row.round_index
                for row in self.trainer.persistent_state.local_swap_tables.get(task_id, [])
            }
        )
        started_at = time.perf_counter()
        all_supports = [current_nonshared] + list(local_neighbors) + list(bank_candidates)
        all_objectives: List[Dict[str, float]] = []
        for chunk, real_count in _candidate_chunks(
            all_supports,
            self.cfg.exact_search.neighbor_support_batch_size,
            pad=True,
        ):
            all_objectives.extend(
                self.boundary_objectives(task_id, chunk, bundle, refresh_certificates=False)[
                    :real_count
                ]
            )

        log_semantic = bool(self.cfg.exact_search.log_semantic_penalty)
        current_objective = all_objectives[0]
        current_semantic = (
            self.trainer.shell_controller.semantic_penalty(
                self.trainer.persistent_state,
                self.trainer.active_full_support(current_nonshared),
            )
            if log_semantic
            else 0.0
        )
        current_base = float(current_objective["total"] + current_semantic)
        current_scored = self._penalise_total(
            base_total=current_base,
            candidate=current_nonshared,
            original=current_nonshared,
            task_id=task_id,
        )

        candidate_supports = list(local_neighbors) + list(bank_candidates)
        candidate_objectives = all_objectives[1:]
        rows: List[LocalSwapRow] = []
        best_row: Optional[LocalSwapRow] = None
        best_score: Optional[_ScoredCandidate] = None
        best_is_bank = False

        for cand, objective in zip(candidate_supports, candidate_objectives):
            semantic_regularizer = (
                self.trainer.shell_controller.semantic_penalty(
                    self.trainer.persistent_state,
                    self.trainer.active_full_support(cand),
                )
                if log_semantic
                else 0.0
            )
            cand_base = float(objective["total"] + semantic_regularizer)
            scored = self._penalise_total(
                base_total=cand_base,
                candidate=cand,
                original=current_nonshared,
                task_id=task_id,
            )
            gain_penalised = current_scored.penalised_total - scored.penalised_total
            row = LocalSwapRow(
                task_id=task_id,
                round_index=round_index,
                current_nonshared=current_nonshared,
                candidate_nonshared=tuple(sorted(cand)),
                total=scored.penalised_total,
                boundary_total=float(objective["total"]),
                current_loss=float(
                    objective["current_first_loss"] + objective["current_remaining_loss"]
                ),
                old_worst_loss=objective["old_worst_loss"],
                old_mix_loss=objective["old_mix_loss"],
                switch_penalty=objective["switch_penalty"],
                semantic_regularizer=float(semantic_regularizer),
                gain=float(gain_penalised),
                accepted=False,
            )
            rows.append(row)
            if best_row is None or row.gain > best_row.gain:
                best_row = row
                best_score = scored
                best_is_bank = cand in set(bank_candidates)

        accepted_support = current_nonshared
        if (
            best_row is not None
            and best_row.gain > self.cfg.exact_search.local_swap_margin
        ):
            accepted_support = best_row.candidate_nonshared
            rows = [
                LocalSwapRow(
                    **{
                        **row.__dict__,
                        "accepted": row.candidate_nonshared == accepted_support,
                    }
                )
                for row in rows
            ]
            self.trainer.set_current_support(
                task_id, accepted_support, self.trainer.current_phi
            )
            # Record acceptance to the cross-run bank.
            self._record_replay_row(
                provenance="local_swap",
                task_id=task_id,
                nonshared=accepted_support,
                phi=self.trainer.current_phi,
                objective={
                    "total": best_row.boundary_total,
                    "old_worst_loss": best_row.old_worst_loss,
                    "old_mix_loss": best_row.old_mix_loss,
                },
            )
            # Audit row for the reselection log: which stream the swap came from.
            proposal_row = ReplayProposalRow(
                task_id=int(task_id),
                global_step=int(self.trainer.persistent_state.global_step),
                provenance="local_swap",
                original_nonshared=current_nonshared,
                local_baseline_nonshared=accepted_support if not best_is_bank else (),
                replay_candidate_nonshared=accepted_support if best_is_bank else (),
                original_total=current_scored.base_total,
                local_total=(0.0 if best_is_bank or best_score is None else best_score.base_total),
                replay_total=(best_score.base_total if best_is_bank and best_score else 0.0),
                original_penalised=current_scored.penalised_total,
                local_penalised=(
                    0.0 if best_is_bank or best_score is None else best_score.penalised_total
                ),
                replay_penalised=(
                    best_score.penalised_total if best_is_bank and best_score else 0.0
                ),
                overlap_jaccard=(
                    best_score.overlap_jaccard if best_score is not None else 1.0
                ),
                jump_size=(best_score.jump_size if best_score is not None else 0),
                history_intersection=(
                    best_score.history_intersection if best_score is not None else 0
                ),
                accepted_source=("replay" if best_is_bank else "local"),
                accepted_nonshared=accepted_support,
                reason="local_swap_accepted",
            )
            self.trainer.persistent_state.replay_proposals.setdefault(task_id, []).append(
                proposal_row
            )

        history = list(self.trainer.persistent_state.local_swap_tables.get(task_id, []))
        self.trainer.persistent_state.local_swap_tables[task_id] = history + rows
        self.trainer.persistent_state.latest_local_swap = max(
            rows, key=lambda row: row.gain, default=None
        )
        self.trainer._record_timing(
            task_id,
            "local_swap_seconds",
            time.perf_counter() - started_at,
            accumulate=True,
        )
        if getattr(self.trainer, "run_logger", None) is not None:
            self.trainer.run_logger.event(
                "local_one_swap_done",
                task_id=task_id,
                round_index=round_index,
                candidate_count=len(rows),
                bank_candidate_count=len(bank_candidates),
                accepted_support=accepted_support,
                best_gain=best_row.gain if best_row is not None else None,
                accepted_source=("replay" if best_is_bank and accepted_support != current_nonshared else "local" if accepted_support != current_nonshared else "original"),
            )
        return accepted_support


def _clip(value: float, bounds: Tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _phi_l1_distance(left: PhiConfig, right: PhiConfig) -> float:
    return float(
        abs(left.outer_quantile - right.outer_quantile)
        + abs(left.middle_quantile - right.middle_quantile)
        + abs(left.replacement_margin_base - right.replacement_margin_base)
        + abs(left.demotion_min_role_gain - right.demotion_min_role_gain)
    )


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _candidate_chunks(
    items: Sequence[Sequence[int]],
    size: int,
    *,
    pad: bool,
) -> Iterable[Tuple[Tuple[Tuple[int, ...], ...], int]]:
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
