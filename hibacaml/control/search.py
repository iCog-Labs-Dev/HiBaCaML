"""Exact support search and one-swap maintenance for HiBaCaML."""

from __future__ import annotations

import gc
import math
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import jax.numpy as jnp

from hibacaml.config import HiBaCaMLConfig, PhiConfig
from hibacaml.control.support import (
    enumerate_nonshared_supports,
    enumerate_reserve_recruitment_supports,
    one_swap_neighbors,
)
from hibacaml.types import (
    BoundaryBundle,
    ControllerSearchRow,
    DemotionSwapAuditRow,
    LocalSwapRow,
    ReserveRecruitmentRow,
    SupportPosteriorSummary,
    SupportSearchRow,
)


class ExactSearchService:
    """Exact support search service."""

    def __init__(self, cfg: HiBaCaMLConfig, trainer):
        self.cfg = cfg
        self.trainer = trainer

    def make_bundle(self, task_id: int) -> BoundaryBundle:
        """Build the audit bundle: current eval batches + worst-old + mixed-old."""
        task = self.trainer.task(task_id)
        current_eval = tuple(
            batch
            for idx, batch in enumerate(task.test_loader)
            if idx < self.cfg.exact_search.boundary_current_batches
        )
        train_batches = tuple(
            batch
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
                )
                if worst_old_eval is not None:
                    worst_old_eval = (worst_old, worst_old_eval)
            mixed_old_eval = list(self._build_mixed_old_fragments(task_id))
        return BoundaryBundle(
            current_eval=current_eval,
            train_batches=train_batches,
            worst_old_eval=worst_old_eval,
            mixed_old_eval=tuple(mixed_old_eval),
            worst_old=worst_old,
        )

    def _concat_loader_prefix(self, loader, max_batches: int):
        pieces = []
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
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
        triggered = (
            self.cfg.exact_search.support_candidate_policy == "adaptive_reserve_gated"
            and bool(self.cfg.column_pool.reserve_indices)
            and saturation_ok
            and confidence_weak
        )
        if triggered:
            reason = "saturation_and_weak_posterior"
        elif self.cfg.exact_search.support_candidate_policy != "adaptive_reserve_gated":
            reason = "policy_disabled"
        elif not saturation_ok:
            reason = "saturation_below_threshold"
        elif not confidence_weak:
            reason = "posterior_confident"
        else:
            reason = "no_reserve_columns"
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
        old_worst = [0.0 for _ in range(count)]
        old_mix = [0.0 for _ in range(count)]

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

        if bundle.worst_old_eval is not None:
            prev_task_id, batch = bundle.worst_old_eval
            old_worst = [
                float(loss)
                for loss in self._evaluate_batch_losses(
                    trainer,
                    trainer.task(prev_task_id),
                    batch,
                    support_list,
                    refresh_certificates=False,
                )
            ]

        if bundle.mixed_old_eval:
            mix_sums = [0.0 for _ in range(count)]
            for prev_task_id, batch in bundle.mixed_old_eval:
                losses = self._evaluate_batch_losses(
                    trainer,
                    trainer.task(prev_task_id),
                    batch,
                    support_list,
                    refresh_certificates=False,
                )
                for row_idx, loss in enumerate(losses):
                    mix_sums[row_idx] += float(loss)
            old_mix = [value / max(len(bundle.mixed_old_eval), 1) for value in mix_sums]

        rows = []
        for idx, support_cols in enumerate(support_list):
            switch_penalty = self._switch_penalty(task_id, support_cols)
            total = (
                current_first[idx]
                + current_remaining[idx]
                + self.cfg.exact_search.exact_old_worst_weight * old_worst[idx]
                + self.cfg.exact_search.exact_old_mix_weight * old_mix[idx]
                + switch_penalty
            )
            rows.append(
                {
                    "current_first_loss": float(current_first[idx]),
                    "current_remaining_loss": float(current_remaining[idx]),
                    "old_worst_loss": float(old_worst[idx]),
                    "old_mix_loss": float(old_mix[idx]),
                    "switch_penalty": float(switch_penalty),
                    "total": float(total),
                }
            )
        return rows

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
            pad=bool(self.cfg.exact_search.pad_support_batches),
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
            final_state, _ = clone.run_batch_inference(bundle.current_eval[0], task, support_cols)
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

    def boundary_search(self, task_id: int) -> Tuple[Tuple[int, ...], PhiConfig]:
        search_started = time.perf_counter()
        bundle = self.make_bundle(task_id)
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
        reserve_triggered = False
        reserve_diag: ReserveRecruitmentRow | None = None
        if self.cfg.exact_search.support_candidate_policy == "adaptive_reserve_gated":
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
        if reserve_diag is None:
            _, reserve_diag = self._reserve_recruitment_diagnostic(task_id, support_rows)

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
        if getattr(self.trainer, "run_logger", None) is not None:
            self.trainer.run_logger.event(
                "boundary_search_done",
                task_id=task_id,
                support_rows=len(support_rows),
                controller_rows=len(controller_rows),
                best_support=best_row.nonshared if best_row is not None else None,
                best_total=best_row.rollout_total if best_row is not None else None,
            )
        if best_row is None:
            fallback = tuple(sorted(self.trainer.current_nonshared))
            self.trainer.set_boundary_choice(task_id, fallback, self.trainer.current_phi)
            return fallback, self.trainer.current_phi
        self.trainer.set_boundary_choice(task_id, best_row.nonshared, best_row.phi)
        return best_row.nonshared, best_row.phi

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
        bundle = self.make_bundle(task_id)
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
        }
        for name, value in defaults.items():
            if not hasattr(self.trainer.persistent_state, name):
                setattr(self.trainer.persistent_state, name, value)

    def local_one_swap(self, task_id: int) -> Tuple[int, ...]:
        """Evaluate one-swap neighbors under the Eq. (1) boundary objective."""
        if not self.cfg.exact_search.enable_local_maintenance:
            return self.trainer.current_nonshared
        if task_id == 0:
            return self.trainer.current_nonshared

        bundle = self.make_bundle(task_id)
        current_nonshared = self.trainer.persistent_state.current_support.get(task_id, self.trainer.current_nonshared)
        neighbors = tuple(one_swap_neighbors(self.cfg, current_nonshared))
        round_index = 1 + len(
            {row.round_index for row in self.trainer.persistent_state.local_swap_tables.get(task_id, [])}
        )
        started_at = time.perf_counter()
        # Score current support + all 45 neighbors in batched JAX calls (same batch
        # size as static scoring) instead of 46 individual boundary_objective() dispatches.
        all_supports = [current_nonshared] + list(neighbors)
        all_objectives: List[Dict[str, float]] = []
        for chunk, real_count in _candidate_chunks(
            all_supports,
            self.cfg.exact_search.static_support_batch_size,
            pad=bool(self.cfg.exact_search.pad_support_batches),
        ):
            all_objectives.extend(
                self.boundary_objectives(task_id, chunk, bundle, refresh_certificates=False)[
                    :real_count
                ]
            )
        current_objective = all_objectives[0]
        current_semantic = (
            self.trainer.shell_controller.semantic_penalty(
                self.trainer.persistent_state,
                self.trainer.active_full_support(current_nonshared),
            )
            if self.cfg.exact_search.log_semantic_penalty
            else 0.0
        )
        current_total = current_objective["total"] + current_semantic
        rows: List[LocalSwapRow] = []
        best_row = None
        for neighbor, objective in zip(neighbors, all_objectives[1:]):
            semantic_regularizer = (
                self.trainer.shell_controller.semantic_penalty(
                    self.trainer.persistent_state,
                    self.trainer.active_full_support(neighbor),
                )
                if self.cfg.exact_search.log_semantic_penalty
                else 0.0
            )
            total = objective["total"] + semantic_regularizer
            gain = current_total - total
            row = LocalSwapRow(
                task_id=task_id,
                round_index=round_index,
                current_nonshared=tuple(sorted(current_nonshared)),
                candidate_nonshared=tuple(sorted(neighbor)),
                total=float(total),
                boundary_total=float(objective["total"]),
                current_loss=float(objective["current_first_loss"] + objective["current_remaining_loss"]),
                old_worst_loss=objective["old_worst_loss"],
                old_mix_loss=objective["old_mix_loss"],
                switch_penalty=objective["switch_penalty"],
                semantic_regularizer=float(semantic_regularizer),
                gain=float(gain),
                accepted=False,
            )
            rows.append(row)
            if best_row is None or row.gain > best_row.gain:
                best_row = row

        accepted_support = tuple(sorted(current_nonshared))
        if best_row is not None and best_row.gain > self.cfg.exact_search.local_swap_margin:
            accepted_support = best_row.candidate_nonshared
            rows = [
                LocalSwapRow(**{**row.__dict__, "accepted": row.candidate_nonshared == accepted_support})
                for row in rows
            ]
            self.trainer.set_current_support(task_id, accepted_support, self.trainer.current_phi)

        history = list(self.trainer.persistent_state.local_swap_tables.get(task_id, []))
        self.trainer.persistent_state.local_swap_tables[task_id] = history + rows
        self.trainer.persistent_state.latest_local_swap = max(rows, key=lambda row: row.gain, default=None)
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
                accepted_support=accepted_support,
                best_gain=best_row.gain if best_row is not None else None,
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
