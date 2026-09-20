"""Exact support search, replay-bank reselection, and one-swap maintenance for HiBaCaML."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from hibacaml.config import HiBaCaMLConfig, PhiConfig
from hibacaml.control.replay_bank import (
    SelectorBank,
    build_context,
    make_replay_row,
)
from hibacaml.control.ranking import (
    candidate_chunks,
    jaccard,
    jump_size,
    mean_or_zero,
    phi_candidates,
    phi_l1_distance,
    posterior_summary,
    rank_support_rows,
    reserve_recruitment_diagnostic,
)
from hibacaml.control.scoring import SupportScorer
from hibacaml.control.support import (
    build_full_support,
    enumerate_nonshared_supports,
    enumerate_reserve_recruitment_supports,
    one_swap_neighbors,
)
from hibacaml.reporting import append_event, rollout_logging
from hibacaml.types import (
    BoundaryBundle,
    ControllerSearchRow,
    DemotionSwapAuditRow,
    LocalSwapRow,
    ReplayProposalRow,
)


@dataclass(frozen=True)
class _ScoredCandidate:
    """One scored support candidate for the reselection layer."""

    nonshared: Tuple[int, ...]
    base_total: float
    overlap_jaccard: float
    jump_size: int
    history_intersection: int
    penalised_total: float


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
        self.scorer = SupportScorer(cfg)


    def rollout_score(
        self,
        task_id: int,
        support_cols: Sequence[int],
        phi: PhiConfig,
        bundle: BoundaryBundle,
    ) -> ControllerSearchRow:
        with rollout_logging():
            return self._rollout_score(task_id, support_cols, phi, bundle)

    def _rollout_score(
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
            # Use the clone's normal learner update for rollouts.
            _, final_state = clone.training_update(
                batch,
                task,
                support_cols,
            )
            updated_params = clone.shell_controller.apply_structural_edits(
                clone.params,
                final_state,
                clone.persistent_state,
                clone.active_full_support(support_cols),
                phi,
            )
            if updated_params is not clone.params:
                clone.params = updated_params
                clone.persistent_state.params = clone.params
                clone._bump_params_revision()
        if final_state is None and bundle.current_eval:
            final_state, _ = clone.run_batch_evaluation_inference(
                bundle.current_eval[0],
                task,
                support_cols,
            )
        objective = self.scorer.objective(
            clone,
            task_id,
            support_cols,
            bundle,
            refresh_certificates=True,
        )
        semantic_regularizer = clone.certificate_controller.semantic_penalty(
            clone.persistent_state,
            clone.active_full_support(support_cols),
        )
        phi_l1_penalty = self.cfg.exact_search.controller_l1_penalty * phi_l1_distance(
            phi,
            self.cfg.phi,
        )
        if final_state is not None:
            composer_diag = clone.composer_diagnostics_from_state(final_state, task, support_cols)
        else:
            composer_diag = {
                "gate_entropy": 0.0,
                "gate_deviation": 0.0,
                "cert_prior_mean": 0.0,
            }
        certs = list(clone.persistent_state.certificates.values())
        occ_tier1 = mean_or_zero([cert.tier_occ[0] for cert in certs])
        occ_tier2 = mean_or_zero([cert.tier_occ[1] for cert in certs])
        occ_tier3 = mean_or_zero([cert.tier_occ[2] for cert in certs])
        q_tier1 = mean_or_zero([cert.tier_q[0] for cert in certs])
        q_tier2 = mean_or_zero([cert.tier_q[1] for cert in certs])
        q_tier3 = mean_or_zero([cert.tier_q[2] for cert in certs])
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
        overlap = jaccard(candidate, original)
        jump = jump_size(candidate, original)
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

    def _objectives_in_chunks(
        self,
        task_id: int,
        supports: Sequence[Sequence[int]],
        bundle: BoundaryBundle,
    ) -> List[Dict[str, float]]:
        """Score supports in configured-width chunks, dropping the padding."""
        objectives: List[Dict[str, float]] = []
        for chunk, real_count in candidate_chunks(
            tuple(supports),
            self.cfg.exact_search.neighbor_support_batch_size,
            pad=True,
        ):
            objectives.extend(
                self.scorer.objectives(
                    self.trainer, task_id, chunk, bundle, refresh_certificates=False
                )[:real_count]
            )
        return objectives

    def _audited_semantic_penalty(self, trainer, support_cols: Sequence[int]) -> float:
        """Semantic regularizer for the audit objectives, or 0.0 when disabled.

        The rollout objective always carries this term; the local-swap and
        demotion audits carry it only under `log_semantic_penalty`. That
        difference is deliberate, so this covers the gated callers only.
        """
        if not self.cfg.exact_search.log_semantic_penalty:
            return 0.0
        return trainer.certificate_controller.semantic_penalty(
            trainer.persistent_state,
            trainer.active_full_support(support_cols),
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
    ) -> Tuple[Tuple[int, ...], PhiConfig, str, ReplayProposalRow]:
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
        original_objective = self.scorer.objective(
            self.trainer, task_id, original_nonshared, bundle, refresh_certificates=False
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
            objectives = self._objectives_in_chunks(task_id, local_neighbors, bundle)
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
            objectives = self._objectives_in_chunks(task_id, replay_candidates, bundle)
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
        return accepted_nonshared, accepted_phi, accepted_source, proposal_row

    def boundary_search(self, task_id: int) -> Tuple[Tuple[int, ...], PhiConfig]:
        search_started = time.perf_counter()
        bundle = self.scorer.make_bundle(self.trainer, task_id, purpose="boundary")
        support_candidates = tuple(enumerate_nonshared_supports(self.cfg))
        phi_neighbourhood = phi_candidates(self.cfg, self.trainer.current_phi)
        if self.trainer.run_root is not None:
            append_event(
                self.trainer.run_root,
                "boundary_search_start",
                task_id=task_id,
                candidate_count=len(support_candidates),
                full_audit_candidate_count=len(support_candidates),
                phi_candidate_count=len(phi_neighbourhood),
            )
        support_rows = self.scorer.static_rows(
            self.trainer,
            task_id,
            support_candidates,
            bundle,
        )
        support_rows = rank_support_rows(self.cfg, support_rows)
        reserve_triggered, reserve_diag = reserve_recruitment_diagnostic(
            self.cfg,
            self.trainer.persistent_state,
            task_id,
            support_rows,
        )
        if reserve_triggered:
            reserve_rows = self.scorer.static_rows(
                self.trainer,
                task_id,
                enumerate_reserve_recruitment_supports(self.cfg),
                bundle,
                reserve_recruitment_candidate=True,
            )
            support_rows = rank_support_rows(self.cfg, [*support_rows, *reserve_rows])

        support_rows.sort(key=lambda row: row.posterior_energy)
        # Rank once, then take the shortlist off the ranked list: the sliced rows
        # carry the same 0..N-1 ranks they were previously rebuilt with.
        shortlist_ranks = {
            row.nonshared: rank
            for rank, row in enumerate(
                support_rows[: self.cfg.exact_search.boundary_shortlist]
            )
        }
        support_rows = [
            replace(row, shortlist_rank=shortlist_ranks.get(row.nonshared, -1))
            for row in support_rows
        ]
        shortlisted = support_rows[: self.cfg.exact_search.boundary_shortlist]
        controller_rows: List[ControllerSearchRow] = []
        best_row = None
        for support_row in shortlisted:
            for phi in phi_neighbourhood:
                row = self.rollout_score(task_id, support_row.nonshared, phi, bundle)
                controller_rows.append(row)
                if self.trainer.run_root is not None:
                    append_event(
                        self.trainer.run_root,
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
        self.trainer.persistent_state.support_tables[task_id] = support_rows
        self.trainer.persistent_state.support_posterior_tables[task_id] = posterior_summary(
            task_id,
            support_rows,
            reserve_recruitment_triggered=reserve_triggered,
        )
        self.trainer.persistent_state.reserve_recruitment_tables[task_id] = [reserve_diag]
        self.trainer.persistent_state.controller_tables[task_id] = controller_rows
        if best_row is None:
            fallback = tuple(sorted(self.trainer.current_nonshared))
            self.trainer.set_boundary_choice(task_id, fallback, self.trainer.current_phi)
            if self.trainer.run_root is not None:
                append_event(
                    self.trainer.run_root,
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
        accepted_nonshared, accepted_phi, accepted_source, proposal_row = self._reselect_with_bank(
            task_id,
            original_nonshared=tuple(best_row.nonshared),
            original_phi=best_row.phi,
            bundle=bundle,
        )
        self.trainer.persistent_state.replay_proposals.setdefault(task_id, []).append(
            proposal_row
        )
        self.trainer.set_boundary_choice(
            task_id, accepted_nonshared, accepted_phi
        )
        # Record the accepted support to the cross-run bank (boundary provenance).
        accepted_objective = self.scorer.objective(
            self.trainer,
            task_id,
            accepted_nonshared,
            bundle,
            refresh_certificates=False,
        )
        self._record_replay_row(
            provenance="boundary",
            task_id=task_id,
            nonshared=accepted_nonshared,
            phi=accepted_phi,
            objective=accepted_objective,
        )
        if self.trainer.run_root is not None:
            append_event(
                self.trainer.run_root,
                "boundary_search_done",
                task_id=task_id,
                support_rows=len(support_rows),
                controller_rows=len(controller_rows),
                best_support=accepted_nonshared,
                best_total=best_row.rollout_total,
                accepted_source=accepted_source,
                exact_winner=tuple(best_row.nonshared),
                reselection_reason=proposal_row.reason,
            )
        return accepted_nonshared, accepted_phi

    def demotion_swap_audit(
        self,
        task_id: int,
        final_state,
        nonshared: Sequence[int],
    ) -> List[DemotionSwapAuditRow]:
        """Audit a small set of internal demotion swaps before mutating shells."""
        if not self.cfg.exact_search.enable_demotion_swap_audit:
            return []
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
        bundle = self.scorer.make_bundle(self.trainer, task_id, purpose="demotion")
        current_support = tuple(sorted(nonshared))
        round_index = 1 + len(
            {row.round_index for row in self.trainer.persistent_state.demotion_swap_tables.get(task_id, [])}
        )
        baseline_objective = self.scorer.objective(
            self.trainer,
            task_id,
            current_support,
            bundle,
            refresh_certificates=False,
        )
        baseline_semantic = self._audited_semantic_penalty(self.trainer, current_support)
        baseline_total = float(baseline_objective["total"] + baseline_semantic)
        rows: List[DemotionSwapAuditRow] = []
        best_row = None
        # Each candidate is scored on a clone; demote their progress lines.
        with rollout_logging():
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
                candidate_objective = self.scorer.objective(
                    clone,
                    task_id,
                    current_support,
                    bundle,
                    refresh_certificates=True,
                )
                candidate_semantic = self._audited_semantic_penalty(clone, current_support)
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
                replace(
                    row,
                    accepted=(
                        row.node_name == best_row.node_name
                        and row.inner_shell == best_row.inner_shell
                        and row.outer_shell == best_row.outer_shell
                        and row.inner_index == best_row.inner_index
                        and row.outer_index == best_row.outer_index
                    ),
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
        if self.trainer.run_root is not None:
            append_event(
                self.trainer.run_root,
                "demotion_swap_audit_done",
                task_id=task_id,
                round_index=round_index,
                candidate_count=len(rows),
                accepted=any(row.accepted for row in rows),
                best_gain=best_row.gain if best_row is not None else None,
            )
        gc.collect()
        return rows

    def local_one_swap(self, task_id: int) -> Tuple[int, ...]:
        """V20.2b in-task support refinement.

        Merges V18 1-hop neighbours with replay-bank candidates filtered by
        `replay_overlap_floor`, scores the union under the penalised objective,
        and accepts the best candidate if it beats current by `local_swap_margin`.
        """
        if task_id == 0:
            return self.trainer.current_nonshared

        bundle = self.scorer.make_bundle(self.trainer, task_id, purpose="local_swap")
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
        seen = set(local_neighbors) | {current_nonshared}
        for cand in bank_proposals:
            if cand in seen:
                continue
            if jaccard(cand, current_nonshared) < overlap_floor:
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
        all_objectives = self._objectives_in_chunks(task_id, all_supports, bundle)

        current_objective = all_objectives[0]
        current_semantic = self._audited_semantic_penalty(self.trainer, current_nonshared)
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
            semantic_regularizer = self._audited_semantic_penalty(self.trainer, cand)
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
                replace(row, accepted=row.candidate_nonshared == accepted_support)
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
        if self.trainer.run_root is not None:
            append_event(
                self.trainer.run_root,
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
