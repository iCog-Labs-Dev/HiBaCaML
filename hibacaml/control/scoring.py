"""Evaluate support candidates for exact search."""

from __future__ import annotations

from itertools import islice
from typing import Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp

from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.ranking import candidate_chunks
from hibacaml.types import BoundaryBundle, SupportSearchRow


class SupportScorer:
    """Builds audit bundles and scores support candidates against them."""

    def __init__(self, cfg: HiBaCaMLConfig):
        self.cfg = cfg
        self._old_audit_cache: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}


    # Bundles

    def make_bundle(
        self,
        trainer,
        task_id: int,
        *,
        purpose: str = "boundary",
    ) -> BoundaryBundle:
        """Build the audit bundle: current eval batches + worst-old + mixed-old."""
        # One bundle owns the cache: entries keyed on a freed bundle's address
        # are unreachable, and clearing here keeps id(bundle) sound.
        self._old_audit_cache.clear()
        caps = self._bundle_batch_caps(purpose)
        task = trainer.task(task_id)
        current_eval = tuple(
            self._slice_batch(batch, caps["current"])
            for batch in islice(
                task.test_loader, self.cfg.exact_search.boundary_current_batches
            )
        )
        train_batches = ()
        if caps["include_rollout"]:
            train_batches = tuple(
                self._slice_batch(batch, caps["rollout"])
                for batch in islice(
                    task.train_loader, self.cfg.exact_search.boundary_rollout_steps
                )
            )

        worst_old = None
        worst_old_eval = None
        mixed_old_eval: List[Tuple[int, Dict[str, jnp.ndarray]]] = []
        if task_id > 0:
            saved = trainer.evaluate_all_saved_supports()
            if saved:
                worst_old = min(saved, key=lambda key: saved[key]["accuracy"])
            if worst_old is not None:
                worst_old_task = trainer.task(worst_old)
                worst_old_eval = self._concat_loader_prefix(
                    worst_old_task.test_loader,
                    self.cfg.exact_search.boundary_old_batches,
                    batch_size=caps["worst_old"],
                )
                if worst_old_eval is not None:
                    worst_old_eval = (worst_old, worst_old_eval)
            mixed_old_eval = list(
                self._build_mixed_old_fragments(
                    trainer,
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

    def _bundle_batch_caps(self, purpose: str) -> Dict[str, Optional[int] | bool]:
        cfg = self.cfg.exact_search
        if purpose == "boundary":
            return {
                "current": cfg.boundary_current_data_batch_size,
                "rollout": cfg.rollout_train_data_batch_size,
                "worst_old": cfg.boundary_worst_old_data_batch_size,
                "mixed_old": cfg.boundary_mixed_old_data_batch_size,
                "include_rollout": True,
            }
        if purpose == "local_swap":
            cap = cfg.local_swap_audit_data_batch_size
            return {
                "current": cap,
                "rollout": None,
                "worst_old": cap,
                "mixed_old": cap,
                "include_rollout": False,
            }
        if purpose == "demotion":
            cap = cfg.demotion_audit_data_batch_size
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
        trainer,
        task_id: int,
        *,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tuple[int, Dict[str, jnp.ndarray]], ...]:
        """Build one deterministic mixed-old batch from all prior tasks."""
        prior_batches: List[Tuple[int, Dict[str, jnp.ndarray]]] = []
        for prev_task_id in range(task_id):
            batch = next(iter(trainer.task(prev_task_id).test_loader), None)
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
            for batch in trainer.task(prev_task_id).test_loader:
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


    # Objectives

    def objective(
        self,
        trainer,
        task_id: int,
        support_cols: Sequence[int],
        bundle: BoundaryBundle,
        *,
        refresh_certificates: bool,
        cache_audit: bool = True,
    ) -> Dict[str, float]:
        """Boundary objective from Eq. (1)."""
        return self.objectives(
            trainer,
            task_id,
            [support_cols],
            bundle,
            refresh_certificates=refresh_certificates,
            cache_audit=cache_audit,
        )[0]

    def objectives(
        self,
        trainer,
        task_id: int,
        supports: Sequence[Sequence[int]],
        bundle: BoundaryBundle,
        *,
        refresh_certificates: bool,
        cache_audit: bool = True,
    ) -> List[Dict[str, float]]:
        """Batched Eq. (1) boundary objective for support candidates."""
        support_list = [tuple(sorted(support)) for support in supports]
        count = len(support_list)
        task = trainer.task(task_id)
        current_first = [0.0 for _ in range(count)]
        current_remaining = [0.0 for _ in range(count)]

        for idx, batch in enumerate(bundle.current_eval):
            losses = list(
                trainer.evaluate_batch_losses(
                    task,
                    batch,
                    support_list,
                    refresh_certificates=refresh_certificates and idx == 0,
                )
            )
            target = current_first if idx == 0 else current_remaining
            for row_idx, loss in enumerate(losses):
                target[row_idx] += float(loss)

        old_terms = self._old_audit_terms(trainer, bundle, cache_audit=cache_audit)
        old_worst_loss = old_terms["old_worst_loss"]
        old_mix_loss = old_terms["old_mix_loss"]

        rows = []
        for idx, support_cols in enumerate(support_list):
            switch_penalty = self._switch_penalty(trainer, task_id, support_cols)
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

    def _old_audit_terms(
        self,
        trainer,
        bundle: BoundaryBundle,
        *,
        cache_audit: bool = True,
    ) -> Dict[str, float]:
        """Compute old-task audit losses under their frozen saved supports."""
        
        cache_key = (
            # Instance IDs avoid collisions when freed clone addresses are reused.
            getattr(trainer, "instance_id", id(trainer)),
            id(bundle),
            int(getattr(trainer.persistent_state, "params_revision", 0)),
            int(getattr(trainer.persistent_state, "certificates_revision", 0)),
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
        if cache_audit:
            self._old_audit_cache[cache_key] = dict(terms)
        return terms

    def _switch_penalty(self, trainer, task_id: int, support_cols: Sequence[int]) -> float:
        if task_id == 0:
            return 0.0
        prev = trainer.persistent_state.task_support_snapshots.get(task_id - 1)
        prev_set = set(prev.nonshared) if prev is not None else set(trainer.current_nonshared)
        return self.cfg.exact_search.switch_penalty * float(
            len(set(support_cols).symmetric_difference(prev_set))
        )


    # Static search rows

    def static_rows(
        self,
        trainer,
        task_id: int,
        support_candidates: Sequence[Sequence[int]],
        bundle: BoundaryBundle,
        *,
        reserve_recruitment_candidate: bool = False,
    ) -> List[SupportSearchRow]:
        """Score exact support candidates in JAX-friendly batches."""
        rows: List[SupportSearchRow] = []
        batch_size = max(1, int(self.cfg.exact_search.static_support_batch_size))
        for chunk, real_count in candidate_chunks(
            tuple(support_candidates),
            batch_size,
            pad=True,
        ):
            objectives = self.objectives(
                trainer,
                task_id,
                chunk,
                bundle,
                refresh_certificates=False,
            )
            for support_cols, objective in zip(chunk[:real_count], objectives[:real_count]):
                rows.append(
                    self._support_row_from_objective(
                        trainer,
                        task_id,
                        support_cols,
                        objective,
                        reserve_recruitment_candidate=reserve_recruitment_candidate,
                    )
                )
        return rows

    def _support_row_from_objective(
        self,
        trainer,
        task_id: int,
        support_cols: Sequence[int],
        objective: Dict[str, float],
        *,
        reserve_recruitment_candidate: bool = False,
    ) -> SupportSearchRow:
        cert_score = self._certificate_reuse_score(trainer, support_cols)
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

    def _certificate_reuse_score(self, trainer, support_cols: Sequence[int]) -> float:
        certificates = getattr(trainer.persistent_state, "certificates", {})
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
