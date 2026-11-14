"""Fixed-batch, epoch and task loop of the HiBaCaML trainer.

`TrainerLoopMixin` defines methods only; they run on a `HiBaCaMLTrainer`.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp

from hibacaml.control.support import default_nonshared_support
from hibacaml.reporting import append_event, write_run_state
from hibacaml.reporting.export import write_json
from hibacaml.reporting.integrity import require
from hibacaml.reporting.logger import log_progress
from hibacaml.types import Task, TaskSummary

if TYPE_CHECKING:
    from hibacaml.training.trainer import HiBaCaMLTrainer


class TrainerLoopMixin:
    """`train_batches`, `_train_epoch` and `train_task`."""

    def train_batches(
        self,
        task: Task,
        batches: Sequence[Dict[str, jnp.ndarray]],
        nonshared: Sequence[int],
        *,
        epoch_index: int = 0,
        check_finite: bool = False,
    ) -> Tuple[int, ...]:
        """Train fixed batches through the ordinary per-batch lifecycle; no loaders or validation.

        `check_finite` rejects a nonfinite loss component right after its update
        (the losses are already host floats, so the check adds no transfer).
        """
        count = len(batches)
        for batch_index, batch in enumerate(batches):
            nonshared, losses, _ = self._train_batch(
                task,
                batch,
                nonshared,
                epoch_index=epoch_index,
                batch_index=batch_index,
                train_batches=count,
                batch_log_every=max(1, count // 4),
            )
            if check_finite:
                require(all(math.isfinite(v) for v in losses.values()), "nonfinite training loss",
                        global_step=self.persistent_state.global_step, losses=losses)
        self._close_clipping_epoch()
        return tuple(nonshared)

    def _train_epoch(
        self,
        task: Task,
        nonshared: Sequence[int],
        *,
        epoch_index: int,
        train_batches: int,
        batch_log_every: int,
        evaluate_train_each_epoch: bool,
        on_epoch_end: Optional[Callable[["HiBaCaMLTrainer", Task, int], None]] = None,
    ):
        epoch_started = time.perf_counter()
        epoch_start_step = self.persistent_state.global_step
        epoch_examples = 0
        final_losses = {
            "task": 0.0,
            "hier_mid": 0.0,
            "hier_global": 0.0,
            "parent_child": 0.0,
            "composer": 0.0,
            "total": 0.0,
        }
        epoch_loss_sums = {name: 0.0 for name in final_losses}
        for batch_index, batch in enumerate(task.train_loader):
            nonshared, final_losses, batch_size = self._train_batch(
                task,
                batch,
                nonshared,
                epoch_index=epoch_index,
                batch_index=batch_index,
                train_batches=train_batches,
                batch_log_every=batch_log_every,
            )
            epoch_examples += batch_size
            for name, value in final_losses.items():
                epoch_loss_sums[name] += float(value) * batch_size

        training_seconds = time.perf_counter() - epoch_started
        self._close_clipping_epoch()
        self._record_timing(
            task.task_id,
            f"epoch_{epoch_index + 1}_seconds",
            training_seconds,
        )
        # Runs outside the RNG guard and before validation, which sees its effects.
        if on_epoch_end is not None:
            on_epoch_end(self, task, epoch_index + 1)

        validation_loader = getattr(task, "validation_loader", None)
        if evaluate_train_each_epoch or validation_loader is not None:
            evaluation_started = time.perf_counter()
            train_metrics: Dict[str, object] = {}
            validation_metrics: Dict[str, object] = {}
            saved_training_rng = self.rng_key
            try:
                if evaluate_train_each_epoch:
                    train_metrics = self.evaluate_loader(
                        task,
                        task.train_loader,
                        nonshared,
                        split_name="train",
                        mode=f"epoch_{epoch_index + 1}",
                        refresh_certificates=False,
                    )
                if validation_loader is not None:
                    validation_metrics = self.evaluate_loader(
                        task,
                        validation_loader,
                        nonshared,
                        split_name="validation",
                        mode=f"epoch_{epoch_index + 1}",
                        refresh_certificates=False,
                    )
            finally:
                # Epoch evaluation must not perturb later training randomness.
                self.rng_key = saved_training_rng

            evaluation_seconds = time.perf_counter() - evaluation_started
            self._record_timing(
                task.task_id,
                f"epoch_{epoch_index + 1}_evaluation_seconds",
                evaluation_seconds,
            )
            loss_denominator = max(epoch_examples, 1)
            epoch_record = {
                "task_id": task.task_id,
                "epoch": epoch_index + 1,
                "global_step": self.persistent_state.global_step,
                "epoch_update_count": (
                    self.persistent_state.global_step - epoch_start_step
                ),
                "epoch_examples": epoch_examples,
                "optimization_loss_components": {
                    name: value / loss_denominator
                    for name, value in epoch_loss_sums.items()
                },
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "active_support": self.active_full_support(nonshared),
                "heldout_refresh_certificates": False,
            }
            self.persistent_state.epoch_evaluations.append(epoch_record)
            if self.run_root is not None:
                append_event(self.run_root, "epoch_validation", **epoch_record)

        return nonshared, final_losses

    def train_task(
        self,
        task: Task,
        *,
        evaluate_train_each_epoch: bool = False,
        on_epoch_end: Optional[Callable[["HiBaCaMLTrainer", Task, int], None]] = None,
        collect_examples: bool = False,
    ) -> TaskSummary:
        if task.task_id not in self.persistent_state.current_support:
            if self.cfg.exact_search.enable_exact_search:
                support, phi = self.exact_search.boundary_search(task.task_id)
                self.set_current_support(task.task_id, support, phi)
            else:
                default_support = default_nonshared_support(self.cfg)
                self.set_boundary_choice(
                    task.task_id,
                    default_support,
                    self.current_phi,
                )
                self.set_current_support(
                    task.task_id,
                    default_support,
                    self.current_phi,
                )

        nonshared = self.persistent_state.current_support[task.task_id]
        final_losses = {
            "task": 0.0,
            "hier_mid": 0.0,
            "hier_global": 0.0,
            "parent_child": 0.0,
            "composer": 0.0,
            "total": 0.0,
        }
        train_batches = len(task.train_loader)
        batch_log_every = max(1, train_batches // 4) if train_batches else 1
        log_progress(
            f"task={task.task_id} train start epochs={self.cfg.epochs_per_task} "
            f"train_batches={train_batches} support={tuple(nonshared)} "
            f"cert_refresh_interval={self.cfg.cert_refresh_interval} "
            f"maintenance_interval={self.cfg.exact_search.maintenance_interval}",
            component="trainer",
        )
        if self.run_root is not None:
            append_event(
                self.run_root,
                "task_train_start",
                task_id=task.task_id,
                epochs=self.cfg.epochs_per_task,
                train_batches=train_batches,
                support=tuple(nonshared),
                global_step=self.persistent_state.global_step,
            )
            write_run_state(
                self.run_root,
                phase="task_train",
                task_id=task.task_id,
                global_step=self.persistent_state.global_step,
            )

        for epoch_index in range(self.cfg.epochs_per_task):
            nonshared, final_losses = self._train_epoch(
                task,
                nonshared,
                epoch_index=epoch_index,
                train_batches=train_batches,
                batch_log_every=batch_log_every,
                evaluate_train_each_epoch=evaluate_train_each_epoch,
                on_epoch_end=on_epoch_end,
            )

        self.freeze_task_support(task.task_id)
        if self.cfg.reporting.write_selector_state:
            self.selector_bank.save()
            if self.run_root is not None:
                append_event(
                    self.run_root,
                    "selector_bank_saved",
                    task_id=task.task_id,
                    row_count=len(self.selector_bank),
                    path=str(self.selector_bank.bank_path),
                )
        # Reporting observes the run, so it never refreshes certificates. This pass
        # advances the RNG (one split per batch); with `collect_examples` runners keep
        # its reports and example arrays instead of evaluating again.
        reports: Dict[int, Dict[str, object]] = {}
        extra = {"collect_examples": True, "reports": reports} if collect_examples else {}
        metrics = self.evaluate_task(task.task_id, refresh_certificates=False, **extra)
        self.completion_reports = reports
        summary = TaskSummary(
            task_id=task.task_id,
            classes=task.classes,
            support_indices=self.active_full_support(nonshared),
            accuracy=metrics["accuracy"],
            mean_loss=metrics["mean_loss"],
            best_old_accuracy=metrics["best_old_accuracy"],
            support_entropy=metrics["support_entropy"],
        )
        if self.run_root is not None:
            self._task_summaries[str(task.task_id)] = summary.__dict__
            write_json(self.run_root / "task_summaries.json", self._task_summaries)
            append_event(
                self.run_root,
                "task_train_done",
                task_id=task.task_id,
                global_step=self.persistent_state.global_step,
                summary=summary.__dict__,
            )
        return summary
