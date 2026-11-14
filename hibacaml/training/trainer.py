"""HiBaCaML trainer core: state, cloning, write policies, optimizer, compiled programs and the per-batch update."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import time
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit
from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.replay_bank import SelectorBank
from hibacaml.control.search import ExactSearchService
from hibacaml.control.certificates import CertificateController, shell_slices
from hibacaml.control.shells import ShellController, precision_weight_gradients
from hibacaml.control.support import (
    build_full_support,
    default_nonshared_support,
    support_mask_from_nonshared,
)
from hibacaml.reporting.logger import log_progress
from hibacaml.graph import initialize_hibacaml_state
from hibacaml.training.shared import loss_dict_from_vector
from hibacaml.reporting import (
    append_event,
    start_run,
)
from hibacaml.training import write_policy
from hibacaml.training.trainer_evaluation import TrainerEvaluationMixin
from hibacaml.training.trainer_loop import TrainerLoopMixin
from hibacaml.types import (
    Task,
    PersistentHiBaCaMLState,
    clone_persistent_state,
    SupportSnapshot,
)

_TRAINER_INSTANCE_COUNTER = itertools.count()


def node_column_index(structure) -> Dict[str, int]:
    """Column index of every column-owned node; nodes outside columns are absent."""
    return {
        name: int(name.split("/")[0][3:])
        for name in structure.nodes
        if name.startswith("col")
    }


def _mask_gradients(grads, support_mask, column_of_node):
    masked_nodes = {}
    for node_name, node_grads in grads.nodes.items():
        column_index = column_of_node.get(node_name)
        if column_index is None:
            masked_nodes[node_name] = node_grads
            continue
        active = support_mask[column_index] > 0.0
        masked_nodes[node_name] = jax.tree_util.tree_map(
            lambda grad: jnp.where(active, grad, jnp.zeros_like(grad)),
            node_grads,
        )
    return grads._replace(nodes=masked_nodes)


class GradientClippingState(NamedTuple):
    """Latest device telemetry alongside the unmodified clipping transform state."""

    inner_state: object
    gradient_norm: jax.Array
    clip_scale: jax.Array


def _instrument_gradient_clipping(threshold) -> optax.GradientTransformation:
    # Optax does the clipping; the telemetry never modifies updates.
    clipping = optax.identity() if threshold is None else optax.clip_by_global_norm(threshold)

    def init(params):
        return GradientClippingState(clipping.init(params), jnp.float32(0.0), jnp.float32(1.0))

    def update(updates, state, params=None):
        norm = optax.tree.norm(updates)
        clipped, inner = clipping.update(updates, state.inner_state, params)
        scale = (
            jnp.ones_like(norm)
            if threshold is None
            else jnp.where(norm > 0, jnp.minimum(1.0, threshold / jnp.where(norm > 0, norm, 1.0)), 1.0)
        )
        return clipped, GradientClippingState(inner, norm, scale)

    return optax.GradientTransformation(init, update)


def default_optimizer(cfg: HiBaCaMLConfig) -> optax.GradientTransformation:
    """AdamW, optionally behind a global-norm gradient clip (instrumented when logged)."""
    adamw = optax.adamw(cfg.optimizer_lr, weight_decay=cfg.weight_decay)
    if cfg.reporting.log_gradient_clipping:
        return optax.chain(_instrument_gradient_clipping(cfg.grad_clip_norm), adamw)
    if cfg.grad_clip_norm is None:
        return adamw
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), adamw)


def clipping_telemetry(opt_state) -> Optional[GradientClippingState]:
    """The instrumented clip state of a chained optimizer, or None."""
    first = opt_state[0] if isinstance(opt_state, tuple) and opt_state else None
    return first if isinstance(first, GradientClippingState) else None


def adam_state(opt_state) -> optax.ScaleByAdamState:
    """The Adam count/moments of bare or clip-chained AdamW."""
    return opt_state[0] if isinstance(opt_state[0], optax.ScaleByAdamState) else opt_state[1][0]


def adam_count(opt_state) -> int:
    """The single global Adam step count, for bare or clip-chained AdamW."""
    return int(adam_state(opt_state).count)


def with_moments(opt_state, mu, nu):
    """`opt_state` with new Adam moments; count, clipping state and layout unchanged."""
    if isinstance(opt_state[0], optax.ScaleByAdamState):
        return (opt_state[0]._replace(mu=mu, nu=nu), *opt_state[1:])
    adamw = opt_state[1]
    return (opt_state[0], (adamw[0]._replace(mu=mu, nu=nu), *adamw[1:]), *opt_state[2:])


def build_update_programs(
    gradient_step,
    structure,
    optimizer,
    cfg,
    *,
    allow_lean,
):
    column_of_node = node_column_index(structure)
    support_mask_node = structure.config["hibacaml"]["support_mask_node"]
    shell_names = tuple(shell_slices(cfg))
    resistance_enabled = cfg.exact_search.enable_precision_update_resistance
    resistance_strength = cfg.exact_search.precision_update_strength
    resistance_floor = cfg.exact_search.precision_update_floor
    layout = write_policy.PolicyLayout.from_structure(structure, cfg)

    def update_core(params, opt_state, inputs, rng_key):
        grads, loss_vector, final_state = gradient_step(params, inputs, rng_key)
        support_mask = inputs["clamps"][support_mask_node][0]
        grads = _mask_gradients(grads, support_mask, column_of_node)
        grads = precision_weight_gradients(
            grads,
            params,
            shell_names=shell_names,
            enabled=resistance_enabled,
            strength=resistance_strength,
            floor=resistance_floor,
        )
        # A write policy is an optional input: without one the trace is unchanged.
        policy = "write_mask" in inputs
        if policy:
            grads = write_policy.mask(grads, inputs, support_mask, layout)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        # AdamW momentum and weight decay still move inactive columns after
        # gradient masking, so the updates are masked too.
        updates = _mask_gradients(updates, support_mask, column_of_node)
        if policy:
            updates = write_policy.mask(updates, inputs, support_mask, layout)
            updates = write_policy.scale(updates, inputs, layout)
        new_params = optax.apply_updates(params, updates)
        if policy:
            new_params = write_policy.restore(new_params, params, inputs, support_mask, layout)
        return new_params, new_opt_state, loss_vector, final_state

    programs = {"update": jax.jit(update_core)}
    if allow_lean:
        def update_lean(params, opt_state, inputs, rng_key):
            return update_core(params, opt_state, inputs, rng_key)[:3]

        programs["update_lean"] = jax.jit(update_lean)
    return programs


def build_evaluation_programs(inference_step, evaluation_step):
    def score_core(params, clamps, targets, rng_key):
        result = evaluation_step(params, clamps, targets, rng_key)
        # Retaining the device-only loss vector preserves detailed-path arithmetic.
        return result[1], result[2]

    compiled_score = jax.jit(score_core)

    def evaluation_score(params, clamps, targets, rng_key):
        per_sample_total, _ = compiled_score(params, clamps, targets, rng_key)
        return per_sample_total

    return {
        "inference": jax.jit(inference_step),
        "evaluation": jax.jit(evaluation_step),
        "evaluation_score": evaluation_score,
    }


class HiBaCaMLTrainer(TrainerLoopMixin, TrainerEvaluationMixin, ABC):
    """Shared sequential trainer for the PC and backprop learners."""

    COMPOSER_BEFORE_PARENT: bool
    CLAMPS_MAY_INCLUDE_TARGETS: bool

    def __init__(
        self,
        cfg: HiBaCaMLConfig,
        structure,
        params,
        tasks: Optional[Sequence[Task]] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
        rng_key: Optional[jax.Array] = None,
        persistent_state: Optional[PersistentHiBaCaMLState] = None,
        emit_run_artifacts: bool = True,
        selector_bank: Optional[SelectorBank] = None,
        run_id: str = "",
        opt_state=None,
        experiment_metadata: Optional[Dict[str, object]] = None,
        programs: Optional[Dict[str, Callable]] = None,
    ):
        self.cfg = cfg
        self.structure = structure
        # Unique identifier for this trainer instance.
        self.instance_id = next(_TRAINER_INSTANCE_COUNTER)
        graph_state_initializer = structure.config.get("graph_state_initializer")
        if not isinstance(graph_state_initializer, FeedforwardStateInit):
            raise ValueError(
                "HiBaCaML training requires FeedforwardStateInit; got "
                f"{type(graph_state_initializer).__name__}"
            )
        self.params = params
        self.experiment_metadata = dict(experiment_metadata or {})
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(cfg.seed)
        self.optimizer = optimizer or default_optimizer(cfg)
        # Rollout clones supply the parent's state; initializing here would allocate a full zeros tree (adamw mu + nu) only to have it overwritten.
        self.opt_state = opt_state if opt_state is not None else self.optimizer.init(params)
        self.persistent_state = persistent_state or initialize_hibacaml_state(params, cfg)
        self.persistent_state.params = self.params
        self.persistent_state.opt_state = self.opt_state
        self.tasks: Dict[int, Task] = {}
        self._task_queries: Dict[int, jnp.ndarray] = {}
        # Support-invariant tiled images reused across candidate chunks; a bundle's two eval batches are the whole working set.
        self._tiled_images: Dict[Tuple[int, int], Tuple[jnp.ndarray, jnp.ndarray]] = {}
        self.current_phi = cfg.phi
        self.current_nonshared = default_nonshared_support(cfg)
        self.write_policy: Optional[write_policy.WritePolicy] = None
        self._policy_inputs: Optional[Dict[str, jax.Array]] = None
        # Reporting only: per-epoch clipping aggregates and the last task's completion reports.
        self.clipping_epoch_summaries: List[Dict[str, object]] = []
        self._clipping_epoch: Optional[Dict[str, object]] = None
        self.completion_reports: Dict[int, Dict[str, object]] = {}
        self._programs = (
            programs
            if programs is not None
            else type(self).build_programs(structure, self.optimizer, cfg)
        )
        self.certificate_controller = CertificateController(cfg, structure)
        self.shell_controller = ShellController(cfg, self.certificate_controller)
        # Rollout clones get no run root, which is what keeps their events out of the audit stream.
        self.run_root: Optional[Path] = (
            Path(cfg.reporting.experiment_root) if emit_run_artifacts else None
        )
        self._task_summaries: Dict[str, Dict[str, object]] = {}
        if self.run_root is not None:
            start_run(self.run_root, mode=cfg.mode, global_step=0, phase="created")
            append_event(self.run_root, "trainer_created", mode=cfg.mode)
        if tasks is not None:
            self.register_tasks(tasks)
        self.run_id = str(run_id)
        if selector_bank is None:
            selector_bank = SelectorBank(
                cfg.selector_state_path(),
                bank_filename=cfg.reporting.selector_bank_filename,
                metadata_filename=cfg.reporting.selector_bank_metadata_filename,
            )
            selector_bank.load()
        self.selector_bank = selector_bank

    @cached_property
    def exact_search(self) -> ExactSearchService:
        """Built on first use: the service holds the trainer, so an eager
        instance would put every rollout clone in a reference cycle."""
        return ExactSearchService(
            self.cfg,
            self,
            selector_bank=self.selector_bank,
            run_id=self.run_id,
        )

    @classmethod
    @abstractmethod
    def build_programs(cls, structure, optimizer, cfg) -> Dict[str, Callable]:
        """Return compiled programs reused by this learner."""

    def register_tasks(self, tasks: Sequence[Task]) -> None:
        for task in tasks:
            self.tasks[task.task_id] = task
            self._task_queries[task.task_id] = jnp.asarray(
                task.task_query, dtype=jnp.float32
            )

    def clone(self) -> "HiBaCaMLTrainer":
        # JAX arrays are immutable — any update (apply_updates, structural edits)
        # produces new arrays, never mutating existing ones. So sharing the array
        # leaves via tree_map is safe and avoids copying the full weight tensors
        # for every rollout candidate.

        ps_clone = clone_persistent_state(self.persistent_state)
        cloned = type(self)(
            cfg=self.cfg,                                               # frozen dataclass
            structure=self.structure,
            params=jax.tree_util.tree_map(lambda x: x, self.params),  # share immutable leaves
            tasks=list(self.tasks.values()),
            optimizer=self.optimizer,
            rng_key=self.rng_key,
            persistent_state=ps_clone,
            emit_run_artifacts=False,
            selector_bank=self.selector_bank,  # rollout clones share the bank for proposal scoring
            run_id=self.run_id,
            opt_state=jax.tree_util.tree_map(lambda x: x, self.opt_state),  # share immutable leaves
            experiment_metadata=self.experiment_metadata,
            programs=self._programs,  # compiled once, shared, never rebuilt
        )
        cloned.persistent_state.params = cloned.params
        cloned.persistent_state.opt_state = cloned.opt_state
        cloned.current_phi = self.current_phi                          # frozen dataclass
        cloned.current_nonshared = tuple(self.current_nonshared)
        cloned.write_policy = self.write_policy                        # immutable, with read-only table
        # Own dict, shared immutable arrays: replacing a clone's entry never reaches the parent.
        cloned._policy_inputs = None if self._policy_inputs is None else dict(self._policy_inputs)
        return cloned

    def isolated_clone(self) -> "HiBaCaMLTrainer":
        """Clone with its own loader cursors, so trials never advance the parent's passes."""
        cloned = self.clone()
        cloned.tasks = {
            task_id: dataclasses.replace(
                task,
                train_loader=copy.copy(task.train_loader),
                validation_loader=copy.copy(task.validation_loader),
                test_loader=copy.copy(task.test_loader),
            )
            for task_id, task in self.tasks.items()
        }
        return cloned

    def resume_clone(self) -> "HiBaCaMLTrainer":
        """Clone that continues this trainer's trajectory."""
        resumed = self.clone()
        resumed._last_graph_state = getattr(self, "_last_graph_state", None)
        return resumed

    def set_write_policy(self, policy: Optional[write_policy.WritePolicy]) -> None:
        """Install (or clear, with None) the write protection used by every later update."""
        if policy is not None:
            if type(self).CLAMPS_MAY_INCLUDE_TARGETS:
                raise ValueError("write policies are supported by the backprop learner only")
            search = self.cfg.exact_search
            if search.enable_exact_search or search.enable_structural_edits:
                raise ValueError("write policies require exact support search and structural edits off")
            write_policy.validate(policy, self.cfg)
            table = np.array(policy.unit_scales, dtype=np.float32)
            table.flags.writeable = False
            policy = dataclasses.replace(policy, unit_scales=table)
        self.write_policy = policy
        self._policy_inputs = None if policy is None else write_policy.policy_inputs(policy, self.cfg)

    def _record_timing(
        self,
        task_id: int,
        key: str,
        seconds: float,
        *,
        accumulate: bool = False,
    ) -> None:
        summary = self.persistent_state.timing_summaries.setdefault(task_id, {})
        if accumulate:
            summary[key] = summary.get(key, 0.0) + float(seconds)
        else:
            summary[key] = float(seconds)

    def _bump_params_revision(self) -> None:
        self.persistent_state.params_revision += 1

    def set_boundary_choice(
        self,
        task_id: int,
        nonshared: Sequence[int],
        phi,
        *,
        record_dropped_columns: bool = True,
    ) -> None:
        boundary = tuple(sorted(nonshared))
        self.current_phi = phi
        self.persistent_state.boundary_support[task_id] = boundary
        # track which columns were dropped vs the previous task's snapshot
        # (exact-search demotion history; fixed-allocation protocols skip it).
        if record_dropped_columns and task_id > 0:
            prev_snapshot = self.persistent_state.task_support_snapshots.get(task_id - 1)
            if prev_snapshot is not None:
                dropped = tuple(
                    c for c in prev_snapshot.nonshared if c not in set(boundary)
                )
                self.persistent_state.recently_demoted[task_id] = dropped
                # Prune entries older than the configured history window.
                window = max(0, int(self.cfg.exact_search.replay_history_window))
                cutoff = task_id - window
                stale = [t for t in self.persistent_state.recently_demoted if t < cutoff]
                for t in stale:
                    self.persistent_state.recently_demoted.pop(t, None)
        log_progress(
            f"task={task_id} boundary choice set nonshared={boundary} "
            f"phi=({phi.outer_quantile:.3f},{phi.middle_quantile:.3f},"
            f"{phi.replacement_margin_base:.3f},{phi.demotion_min_role_gain:.3f})",
            component="trainer",
        )

    def set_current_support(self, task_id: int, nonshared: Sequence[int], phi) -> None:
        self.current_nonshared = tuple(sorted(nonshared))
        self.current_phi = phi
        self.persistent_state.current_support[task_id] = self.current_nonshared
        log_progress(
            f"task={task_id} current support set nonshared={self.current_nonshared} "
            f"phi=({phi.outer_quantile:.3f},{phi.middle_quantile:.3f},"
            f"{phi.replacement_margin_base:.3f},{phi.demotion_min_role_gain:.3f})",
            component="trainer",
        )

    def freeze_task_support(self, task_id: int) -> SupportSnapshot:
        snapshot = SupportSnapshot(
            task_id=task_id,
            nonshared=tuple(self.current_nonshared),
            full_support=build_full_support(self.cfg, self.current_nonshared),
            phi=self.current_phi,
            global_step=self.persistent_state.global_step,
        )
        self.persistent_state.task_support_snapshots[task_id] = snapshot
        return snapshot

    def build_support_mask(self, nonshared: Sequence[int]) -> jnp.ndarray:
        return support_mask_from_nonshared(self.cfg, tuple(nonshared))

    def active_full_support(self, nonshared: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
        chosen = tuple(nonshared) if nonshared is not None else self.current_nonshared
        return build_full_support(self.cfg, chosen)

    def task(self, task_id: int) -> Task:
        return self.tasks[task_id]

    @abstractmethod
    def _training_inputs(
        self,
        batch: Dict[str, jnp.ndarray],
        task: Task,
        nonshared: Sequence[int],
    ) -> Dict[str, object]:
        """Build the host-side inputs consumed by this learner's update."""

    @abstractmethod
    def _gradients(self, params, inputs, rng_key):
        """Return ``(grads, loss_vector, final_state)`` for one batch."""

    def training_update(
        self,
        batch: Dict[str, jnp.ndarray],
        task: Task,
        nonshared: Sequence[int],
    ):
        """Run one state-producing learner update."""
        return self._execute_training_update(
            batch,
            task,
            nonshared,
            need_state=True,
        )

    def _execute_training_update(
        self,
        batch: Dict[str, jnp.ndarray],
        task: Task,
        nonshared: Sequence[int],
        *,
        need_state: bool,
    ):
        inputs = self._training_inputs(batch, task, nonshared)
        self.rng_key, state_key = jax.random.split(self.rng_key)
        use_lean = not need_state and "update_lean" in self._programs
        if use_lean:
            self.params, self.opt_state, loss_vector = self._programs[
                "update_lean"
            ](self.params, self.opt_state, inputs, state_key)
            final_state = None
        else:
            self.params, self.opt_state, loss_vector, final_state = self._programs[
                "update"
            ](self.params, self.opt_state, inputs, state_key)
        self.persistent_state.params = self.params
        self.persistent_state.opt_state = self.opt_state
        self._bump_params_revision()
        if final_state is not None:
            self._last_graph_state = final_state
        return loss_dict_from_vector(loss_vector), final_state

    def _train_batch(
        self,
        task: Task,
        batch: Dict[str, jnp.ndarray],
        nonshared: Sequence[int],
        *,
        epoch_index: int,
        batch_index: int,
        train_batches: int,
        batch_log_every: int,
    ):
        next_step = self.persistent_state.global_step + 1
        cert_due = next_step % self.cfg.cert_refresh_interval == 0
        maintenance_due = (
            self.cfg.exact_search.enable_exact_search
            and task.task_id > 0
            and next_step
            - self.persistent_state.last_maintenance_step.get(task.task_id, 0)
            >= self.cfg.exact_search.maintenance_interval
        )
        demotion_due = (
            self.cfg.exact_search.enable_exact_search
            and self.cfg.exact_search.enable_demotion_swap_audit
            and task.task_id > 0
            and next_step
            - self.persistent_state.last_demotion_audit_step.get(task.task_id, 0)
            >= self.cfg.exact_search.demotion_audit_interval
        )
        report_due = (
            batch_index == 0
            or (batch_index + 1) % batch_log_every == 0
            or batch_index + 1 == train_batches
        )
        need_state = (
            "update_lean" not in self._programs
            or self.cfg.exact_search.enable_structural_edits
            or cert_due
            or maintenance_due
            or demotion_due
            or report_due
        )
        if (
            batch_index == 0
            or cert_due
            or maintenance_due
            or demotion_due
            or batch_index + 1 == train_batches
        ):
            log_progress(
                f"task={task.task_id} "
                f"epoch={epoch_index + 1}/{self.cfg.epochs_per_task} "
                f"batch={batch_index + 1}/{train_batches} start step={next_step} "
                f"hooks(cert_refresh={cert_due}, maintenance={maintenance_due}, "
                f"demotion={demotion_due})",
                component="trainer",
            )

        batch_started = time.perf_counter()
        losses, final_state = self._execute_training_update(
            batch,
            task,
            nonshared,
            need_state=need_state,
        )
        if self.cfg.exact_search.enable_structural_edits:
            updated_params = self.shell_controller.apply_structural_edits(
                self.params,
                final_state,
                self.persistent_state,
                self.active_full_support(nonshared),
                self.current_phi,
            )
            if updated_params is not self.params:
                self.params = updated_params
                self.persistent_state.params = self.params
                self._bump_params_revision()
        self.persistent_state.global_step += 1
        telemetry = clipping_telemetry(self.opt_state)
        if telemetry is not None:
            self._accumulate_clipping(telemetry, task.task_id, epoch_index)
        final_losses = dict(losses)
        batch_size = int(batch["x"].shape[0])

        if demotion_due:
            self.exact_search.demotion_swap_audit(
                task.task_id,
                final_state,
                nonshared,
            )
            self.persistent_state.last_demotion_audit_step[
                task.task_id
            ] = self.persistent_state.global_step

        if report_due:
            composer_diag = self.composer_diagnostics_from_state(
                final_state,
                task,
                nonshared,
            )
            self.persistent_state.composer_diagnostics[
                self.persistent_state.global_step
            ] = {
                "task_id": task.task_id,
                "epoch": epoch_index + 1,
                "batch": batch_index + 1,
                **composer_diag,
            }
            log_progress(
                f"task={task.task_id} "
                f"epoch={epoch_index + 1}/{self.cfg.epochs_per_task} "
                f"batch={batch_index + 1}/{train_batches} "
                f"loss_total={final_losses['total']:.4f} "
                f"loss_task={final_losses['task']:.4f} "
                f"step={self.persistent_state.global_step} "
                f"batch_s={time.perf_counter() - batch_started:.2f}",
                component="trainer",
            )
            if self.run_root is not None:
                append_event(
                    self.run_root,
                    "training_step",
                    task_id=task.task_id,
                    epoch=epoch_index + 1,
                    batch=batch_index + 1,
                    train_batches=train_batches,
                    global_step=self.persistent_state.global_step,
                    losses=final_losses,
                    composer=composer_diag,
                )

        if cert_due:
            self._refresh_certificates(
                final_state.batch_size,
                graph_state=final_state,
            )

        if maintenance_due:
            self.exact_search.local_one_swap(task.task_id)
            nonshared = self.persistent_state.current_support[task.task_id]
            self.persistent_state.last_maintenance_step[
                task.task_id
            ] = self.persistent_state.global_step

        return nonshared, final_losses, batch_size

    def _accumulate_clipping(self, telemetry, task_id: int, epoch_index: int) -> None:
        """Fold this step's clipping telemetry into the open epoch aggregate (one host read)."""
        norm, scale = map(float, jax.device_get((telemetry.gradient_norm, telemetry.clip_scale)))
        epoch = self._clipping_epoch
        if epoch is None:
            epoch = self._clipping_epoch = dict(
                task_id=task_id, epoch=epoch_index + 1, count=0, clipped=0,
                norm_sum=0.0, max_gradient_norm=0.0, scale_sum=0.0, min_clip_scale=1.0,
            )
        epoch["count"] += 1
        epoch["clipped"] += int(scale < 1.0)
        epoch["norm_sum"] += norm
        epoch["max_gradient_norm"] = max(epoch["max_gradient_norm"], norm)
        epoch["scale_sum"] += scale
        epoch["min_clip_scale"] = min(epoch["min_clip_scale"], scale)

    def _close_clipping_epoch(self) -> None:
        epoch, self._clipping_epoch = getattr(self, "_clipping_epoch", None), None
        if epoch is None:
            return
        count = epoch["count"]
        self.clipping_epoch_summaries.append(dict(
            task_id=epoch["task_id"],
            epoch=epoch["epoch"],
            completed_updates=count,
            clipped_updates=epoch["clipped"],
            clipped_fraction=epoch["clipped"] / count,
            mean_gradient_norm=epoch["norm_sum"] / count,
            max_gradient_norm=epoch["max_gradient_norm"],
            mean_clip_scale=epoch["scale_sum"] / count,
            min_clip_scale=epoch["min_clip_scale"],
        ))
