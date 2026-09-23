"""Shared HiBaCaML training and evaluation orchestration."""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
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
from hibacaml.nodes.composer import composer_details
from hibacaml.training.shared import (
    EvaluationAccumulator,
    batch_query,
    batch_targets,
    build_multi_support_clamps,
    build_single_support_clamps,
    loss_dict_from_vector,
    support_mean_losses,
    tile_batch,
)
from hibacaml.reporting import (
    append_event,
    start_run,
    write_run_state,
)
from hibacaml.reporting.export import write_json
from hibacaml.types import (
    MnistTask,
    PersistentHiBaCaMLState,
    clone_persistent_state,
    SupportSnapshot,
    TaskSummary,
)

_TRAINER_INSTANCE_COUNTER = itertools.count()


def _mask_gradients(grads, support_mask, node_column_index):
    masked_nodes = {}
    for node_name, node_grads in grads.nodes.items():
        column_index = node_column_index.get(node_name)
        if column_index is None:
            masked_nodes[node_name] = node_grads
            continue
        active = support_mask[column_index] > 0.0
        masked_nodes[node_name] = jax.tree_util.tree_map(
            lambda grad: jnp.where(active, grad, jnp.zeros_like(grad)),
            node_grads,
        )
    return grads._replace(nodes=masked_nodes)


def build_update_programs(
    gradient_step,
    structure,
    optimizer,
    cfg,
    *,
    allow_lean,
):
    node_column_index = {
        name: int(name.split("/")[0][3:])
        for name in structure.nodes
        if name.startswith("col")
    }
    support_mask_node = structure.config["hibacaml"]["support_mask_node"]
    shell_names = tuple(shell_slices(cfg))
    resistance_enabled = cfg.exact_search.enable_precision_update_resistance
    resistance_strength = cfg.exact_search.precision_update_strength
    resistance_floor = cfg.exact_search.precision_update_floor

    def update_core(params, opt_state, inputs, rng_key):
        grads, loss_vector, final_state = gradient_step(params, inputs, rng_key)
        grads = _mask_gradients(
            grads,
            inputs["clamps"][support_mask_node][0],
            node_column_index,
        )
        grads = precision_weight_gradients(
            grads,
            params,
            shell_names=shell_names,
            enabled=resistance_enabled,
            strength=resistance_strength,
            floor=resistance_floor,
        )
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
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


class HiBaCaMLTrainer(ABC):
    """Shared sequential trainer for the PC and backprop learners."""

    COMPOSER_BEFORE_PARENT: bool
    CLAMPS_MAY_INCLUDE_TARGETS: bool

    def __init__(
        self,
        cfg: HiBaCaMLConfig,
        structure,
        params,
        tasks: Optional[Sequence[MnistTask]] = None,
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
        self.optimizer = optimizer or optax.adamw(cfg.optimizer_lr, weight_decay=cfg.weight_decay)
        # Rollout clones supply the parent's state; initializing here would allocate a full zeros tree (adamw mu + nu) only to have it overwritten.
        self.opt_state = opt_state if opt_state is not None else self.optimizer.init(params)
        self.persistent_state = persistent_state or initialize_hibacaml_state(params, cfg)
        self.persistent_state.params = self.params
        self.persistent_state.opt_state = self.opt_state
        self.tasks: Dict[int, MnistTask] = {}
        self._task_queries: Dict[int, jnp.ndarray] = {}
        # Support-invariant tiled images reused across candidate chunks; a bundle's two eval batches are the whole working set.
        self._tiled_images: Dict[Tuple[int, int], Tuple[jnp.ndarray, jnp.ndarray]] = {}
        self.current_phi = cfg.phi
        self.current_nonshared = default_nonshared_support(cfg)
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

    def register_tasks(self, tasks: Sequence[MnistTask]) -> None:
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
        return cloned

    def resume_clone(self) -> "HiBaCaMLTrainer":
        """Clone that continues this trainer's trajectory."""
        resumed = self.clone()
        resumed._last_graph_state = getattr(self, "_last_graph_state", None)
        return resumed

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

    def set_boundary_choice(self, task_id: int, nonshared: Sequence[int], phi) -> None:
        boundary = tuple(sorted(nonshared))
        self.current_phi = phi
        self.persistent_state.boundary_support[task_id] = boundary
        # track which columns were dropped vs the previous task's snapshot.
        if task_id > 0:
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

    def task(self, task_id: int) -> MnistTask:
        return self.tasks[task_id]

    def _refresh_certificates(
        self,
        batch_size: int,
        *,
        params=None,
        graph_state=None,
    ) -> None:
        params = params if params is not None else self.params
        state = graph_state if graph_state is not None else self._last_graph_state_or_placeholder(batch_size)
        self.certificate_controller.refresh_certificates(params, state, self.persistent_state)

    def _task_query(self, task: MnistTask) -> jnp.ndarray:
        """Device-resident task query, converted once per registered task."""
        query = self._task_queries.get(task.task_id)
        if query is None:
            query = jnp.asarray(task.task_query, dtype=jnp.float32)
            self._task_queries[task.task_id] = query
        return query

    def _tiled_batch_images(self, batch: Dict[str, jnp.ndarray], repeats: int) -> jnp.ndarray:
        """Support-major tiled images, reused across a sweep's candidate chunks."""
        images = batch["x"]
        key = (id(images), repeats)
        cached = self._tiled_images.get(key)
        if cached is not None and cached[0] is images:
            return cached[1]
        tiled = tile_batch(jnp.asarray(images, dtype=jnp.float32), repeats)
        if len(self._tiled_images) >= 2:
            self._tiled_images.clear()
        self._tiled_images[key] = (images, tiled)
        return tiled

    def _build_clamps(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        nonshared: Sequence[int],
        *,
        params=None,
        refresh_certificates: bool = True,
        include_targets: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        if include_targets and not self.CLAMPS_MAY_INCLUDE_TARGETS:
            raise ValueError(
                f"{type(self).__name__} must not clamp supervised targets into "
                "the graph: its targets stay external, and clamping them would "
                "hold the output node fixed instead of predicted."
            )
        batch_size = int(batch["x"].shape[0])
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        support_mask = self.build_support_mask(nonshared)
        cert_vectors = self.certificate_controller.certificate_matrix(self.persistent_state, support_mask)
        return build_single_support_clamps(
            self.structure,
            batch,
            self._task_query(task),
            support_mask,
            cert_vectors,
            include_targets=include_targets,
        )

    def _build_multi_support_clamps(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        supports: Sequence[Sequence[int]],
        *,
        params=None,
        refresh_certificates: bool = True,
        include_targets: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        if include_targets and not self.CLAMPS_MAY_INCLUDE_TARGETS:
            raise ValueError(
                f"{type(self).__name__} must not clamp supervised targets into "
                "the graph: its targets stay external, and clamping them would "
                "hold the output node fixed instead of predicted."
            )
        batch_size = int(batch["x"].shape[0])
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        support_masks = jnp.stack(
            [self.build_support_mask(nonshared) for nonshared in supports]
        )
        # One certificate state masked for every support at once.
        cert_stack = self.certificate_controller.certificate_stack_for_masks(
            self.persistent_state, support_masks
        )
        return build_multi_support_clamps(
            self.structure,
            batch,
            self._task_query(task),
            self._tiled_batch_images(batch, len(supports)),
            support_masks,
            cert_stack,
            include_targets=include_targets,
        )

    def _last_graph_state_or_placeholder(self, batch_size: int):
        if getattr(self, "_last_graph_state", None) is not None and self._last_graph_state.batch_size == batch_size:
            return self._last_graph_state
        from fabricpc.core.types import GraphState, NodeState

        zero_nodes = {}
        for node_name, node in self.structure.nodes.items():
            shape = (batch_size, *node.node_info.shape)
            zero_nodes[node_name] = NodeState(
                z_latent=jnp.zeros(shape, dtype=jnp.float32),
                z_mu=jnp.zeros(shape, dtype=jnp.float32),
                error=jnp.zeros(shape, dtype=jnp.float32),
                energy=jnp.zeros((batch_size,), dtype=jnp.float32),
                latent_grad=jnp.zeros(shape, dtype=jnp.float32),
            )

        return GraphState(nodes=zero_nodes, batch_size=batch_size)

    def _run_evaluation_inference(self, params, clamps, rng_key):
        return self._programs["inference"](params, clamps, rng_key)

    def _evaluate_prepared_batch(self, params, clamps, targets, rng_key):
        return self._programs["evaluation"](params, clamps, targets, rng_key)

    def _score_prepared_batch(self, params, clamps, targets, rng_key):
        return self._programs["evaluation_score"](params, clamps, targets, rng_key)

    def run_batch_evaluation_inference(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        nonshared: Sequence[int],
        params=None,
    ):
        """Run target-free learner inference and install its graph state."""
        params = params if params is not None else self.params
        clamps = self._build_clamps(
            batch,
            task,
            tuple(sorted(nonshared)),
            params=params,
            include_targets=False,
        )
        self.rng_key, state_key = jax.random.split(self.rng_key)
        final_state = self._run_evaluation_inference(params, clamps, state_key)
        self._last_graph_state = final_state
        return final_state, clamps

    def composer_diagnostics_from_state(
        self,
        final_state,
        task: MnistTask,
        nonshared: Sequence[int],
    ) -> Dict[str, float]:
        meta = self.structure.config["hibacaml"]
        feature_names = meta["feature_gate_names"]
        features = jnp.stack(
            [
                final_state.nodes[feature_names[idx]].z_mu
                for idx in range(self.cfg.column_pool.total_columns)
            ],
            axis=1,
        )
        support_mask = self.build_support_mask(nonshared)
        cert_vectors = self.certificate_controller.certificate_matrix(self.persistent_state, support_mask)
        certs = jnp.stack(
            [
                jnp.broadcast_to(
                    cert_vectors[idx],
                    (final_state.batch_size, cert_vectors[idx].shape[0]),
                )
                for idx in range(self.cfg.column_pool.total_columns)
            ],
            axis=1,
        )
        query = batch_query(self._task_query(task), final_state.batch_size)
        details = composer_details(
            self.params.nodes[meta["composer_node"]],
            features,
            certs,
            query,
            self.structure.nodes[meta["composer_node"]].node_info.node_config,
        )
        gate_probs = details["gate_probs"]
        values = jax.device_get(
            (
                jnp.mean(details["gate_entropy"]),
                jnp.mean(details["gate_dev"]),
                jnp.mean(certs),
                jnp.mean(details["prior_kl"]),
                jnp.mean(details["top1_mass"]),
                jnp.mean(details["effective_k"]),
                jnp.max(gate_probs),
                jnp.min(jnp.where(gate_probs > 0.0, gate_probs, 1.0)),
            )
        )
        return {
            "gate_entropy": float(values[0]),
            "gate_deviation": float(values[1]),
            "cert_prior_mean": float(values[2]),
            "prior_kl": float(values[3]),
            "top1_mass": float(values[4]),
            "effective_k": float(values[5]),
            "gate_max": float(values[6]),
            "gate_min_active": float(values[7]),
        }

    @abstractmethod
    def _training_inputs(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        nonshared: Sequence[int],
    ) -> Dict[str, object]:
        """Build the host-side inputs consumed by this learner's update."""

    @abstractmethod
    def _gradients(self, params, inputs, rng_key):
        """Return ``(grads, loss_vector, final_state)`` for one batch."""

    def training_update(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
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
        task: MnistTask,
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

    def _mask_grads(self, grads, nonshared: Sequence[int]):
        return _mask_gradients(
            grads,
            self.build_support_mask(nonshared),
            {
                name: int(name.split("/")[0][3:])
                for name in self.structure.nodes
                if name.startswith("col")
            },
        )

    def _train_batch(
        self,
        task: MnistTask,
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

    def _train_epoch(
        self,
        task: MnistTask,
        nonshared: Sequence[int],
        *,
        epoch_index: int,
        train_batches: int,
        batch_log_every: int,
        evaluate_train_each_epoch: bool,
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
        self._record_timing(
            task.task_id,
            f"epoch_{epoch_index + 1}_seconds",
            training_seconds,
        )

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
        task: MnistTask,
        *,
        final_evaluation_refresh_certificates: bool = True,
        evaluate_train_each_epoch: bool = False,
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
        metrics = self.evaluate_task(
            task.task_id,
            refresh_certificates=final_evaluation_refresh_certificates,
        )
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

    def _prepare_evaluation_batch(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        supports: Sequence[Sequence[int]],
        *,
        params,
        refresh_certificates: bool,
    ):
        """Build target-free clamps and external targets for ordered supports."""
        if not supports:
            raise ValueError("evaluation requires at least one support")
        if len(supports) == 1:
            clamps = self._build_clamps(
                batch,
                task,
                supports[0],
                params=params,
                refresh_certificates=refresh_certificates,
                include_targets=False,
            )
        else:
            clamps = self._build_multi_support_clamps(
                batch,
                task,
                supports,
                params=params,
                refresh_certificates=refresh_certificates,
                include_targets=False,
            )
        targets = batch_targets(batch, repeats=len(supports))
        return clamps, targets

    def _evaluate_supports_batch(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        supports: Sequence[Sequence[int]],
        *,
        params=None,
        refresh_certificates: bool = True,
        score_only: bool = False,
    ):
        """Execute one prepared target-free batch for ordered supports."""
        support_list = tuple(tuple(sorted(support)) for support in supports)
        if not support_list:
            raise ValueError("evaluation requires at least one support")
        params = params if params is not None else self.params
        clamps, targets = self._prepare_evaluation_batch(
            batch,
            task,
            support_list,
            params=params,
            refresh_certificates=refresh_certificates,
        )
        self.rng_key, state_key = jax.random.split(self.rng_key)
        evaluate = (
            self._score_prepared_batch
            if score_only
            else self._evaluate_prepared_batch
        )
        return evaluate(params, clamps, targets, state_key)

    def evaluate_batch_outputs(
        self,
        task: MnistTask,
        batch: Dict[str, jnp.ndarray],
        nonshared: Sequence[int],
        params=None,
        *,
        refresh_certificates: bool = True,
    ):
        """Return target-free predictions and externally scored losses."""
        logits, per_sample_total, _, _ = self._evaluate_supports_batch(
            batch,
            task,
            (nonshared,),
            params=params,
            refresh_certificates=refresh_certificates,
        )
        return logits, per_sample_total

    def evaluate_batch_loss(
        self,
        task: MnistTask,
        batch: Dict[str, jnp.ndarray],
        nonshared: Sequence[int],
        params=None,
        *,
        refresh_certificates: bool = True,
    ) -> float:
        per_sample_total = self._evaluate_supports_batch(
            batch,
            task,
            (nonshared,),
            params=params,
            refresh_certificates=refresh_certificates,
            score_only=True,
        )
        return support_mean_losses(
            per_sample_total,
            1,
            int(batch["x"].shape[0]),
        )[0]

    def evaluate_batch_losses(
        self,
        task: MnistTask,
        batch: Dict[str, jnp.ndarray],
        supports: Sequence[Sequence[int]],
        params=None,
        *,
        refresh_certificates: bool = True,
    ) -> List[float]:
        """Score one batch against several candidate supports, in order."""
        support_list = tuple(supports)
        if not support_list:
            return []
        if len(support_list) == 1:
            return [
                self.evaluate_batch_loss(
                    task,
                    batch,
                    support_list[0],
                    params=params,
                    refresh_certificates=refresh_certificates,
                )
            ]
        per_sample_total = self._evaluate_supports_batch(
            batch,
            task,
            support_list,
            params=params,
            refresh_certificates=refresh_certificates,
            score_only=True,
        )
        return support_mean_losses(
            per_sample_total,
            len(support_list),
            int(batch["x"].shape[0]),
        )

    def evaluate_loader(
        self,
        task: MnistTask,
        loader,
        nonshared: Optional[Sequence[int]] = None,
        *,
        split_name: str,
        mode: Optional[str] = None,
        refresh_certificates: bool,
    ) -> Dict[str, object]:
        """Evaluate one named data split without exposing labels to inference."""

        if not split_name:
            raise ValueError("split_name must be a non-empty string")
        if loader is None:
            raise ValueError(f"loader for split {split_name!r} is None")
        if nonshared is None:
            nonshared = self.persistent_state.current_support.get(
                task.task_id,
                self.current_nonshared,
            )
        nonshared = tuple(sorted(nonshared))
        mode = mode or "current"
        accumulator = EvaluationAccumulator(
            task.output_dim,
            self.cfg.column_pool.total_columns,
        )
        refresh_next = bool(refresh_certificates)

        for batch in loader:
            details = self._evaluate_supports_batch(
                batch,
                task,
                (nonshared,),
                refresh_certificates=refresh_next,
            )
            refresh_next = False
            accumulator.add(batch, details)

        return accumulator.finalize(
            split_name,
            mode,
            refresh_certificates,
        )

    def _support_usage_entropy(self, *, extra_support: Optional[Sequence[int]] = None) -> float:
        supports = [
            snapshot.nonshared for _, snapshot in sorted(self.persistent_state.task_support_snapshots.items())
        ]
        if extra_support is not None:
            supports.append(tuple(sorted(extra_support)))
        if not supports:
            supports = [tuple(self.current_nonshared)]
        counts = jnp.zeros((self.cfg.column_pool.total_columns,), dtype=jnp.float32)
        for support in supports:
            counts = counts.at[jnp.asarray(support, dtype=jnp.int32)].add(1.0)
        probs = counts / jnp.maximum(jnp.sum(counts), 1.0)
        active = probs > 0.0
        return float(-jnp.sum(jnp.where(active, probs * jnp.log(probs + 1e-8), 0.0)))

    def evaluate_task(
        self,
        task_id: int,
        nonshared: Optional[Sequence[int]] = None,
        *,
        mode: Optional[str] = None,
        refresh_certificates: bool = True,
    ) -> Dict[str, object]:
        task = self.tasks[task_id]
        if nonshared is None:
            snapshot = self.persistent_state.task_support_snapshots.get(task_id)
            nonshared = snapshot.nonshared if snapshot is not None else self.persistent_state.current_support.get(task_id, self.current_nonshared)
            mode = "saved" if snapshot is not None else "current"
        else:
            mode = mode or "custom"
        metrics = self.evaluate_loader(
            task,
            task.test_loader,
            nonshared,
            split_name="test",
            mode=mode,
            refresh_certificates=refresh_certificates,
        )
        old_accuracies = [1.0] if task_id == 0 else [
            self.evaluate_loader(
                self.tasks[prev_task_id],
                self.tasks[prev_task_id].test_loader,
                snapshot.nonshared,
                split_name="test",
                mode="saved",
                refresh_certificates=refresh_certificates,
            )["accuracy"]
            for prev_task_id, snapshot in (
                self.persistent_state.task_support_snapshots.items()
            )
            if prev_task_id < task_id
        ]
        metrics = dict(metrics)
        metrics["best_old_accuracy"] = float(
            min(old_accuracies) if old_accuracies else 1.0
        )
        metrics["support_entropy"] = self._support_usage_entropy(
            extra_support=nonshared if mode == "current" else None,
        )
        return metrics

    def evaluate_all_saved_supports(
        self,
        *,
        refresh_certificates: bool = True,
    ) -> Dict[int, Dict[str, object]]:
        task_ids = sorted(self.persistent_state.task_support_snapshots)
        results = {}
        for task_id in task_ids:
            results[task_id] = self.evaluate_task(
                task_id,
                refresh_certificates=refresh_certificates,
            )
        return results
