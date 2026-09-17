"""Main HiBaCaML trainer."""

from __future__ import annotations

import copy
import itertools
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax

from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization.state_initializer import (
    FeedforwardStateInit,
    initialize_graph_state,
)
from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.replay_bank import SelectorBank
from hibacaml.control.search import ExactSearchService
from hibacaml.control.shells import ShellController
from hibacaml.control.support import (
    build_full_support,
    default_nonshared_support,
    support_mask_from_nonshared,
)
from hibacaml.reporting.logger import log_progress
from hibacaml.graph import initialize_hibacaml_state
from hibacaml.nodes.core import composer_stage2_details
from hibacaml.reporting import (
    append_event,
    export_run_artifacts,
    start_run,
    write_run_state,
)
from hibacaml.reporting.export import write_json
from hibacaml.types import (
    MnistTask,
    PersistentHiBaCaMLState,
    SupportSnapshot,
    TaskSummary,
)

def _composer_details_from_runtime(params, final_state, clamps, structure):
    meta = structure.config["hibacaml"]
    composer_name = meta["composer2_node"]
    feature_names = [meta["feature_gate_names"][idx] for idx in sorted(meta["feature_gate_names"])]
    cert_names = [meta["cert_input_names"][idx] for idx in sorted(meta["cert_input_names"])]
    features = jnp.stack([final_state.nodes[name].z_mu for name in feature_names], axis=1)
    certs = jnp.stack(
        [clamps.get(name, final_state.nodes[name].z_mu) for name in cert_names],
        axis=1,
    )
    query = clamps[meta["task_query_node"]]
    node_info = structure.nodes[composer_name].node_info
    node_params = params.nodes[composer_name]
    return composer_stage2_details(node_params, features, certs, query, node_info.node_config)


def _hierarchy_parent_child_penalty(final_state, structure, weight: float) -> jnp.ndarray:
    if weight <= 0.0:
        return jnp.zeros((final_state.batch_size,), dtype=jnp.float32)
    hier_mid = final_state.nodes[structure.task_map["hier_mid"]].z_mu
    hier_global = final_state.nodes[structure.task_map["hier_global"]].z_mu
    mid_parent = jnp.mean(hier_mid, axis=1)
    return weight * jnp.mean(jnp.square(mid_parent - hier_global), axis=-1)


def _loss_vector_from_state(
    final_state,
    structure,
    *,
    composer_aux_mean: jnp.ndarray,
    parent_child_mean: jnp.ndarray,
) -> jnp.ndarray:
    output_name = structure.task_map["y"]
    hier_mid_name = structure.task_map["hier_mid"]
    hier_global_name = structure.task_map["hier_global"]
    task = jnp.mean(final_state.nodes[output_name].energy)
    hier_mid = jnp.mean(final_state.nodes[hier_mid_name].energy)
    hier_global = jnp.mean(final_state.nodes[hier_global_name].energy)
    total = task + hier_mid + hier_global + parent_child_mean + composer_aux_mean
    return jnp.asarray(
        [task, hier_mid, hier_global, parent_child_mean, composer_aux_mean, total],
        dtype=jnp.float32,
    )


def _loss_dict_from_vector(losses: jnp.ndarray) -> Dict[str, float]:
    return {
        "task": float(losses[0]),
        "hier_mid": float(losses[1]),
        "hier_global": float(losses[2]),
        "parent_child": float(losses[3]),
        "composer": float(losses[4]),
        "total": float(losses[5]),
    }


def _per_sample_parent_child_from_state(final_state, structure) -> jnp.ndarray:
    cfg = structure.config["hibacaml"]["cfg"]
    return _hierarchy_parent_child_penalty(
        final_state,
        structure,
        cfg.hierarchy.parent_child_loss_weight,
    )


def _per_sample_total_from_state(
    final_state,
    structure,
    *,
    composer_aux: jnp.ndarray,
    parent_child: jnp.ndarray,
) -> jnp.ndarray:
    output_name = structure.task_map["y"]
    hier_mid_name = structure.task_map["hier_mid"]
    hier_global_name = structure.task_map["hier_global"]
    return (
        final_state.nodes[output_name].energy
        + final_state.nodes[hier_mid_name].energy
        + final_state.nodes[hier_global_name].energy
        + parent_child
        + composer_aux
    )


def _cross_entropy_per_sample(
    probs: jnp.ndarray,
    targets: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Return externally supervised cross-entropy for each example."""
    safe = jnp.clip(probs, 1e-7, 1.0)
    axes = tuple(range(1, targets.ndim))
    return weight * (-jnp.sum(targets * jnp.log(safe), axis=axes))


def _per_sample_supervised_total_from_state(
    final_state,
    structure,
    targets: Dict[str, jnp.ndarray],
    *,
    composer_aux: jnp.ndarray,
    parent_child: jnp.ndarray,
    composer_before_parent: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute label-based losses without feeding labels into inference."""
    cfg = structure.config["hibacaml"]["cfg"]
    output_name = structure.task_map["y"]
    hier_mid_name = structure.task_map["hier_mid"]
    hier_global_name = structure.task_map["hier_global"]
    task = _cross_entropy_per_sample(
        final_state.nodes[output_name].z_mu,
        targets["y"],
    )
    hier_mid = _cross_entropy_per_sample(
        final_state.nodes[hier_mid_name].z_mu,
        targets["hier_mid"],
        cfg.hierarchy.mid_loss_weight,
    )
    hier_global = _cross_entropy_per_sample(
        final_state.nodes[hier_global_name].z_mu,
        targets["hier_global"],
        cfg.hierarchy.global_loss_weight,
    )
    if composer_before_parent:
        # Preserve the established backprop summation order exactly.
        total = task + hier_mid + hier_global + composer_aux + parent_child
    else:
        total = task + hier_mid + hier_global + parent_child + composer_aux
    losses = jnp.asarray(
        [
            jnp.mean(task),
            jnp.mean(hier_mid),
            jnp.mean(hier_global),
            jnp.mean(parent_child),
            jnp.mean(composer_aux),
            jnp.mean(total),
        ],
        dtype=jnp.float32,
    )
    return total, losses


def _evaluation_details_from_state(
    params,
    final_state,
    clamps,
    targets,
    structure,
    *,
    composer_before_parent: bool = False,
):
    """Score one target-free state and retain already-computed diagnostics."""
    composer_details = _composer_details_from_runtime(
        params,
        final_state,
        clamps,
        structure,
    )
    parent_child = _per_sample_parent_child_from_state(final_state, structure)
    per_sample_total, losses = _per_sample_supervised_total_from_state(
        final_state,
        structure,
        targets,
        composer_aux=composer_details["aux_penalty"],
        parent_child=parent_child,
        composer_before_parent=composer_before_parent,
    )
    logits = final_state.nodes[structure.task_map["y"]].z_mu
    composer_payload = {
        "gate_probs": composer_details["gate_probs"],
        "gate_entropy": composer_details["gate_entropy"],
        "top1_mass": composer_details["top1_mass"],
        "effective_k": composer_details["effective_k"],
        "gate_dev": composer_details["gate_dev"],
        "prior_kl": composer_details["prior_kl"],
    }
    return logits, per_sample_total, losses, composer_payload


def _batch_targets(
    batch: Dict[str, jnp.ndarray],
    *,
    repeats: int = 1,
) -> Dict[str, jnp.ndarray]:
    """Build external evaluation targets, optionally repeated by support."""
    targets = {
        "y": jnp.asarray(batch["y"], dtype=jnp.float32),
        "hier_mid": jnp.asarray(batch["hier_mid"], dtype=jnp.float32),
        "hier_global": jnp.asarray(batch["hier_global"], dtype=jnp.float32),
    }
    if repeats == 1:
        return targets
    return {
        name: jnp.concatenate([value] * repeats, axis=0)
        for name, value in targets.items()
    }


def _per_class_metrics_from_confusion(
    confusion: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-class precision and recall for confusion[target, prediction]."""
    row_totals = jnp.sum(confusion, axis=1)
    column_totals = jnp.sum(confusion, axis=0)
    diagonal = jnp.diag(confusion).astype(jnp.float32)
    recall = jnp.where(row_totals > 0, diagonal / row_totals, 0.0)
    precision = jnp.where(column_totals > 0, diagonal / column_totals, 0.0)
    return precision, recall


def _run_inference_step(params, clamps, structure, rng_key):
    batch_size = next(iter(clamps.values())).shape[0]
    init_state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )
    return run_inference(params, init_state, clamps, structure)


def _pc_gradient_step(params, clamps, structure, rng_key):
    final_state = _run_inference_step(params, clamps, structure, rng_key)
    grads = compute_local_weight_gradients(params, final_state, structure)
    composer_details = _composer_details_from_runtime(params, final_state, clamps, structure)
    parent_child = _per_sample_parent_child_from_state(final_state, structure)
    losses = _loss_vector_from_state(
        final_state,
        structure,
        composer_aux_mean=jnp.mean(composer_details["aux_penalty"]),
        parent_child_mean=jnp.mean(parent_child),
    )
    return grads, losses, final_state


def _eval_batch_step(params, clamps, targets, structure, rng_key):
    """Run target-free inference, then score predictions against held-out labels."""
    final_state = _run_inference_step(params, clamps, structure, rng_key)
    return _evaluation_details_from_state(
        params,
        final_state,
        clamps,
        targets,
        structure,
    )


def _make_jitted_runtime(structure):
    jit_inference = jax.jit(
        lambda params, clamps, rng_key: _run_inference_step(
            params, clamps, structure, rng_key
        )
    )
    jit_pc_gradients = jax.jit(
        lambda params, clamps, rng_key: _pc_gradient_step(
            params, clamps, structure, rng_key
        )
    )
    jit_eval_batch = jax.jit(
        lambda params, clamps, targets, rng_key: _eval_batch_step(
            params, clamps, targets, structure, rng_key
        )
    )
    return jit_inference, jit_pc_gradients, jit_eval_batch


_TRAINER_INSTANCE_COUNTER = itertools.count()


class HiBaCaMLTrainer:
    """Sequential trainer for HiBaCaML."""

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
    ):
        self.cfg = cfg
        self.structure = structure
        # Unique identifier for this trainer instance.
        self.instance_id = next(_TRAINER_INSTANCE_COUNTER)
        # Parsed once: _mask_grads would otherwise re-split every node name each step.
        self._node_column_index = {
            name: int(name.split("/")[0][3:])
            for name in structure.nodes
            if name.startswith("col")
        }
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
        # Rollout clones supply the parent's state; initializing here would allocate a
        # full zeros tree (adamw mu + nu) only to have it overwritten.
        self.opt_state = opt_state if opt_state is not None else self.optimizer.init(params)
        self.persistent_state = persistent_state or initialize_hibacaml_state(params, cfg)
        self.persistent_state.params = self.params
        self.persistent_state.opt_state = self.opt_state
        self.tasks: Dict[int, MnistTask] = {}
        self.current_phi = cfg.phi
        self.current_nonshared = default_nonshared_support(cfg)
        (
            self._jit_run_batch_inference,
            self._jit_compute_pc_gradients,
            self._jit_eval_batch,
        ) = _make_jitted_runtime(structure)
        self._eval_cache: Dict[
            Tuple[int, int, Tuple[int, ...], str, str, bool], Dict[str, object]
        ] = {}
        self.shell_controller = ShellController(cfg, structure)
        # Rollout clones get no run root, which is what keeps their events out
        # of the audit stream.
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
        self.exact_search = ExactSearchService(
            cfg,
            self,
            selector_bank=self.selector_bank,
            run_id=self.run_id,
        )

    def register_tasks(self, tasks: Sequence[MnistTask]) -> None:
        for task in tasks:
            self.tasks[task.task_id] = task

    def clone(self) -> "HiBaCaMLTrainer":
        # JAX arrays are immutable — any update (apply_updates, structural edits)
        # produces new arrays, never mutating existing ones. So sharing the array
        # leaves via tree_map is safe and avoids copying the full weight tensors
        # for every rollout candidate.

        ps = self.persistent_state
        saved_ps_params = ps.params
        saved_ps_opt = ps.opt_state
        saved_epoch_evaluations = ps.epoch_evaluations
        ps.params = None
        ps.opt_state = None
        ps.epoch_evaluations = []
        try:
            ps_clone = copy.deepcopy(ps)
        finally:
            ps.params = saved_ps_params
            ps.opt_state = saved_ps_opt
            ps.epoch_evaluations = saved_epoch_evaluations

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
        )
        cloned.persistent_state.params = cloned.params
        cloned.persistent_state.opt_state = cloned.opt_state
        cloned.current_phi = self.current_phi                          # frozen dataclass
        cloned.current_nonshared = tuple(self.current_nonshared)
        cloned._jit_run_batch_inference = self._jit_run_batch_inference
        cloned._jit_compute_pc_gradients = self._jit_compute_pc_gradients
        cloned._jit_eval_batch = self._jit_eval_batch
        cloned._eval_cache = {}  # rollout clone builds its own; parent's entries are stale after training
        return cloned

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

    def _invalidate_eval_cache(self) -> None:
        self._eval_cache = {}

    def _bump_params_revision(self) -> None:
        self.persistent_state.params_revision += 1
        self._invalidate_eval_cache()

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

    def set_active_support(self, task_id: int, nonshared: Sequence[int], phi) -> None:
        self.set_current_support(task_id, nonshared, phi)

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

    def _batch_query(self, task: MnistTask, batch_size: int) -> jnp.ndarray:
        return jnp.broadcast_to(
            jnp.asarray(task.task_query, dtype=jnp.float32),
            (batch_size, task.task_query.shape[0]),
        )

    def _refresh_certificates(
        self,
        batch_size: int,
        *,
        params=None,
        graph_state=None,
    ) -> None:
        params = params if params is not None else self.params
        state = graph_state if graph_state is not None else self._last_graph_state_or_placeholder(batch_size)
        self.shell_controller.refresh_certificates(params, state, self.persistent_state)

    def _build_single_support_clamps(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        support_mask: jnp.ndarray,
        cert_vectors: Dict[int, jnp.ndarray],
        *,
        include_targets: bool,
    ) -> Dict[str, jnp.ndarray]:
        meta = self.structure.config["hibacaml"]
        batch_size = int(batch["x"].shape[0])
        clamps = {
            self.structure.task_map["x"]: jnp.asarray(batch["x"], dtype=jnp.float32),
            meta["support_mask_node"]: jnp.broadcast_to(support_mask, (batch_size, support_mask.shape[0])),
            meta["task_query_node"]: self._batch_query(task, batch_size),
        }
        if include_targets:
            clamps.update(
                {
                    self.structure.task_map["y"]: jnp.asarray(
                        batch["y"], dtype=jnp.float32
                    ),
                    self.structure.task_map["hier_mid"]: jnp.asarray(
                        batch["hier_mid"], dtype=jnp.float32
                    ),
                    self.structure.task_map["hier_global"]: jnp.asarray(
                        batch["hier_global"], dtype=jnp.float32
                    ),
                }
            )
        for column_index, node_name in meta["cert_input_names"].items():
            vec = cert_vectors[column_index]
            clamps[node_name] = jnp.broadcast_to(vec, (batch_size, vec.shape[0]))
        return clamps

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
        """Build PC clamps; supervised training includes target nodes."""
        batch_size = int(batch["x"].shape[0])
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        support_mask = self.build_support_mask(nonshared)
        cert_vectors = self.shell_controller.certificate_matrix(self.persistent_state, support_mask)
        return self._build_single_support_clamps(
            batch,
            task,
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
        batch_size = int(batch["x"].shape[0])
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        support_masks = [self.build_support_mask(nonshared) for nonshared in supports]
        # The unmasked vectors are identical for every support; only the mask differs.
        cert_base = self.shell_controller.certificate_vectors(self.persistent_state)
        cert_matrices = [
            self.shell_controller.mask_certificate_vectors(cert_base, support_mask)
            for support_mask in support_masks
        ]
        meta = self.structure.config["hibacaml"]
        total_batch = batch_size * len(supports)
        clamps = {
            self.structure.task_map["x"]: jnp.concatenate(
                [jnp.asarray(batch["x"], dtype=jnp.float32)] * len(supports),
                axis=0,
            ),
            meta["support_mask_node"]: jnp.concatenate(
                [jnp.broadcast_to(mask, (batch_size, mask.shape[0])) for mask in support_masks],
                axis=0,
            ),
            meta["task_query_node"]: self._batch_query(task, total_batch),
        }
        if include_targets:
            clamps.update(
                {
                    self.structure.task_map["y"]: jnp.concatenate(
                        [jnp.asarray(batch["y"], dtype=jnp.float32)]
                        * len(supports),
                        axis=0,
                    ),
                    self.structure.task_map["hier_mid"]: jnp.concatenate(
                        [jnp.asarray(batch["hier_mid"], dtype=jnp.float32)]
                        * len(supports),
                        axis=0,
                    ),
                    self.structure.task_map["hier_global"]: jnp.concatenate(
                        [jnp.asarray(batch["hier_global"], dtype=jnp.float32)]
                        * len(supports),
                        axis=0,
                    ),
                }
            )
        for column_index, node_name in meta["cert_input_names"].items():
            clamps[node_name] = jnp.concatenate(
                [
                    jnp.broadcast_to(matrix[column_index], (batch_size, matrix[column_index].shape[0]))
                    for matrix in cert_matrices
                ],
                axis=0,
            )
        return clamps

    def _last_graph_state_or_placeholder(self, batch_size: int):
        if getattr(self, "_last_graph_state", None) is not None and self._last_graph_state.batch_size == batch_size:
            return self._last_graph_state
        zero_nodes = {}
        for node_name, node in self.structure.nodes.items():
            shape = (batch_size, *node.node_info.shape)
            zero_nodes[node_name] = self._make_zero_node_state(shape)
        from fabricpc.core.types import GraphState

        return GraphState(nodes=zero_nodes, batch_size=batch_size)

    @staticmethod
    def _make_zero_node_state(shape):
        from fabricpc.core.types import NodeState

        return NodeState(
            z_latent=jnp.zeros(shape, dtype=jnp.float32),
            z_mu=jnp.zeros(shape, dtype=jnp.float32),
            error=jnp.zeros(shape, dtype=jnp.float32),
            energy=jnp.zeros((shape[0],), dtype=jnp.float32),
            latent_grad=jnp.zeros(shape, dtype=jnp.float32),
        )

    def run_batch_evaluation_inference(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        nonshared: Sequence[int],
        params=None,
    ):
        """Run inference without exposing any class-derived target nodes."""
        params = params if params is not None else self.params
        clamps = self._build_clamps(
            batch,
            task,
            nonshared,
            params=params,
            include_targets=False,
        )
        self.rng_key, state_key = jax.random.split(self.rng_key)
        final_state = self._jit_run_batch_inference(params, clamps, state_key)
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
        cert_vectors = self.shell_controller.certificate_matrix(self.persistent_state, support_mask)
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
        query = self._batch_query(task, final_state.batch_size)
        details = composer_stage2_details(
            self.params.nodes[meta["composer2_node"]],
            features,
            certs,
            query,
            self.structure.nodes[meta["composer2_node"]].node_info.node_config,
        )
        gate_probs = details["gate_probs"]
        return {
            "gate_entropy": float(jnp.mean(details["gate_entropy"])),
            "gate_deviation": float(jnp.mean(details["gate_dev"])),
            "cert_prior_mean": float(jnp.mean(certs)),
            "prior_kl": float(jnp.mean(details["prior_kl"])),
            "top1_mass": float(jnp.mean(details["top1_mass"])),
            "effective_k": float(jnp.mean(details["effective_k"])),
            "gate_max": float(jnp.max(gate_probs)),
            "gate_min_active": float(jnp.min(jnp.where(gate_probs > 0.0, gate_probs, 1.0))),
        }

    def compute_pc_gradients(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        nonshared: Sequence[int],
        params=None,
    ):
        params = params if params is not None else self.params
        clamps = HiBaCaMLTrainer._build_clamps(
            self,
            batch,
            task,
            nonshared,
            params=params,
            include_targets=True,
        )
        self.rng_key, state_key = jax.random.split(self.rng_key)
        grads, loss_vector, final_state = self._jit_compute_pc_gradients(
            params, clamps, state_key
        )
        self._last_graph_state = final_state
        return grads, _loss_dict_from_vector(loss_vector), final_state

    def compute_training_gradients(
        self,
        batch: Dict[str, jnp.ndarray],
        task: MnistTask,
        nonshared: Sequence[int],
        params=None,
    ):
        """Compute gradients for the trainer's configured learning mode."""
        return self.compute_pc_gradients(
            batch,
            task,
            nonshared,
            params=params,
        )

    def _mask_grads(self, grads, nonshared: Sequence[int]):
        keep_columns = set(self.cfg.column_pool.shared_indices + tuple(nonshared))
        masked_nodes = {}
        for node_name, node_grads in grads.nodes.items():
            column_index = self._node_column_index.get(node_name)
            if column_index is not None:
                if column_index not in keep_columns:
                    masked_nodes[node_name] = jax.tree_util.tree_map(jnp.zeros_like, node_grads)
                    continue
            masked_nodes[node_name] = node_grads
        return grads._replace(nodes=masked_nodes)

    def _apply_grads(self, grads):
        grads = self.shell_controller.precision_weight_gradients(grads, self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        self.persistent_state.params = self.params
        self.persistent_state.opt_state = self.opt_state
        self._bump_params_revision()

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
                self.set_boundary_choice(task.task_id, default_support, self.current_phi)
                self.set_current_support(task.task_id, default_support, self.current_phi)

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
        for epoch_idx in range(self.cfg.epochs_per_task):
            epoch_started = time.perf_counter()
            epoch_start_step = self.persistent_state.global_step
            epoch_examples = 0
            epoch_loss_sums = {name: 0.0 for name in final_losses}
            for batch_idx, batch in enumerate(task.train_loader):
                next_step = self.persistent_state.global_step + 1
                cert_due = next_step % self.cfg.cert_refresh_interval == 0
                maintenance_due = (
                    self.cfg.exact_search.enable_exact_search
                    and task.task_id > 0
                    and next_step - self.persistent_state.last_maintenance_step.get(task.task_id, 0)
                    >= self.cfg.exact_search.maintenance_interval
                )
                demotion_due = (
                    self.cfg.exact_search.enable_exact_search
                    and self.cfg.exact_search.enable_demotion_swap_audit
                    and task.task_id > 0
                    and next_step - self.persistent_state.last_demotion_audit_step.get(task.task_id, 0)
                    >= self.cfg.exact_search.demotion_audit_interval
                )
                if batch_idx == 0 or cert_due or maintenance_due or demotion_due or batch_idx + 1 == train_batches:
                    log_progress(
                        f"task={task.task_id} epoch={epoch_idx + 1}/{self.cfg.epochs_per_task} "
                        f"batch={batch_idx + 1}/{train_batches} start step={next_step} "
                        f"hooks(cert_refresh={cert_due}, maintenance={maintenance_due}, demotion={demotion_due})",
                        component="trainer",
                    )

                batch_started = time.perf_counter()
                grads, losses, final_state = self.compute_training_gradients(
                    batch,
                    task,
                    nonshared,
                )
                jax.block_until_ready(final_state.nodes[self.structure.task_map["y"]].z_mu)
                grads = self._mask_grads(grads, nonshared)
                self._apply_grads(grads)
                active_columns = self.active_full_support(nonshared)
                updated_params = self.shell_controller.apply_structural_edits(
                    self.params,
                    final_state,
                    self.persistent_state,
                    active_columns,
                    self.current_phi,
                )
                if updated_params is not self.params:
                    self.params = updated_params
                    self.persistent_state.params = self.params
                    self._bump_params_revision()
                self.persistent_state.global_step += 1
                final_losses = dict(losses)
                batch_size = int(batch["x"].shape[0])
                epoch_examples += batch_size
                for name, value in final_losses.items():
                    epoch_loss_sums[name] += float(value) * batch_size

                if demotion_due:
                    self.exact_search.demotion_swap_audit(task.task_id, final_state, nonshared)
                    self.persistent_state.last_demotion_audit_step[task.task_id] = self.persistent_state.global_step

                if (
                    batch_idx == 0
                    or (batch_idx + 1) % batch_log_every == 0
                    or batch_idx + 1 == train_batches
                ):
                    composer_diag = self.composer_diagnostics_from_state(
                        final_state,
                        task,
                        nonshared,
                    )
                    self.persistent_state.composer_diagnostics[
                        self.persistent_state.global_step
                    ] = {
                        "task_id": task.task_id,
                        "epoch": epoch_idx + 1,
                        "batch": batch_idx + 1,
                        **composer_diag,
                    }
                    log_progress(
                        f"task={task.task_id} epoch={epoch_idx + 1}/{self.cfg.epochs_per_task} "
                        f"batch={batch_idx + 1}/{train_batches} "
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
                            epoch=epoch_idx + 1,
                            batch=batch_idx + 1,
                            train_batches=train_batches,
                            global_step=self.persistent_state.global_step,
                            losses=final_losses,
                            composer=composer_diag,
                        )

                if self.persistent_state.global_step % self.cfg.cert_refresh_interval == 0:
                    self._refresh_certificates(final_state.batch_size, graph_state=final_state)

                if (
                    self.cfg.exact_search.enable_exact_search
                    and task.task_id > 0
                    and self.persistent_state.global_step - self.persistent_state.last_maintenance_step.get(task.task_id, 0)
                    >= self.cfg.exact_search.maintenance_interval
                ):
                    self.exact_search.local_one_swap(task.task_id)
                    nonshared = self.persistent_state.current_support[task.task_id]
                    self.persistent_state.last_maintenance_step[task.task_id] = self.persistent_state.global_step

            training_seconds = time.perf_counter() - epoch_started
            self._record_timing(
                task.task_id,
                f"epoch_{epoch_idx + 1}_seconds",
                training_seconds,
            )

            validation_loader = getattr(task, "validation_loader", None)
            record_epoch = evaluate_train_each_epoch or validation_loader is not None
            if record_epoch:
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
                            mode=f"epoch_{epoch_idx + 1}",
                            refresh_certificates=False,
                        )
                    if validation_loader is not None:
                        validation_metrics = self.evaluate_loader(
                            task,
                            validation_loader,
                            nonshared,
                            split_name="validation",
                            mode=f"epoch_{epoch_idx + 1}",
                            refresh_certificates=False,
                        )
                finally:
                    # Epoch evaluation must not perturb later training randomness.
                    self.rng_key = saved_training_rng

                evaluation_seconds = time.perf_counter() - evaluation_started
                self._record_timing(
                    task.task_id,
                    f"epoch_{epoch_idx + 1}_evaluation_seconds",
                    evaluation_seconds,
                )
                loss_denominator = max(epoch_examples, 1)
                epoch_record = {
                    "task_id": task.task_id,
                    "epoch": epoch_idx + 1,
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

    def _evaluate_batch_details(
        self,
        task: MnistTask,
        batch: Dict[str, jnp.ndarray],
        nonshared: Sequence[int],
        params=None,
        *,
        refresh_certificates: bool = True,
    ):
        """Return one target-free batch plus its already-computed details."""

        params = params if params is not None else self.params
        clamps = self._build_clamps(
            batch,
            task,
            nonshared,
            params=params,
            refresh_certificates=refresh_certificates,
            include_targets=False,
        )
        targets = _batch_targets(batch)
        self.rng_key, state_key = jax.random.split(self.rng_key)
        details = self._jit_eval_batch(
            params,
            clamps,
            targets,
            state_key,
        )
        _, per_sample_total, _, _ = details
        jax.block_until_ready(per_sample_total)
        return details

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
        logits, per_sample_total, _, _ = self._evaluate_batch_details(
            task,
            batch,
            nonshared,
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
        _, per_sample_total = self.evaluate_batch_outputs(
            task,
            batch,
            nonshared,
            params=params,
            refresh_certificates=refresh_certificates,
        )
        return float(jnp.mean(per_sample_total))

    def evaluate_batch_losses(
        self,
        task: MnistTask,
        batch: Dict[str, jnp.ndarray],
        supports: Sequence[Sequence[int]],
        params=None,
        *,
        refresh_certificates: bool = True,
    ):
        support_list = [tuple(sorted(support)) for support in supports]
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
        params = params if params is not None else self.params
        clamps = self._build_multi_support_clamps(
            batch,
            task,
            support_list,
            params=params,
            refresh_certificates=refresh_certificates,
            include_targets=False,
        )
        targets = _batch_targets(batch, repeats=len(support_list))
        self.rng_key, state_key = jax.random.split(self.rng_key)
        _, per_sample_total, _, _ = self._jit_eval_batch(
            params,
            clamps,
            targets,
            state_key,
        )
        jax.block_until_ready(per_sample_total)
        batch_size = int(batch["x"].shape[0])
        losses = per_sample_total.reshape(len(support_list), batch_size).mean(axis=1)
        return [float(value) for value in losses]

    def _evaluation_cache_key(
        self,
        task_id: int,
        nonshared: Sequence[int],
        mode: str,
        split_name: str,
        refresh_certificates: bool,
    ) -> Tuple[int, int, Tuple[int, ...], str, str, bool]:
        return (
            self.persistent_state.params_revision,
            task_id,
            tuple(sorted(nonshared)),
            mode,
            split_name,
            bool(refresh_certificates),
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

        # TODO: Make this cache certificate- and loader-aware before adaptive/controller runs.
        cache_key = self._evaluation_cache_key(
            task.task_id,
            nonshared,
            mode,
            split_name,
            refresh_certificates,
        )
        if self.cfg.exact_search.cache_evaluations and cache_key in self._eval_cache:
            return dict(self._eval_cache[cache_key])

        loss_names = (
            "task",
            "hier_mid",
            "hier_global",
            "parent_child",
            "composer",
            "total",
        )
        loss_sums = jnp.zeros((len(loss_names),), dtype=jnp.float32)
        gate_sums = jnp.zeros(
            (self.cfg.column_pool.total_columns,),
            dtype=jnp.float32,
        )
        selection_counts = jnp.zeros_like(gate_sums)
        confusion = jnp.zeros(
            (task.output_dim, task.output_dim),
            dtype=jnp.int32,
        )
        # Per-batch composer scalars cross to the host in one device_get rather
        # than seven separate blocking reads. Accumulation stays in Python floats
        # at the established order and dtype, so the reported means are unchanged.
        composer_names = (
            "gate_entropy",
            "top1_mass",
            "effective_k",
            "gate_dev",
            "prior_kl",
        )
        composer_sums = dict.fromkeys(composer_names, 0.0)
        correct = 0
        total = 0
        max_selected_columns = 0
        refresh_next = bool(refresh_certificates)

        for batch in loader:
            logits, _, losses, composer = self._evaluate_batch_details(
                task,
                batch,
                nonshared,
                refresh_certificates=refresh_next,
            )
            refresh_next = False
            batch_size = int(logits.shape[0])
            pred = jnp.argmax(logits, axis=-1)
            target = jnp.argmax(jnp.asarray(batch["y"]), axis=-1)
            total += batch_size
            confusion = confusion.at[target, pred].add(1)
            loss_sums = loss_sums + losses * batch_size

            gate_probs = composer["gate_probs"]
            selected = gate_probs > 0.0
            gate_sums = gate_sums + jnp.sum(gate_probs, axis=0)
            selection_counts = selection_counts + jnp.sum(selected, axis=0)

            batch_scalars = jax.device_get(
                tuple(jnp.sum(composer[name]) for name in composer_names)
                + (jnp.sum(pred == target), jnp.max(jnp.sum(selected, axis=-1)))
            )
            for index, name in enumerate(composer_names):
                composer_sums[name] += float(batch_scalars[index])
            correct += int(batch_scalars[-2])
            max_selected_columns = max(max_selected_columns, int(batch_scalars[-1]))

        denominator = max(total, 1)
        component_means = loss_sums / denominator
        per_class_precision, per_class_recall = _per_class_metrics_from_confusion(
            confusion,
        )
        loss_components = {
            name: float(component_means[index])
            for index, name in enumerate(loss_names)
        }
        metrics = {
            "split_name": split_name,
            "mode": mode,
            "accuracy": correct / denominator,
            "cross_entropy": loss_components["task"],
            "mean_loss": loss_components["total"],
            "loss_components": loss_components,
            "num_examples": int(total),
            "correct_examples": int(correct),
            "confusion_matrix": confusion.tolist(),
            "per_class_recall": per_class_recall.tolist(),
            "per_class_precision": per_class_precision.tolist(),
            "column_gate_mean": (gate_sums / denominator).tolist(),
            "column_selection_fraction": (
                selection_counts / denominator
            ).tolist(),
            "gate_entropy": composer_sums["gate_entropy"] / denominator,
            "top1_mass": composer_sums["top1_mass"] / denominator,
            "effective_k": composer_sums["effective_k"] / denominator,
            "gate_deviation": composer_sums["gate_dev"] / denominator,
            "prior_kl": composer_sums["prior_kl"] / denominator,
            "max_selected_columns": int(max_selected_columns),
            "refresh_certificates": bool(refresh_certificates),
        }
        if self.cfg.exact_search.cache_evaluations:
            self._eval_cache[cache_key] = dict(metrics)
        return metrics

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

    def _support_diagnostics(self) -> Dict[str, object]:
        snapshots = self.persistent_state.task_support_snapshots
        supports = {task_id: tuple(snapshot.nonshared) for task_id, snapshot in sorted(snapshots.items())}
        counts = {idx: 0 for idx in range(self.cfg.column_pool.total_columns)}
        for support in supports.values():
            for col in support:
                counts[col] += 1
        pairwise = []
        for left_id, right_id in itertools.combinations(sorted(supports), 2):
            left = set(supports[left_id])
            right = set(supports[right_id])
            pairwise.append(
                {
                    "left_task_id": left_id,
                    "right_task_id": right_id,
                    "jaccard": float(len(left & right) / max(len(left | right), 1)),
                }
            )
        return {
            "support_usage_entropy": self._support_usage_entropy(),
            "column_usage_counts": counts,
            "pairwise_jaccard": pairwise,
            "support_sequence": {task_id: list(snapshot.full_support) for task_id, snapshot in sorted(snapshots.items())},
        }

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

    def export_task_artifacts(
        self,
        task_id: int,
        root: Optional[str | Path] = None,
        *,
        refresh_evaluation_certificates: bool = True,
    ) -> Path:
        run_root = Path(root) if root is not None else self.cfg.experiment_root_path() / f"task_{task_id}"
        log_progress(f"export task={task_id} start root={run_root}", component="report")
        if self.run_root is not None:
            append_event(self.run_root, "export_start", task_id=task_id, root=str(run_root))
        run_root.mkdir(parents=True, exist_ok=True)
        export_run_artifacts(
            self.snapshot(
                refresh_evaluation_certificates=refresh_evaluation_certificates,
            ),
            run_root,
        )
        log_progress(f"export task={task_id} done root={run_root}", component="report")
        if self.run_root is not None:
            append_event(self.run_root, "export_done", task_id=task_id, root=str(run_root))
        return run_root

    def snapshot(
        self,
        *,
        refresh_evaluation_certificates: bool = True,
    ) -> Dict[str, object]:
        return {
            "cfg": self.cfg.to_dict(),
            "current_support": self.persistent_state.current_support,
            "boundary_support": self.persistent_state.boundary_support,
            "support_tables": self.persistent_state.support_tables,
            "support_posterior_tables": self.persistent_state.support_posterior_tables,
            "reserve_recruitment_tables": self.persistent_state.reserve_recruitment_tables,
            "controller_tables": self.persistent_state.controller_tables,
            "local_swap_tables": self.persistent_state.local_swap_tables,
            "demotion_swap_tables": self.persistent_state.demotion_swap_tables,
            "replay_proposals": self.persistent_state.replay_proposals,
            "recently_demoted": self.persistent_state.recently_demoted,
            "certificates": self.persistent_state.certificates,
            "task_support_snapshots": self.persistent_state.task_support_snapshots,
            "support_sequence": {
                task_id: snapshot.full_support
                for task_id, snapshot in sorted(self.persistent_state.task_support_snapshots.items())
            },
            "phi_trajectory": {
                task_id: snapshot.phi
                for task_id, snapshot in sorted(self.persistent_state.task_support_snapshots.items())
            },
            "evaluations": self.evaluate_all_saved_supports(
                refresh_certificates=refresh_evaluation_certificates,
            ),
            "global_step": self.persistent_state.global_step,
            "params_revision": self.persistent_state.params_revision,
            "timing_summaries": self.persistent_state.timing_summaries,
            "support_diagnostics": self._support_diagnostics(),
            "composer_diagnostics": self.persistent_state.composer_diagnostics,
            "epoch_evaluations": self.persistent_state.epoch_evaluations,
            "selector_bank_summary": self.selector_bank.summary(),
            "run_id": self.run_id,
            "experiment_metadata": self.experiment_metadata,
            "evaluation_protocol": "target_free_inference_external_supervision_v1",
        }

    def save_checkpoint(self, task_id: int, root: Optional[str | Path] = None) -> Path:
        checkpoint_root = Path(root) if root is not None else self.cfg.experiment_root_path()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        path = checkpoint_root / self.cfg.reporting.checkpoint_filename
        payload = {
            "task_id": task_id,
            "persistent_state": self.persistent_state,
            "params": self.params,
            "opt_state": self.opt_state,
            "experiment_metadata": self.experiment_metadata,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        if self.run_root is not None:
            append_event(self.run_root, "checkpoint_saved", task_id=task_id, path=str(path))
        return path
