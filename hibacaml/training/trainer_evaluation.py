"""Clamp preparation, certificate inputs, evaluation and composer diagnostics for the HiBaCaML trainer.

`TrainerEvaluationMixin` defines methods only; they run on a `HiBaCaMLTrainer`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp

from hibacaml.nodes.composer import composer_details
from hibacaml.training.shared import (
    EvaluationAccumulator,
    batch_query,
    batch_targets,
    build_multi_support_clamps,
    build_single_support_clamps,
    support_mean_losses,
    tile_batch,
)
from hibacaml.types import Task


class TrainerEvaluationMixin:
    """Clamps and certificate inputs for training and evaluation, scoring and diagnostics."""

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

    def _task_query(self, task: Task) -> jnp.ndarray:
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
        task: Task,
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
        task: Task,
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
        task: Task,
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
        task: Task,
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

    def _prepare_evaluation_batch(
        self,
        batch: Dict[str, jnp.ndarray],
        task: Task,
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
        task: Task,
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

    def evaluate_batch_loss(
        self,
        task: Task,
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
        task: Task,
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
        task: Task,
        loader,
        nonshared: Optional[Sequence[int]] = None,
        *,
        split_name: str,
        mode: Optional[str] = None,
        refresh_certificates: bool,
        collect_examples: bool = False,
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
            collect_examples=collect_examples,
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
        split: str = "test",
        collect_examples: bool = False,
        reports: Optional[Dict[int, Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        """Evaluate a task and every earlier saved support; `reports` receives each raw report."""
        task = self.tasks[task_id]
        if nonshared is None:
            snapshot = self.persistent_state.task_support_snapshots.get(task_id)
            nonshared = snapshot.nonshared if snapshot is not None else self.persistent_state.current_support.get(task_id, self.current_nonshared)
            mode = "saved" if snapshot is not None else "current"
        else:
            mode = mode or "custom"
        metrics = self.evaluate_loader(
            task,
            task.loader(split),
            nonshared,
            split_name=split,
            mode=mode,
            refresh_certificates=refresh_certificates,
            collect_examples=collect_examples,
        )
        if reports is not None:
            reports[task_id] = metrics
        old_accuracies = [1.0] if task_id == 0 else []
        for prev_task_id, snapshot in self.persistent_state.task_support_snapshots.items():
            if prev_task_id >= task_id:
                continue
            old = self.evaluate_loader(
                self.tasks[prev_task_id],
                self.tasks[prev_task_id].loader(split),
                snapshot.nonshared,
                split_name=split,
                mode="saved",
                refresh_certificates=refresh_certificates,
                collect_examples=collect_examples,
            )
            if reports is not None:
                reports[prev_task_id] = old
            old_accuracies.append(old["accuracy"])
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
        split: str = "test",
    ) -> Dict[int, Dict[str, object]]:
        task_ids = sorted(self.persistent_state.task_support_snapshots)
        results = {}
        for task_id in task_ids:
            results[task_id] = self.evaluate_task(
                task_id,
                refresh_certificates=refresh_certificates,
                split=split,
            )
        return results
