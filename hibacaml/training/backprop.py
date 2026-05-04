"""Backprop runner for the HiBaCaML subsystem."""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp

from fabricpc.graph.state_initializer import FeedforwardStateInit, initialize_graph_state
from hibacaml.training.trainer import (
    HiBaCaMLTrainer,
    _composer_details_from_runtime,
    _hierarchy_parent_child_penalty,
)


def _cross_entropy_from_probs(
    probs: jnp.ndarray,
    targets: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    eps = 1e-7
    safe = jnp.clip(probs, eps, 1.0)
    axes = tuple(range(1, targets.ndim))
    return weight * jnp.mean(-jnp.sum(targets * jnp.log(safe), axis=axes))


class HiBaCaMLBackpropRunner(HiBaCaMLTrainer):
    """End-to-end autodiff runner that preserves HiBaCaML control semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_init = self.structure.config["graph_state_initializer"]
        if not isinstance(state_init, FeedforwardStateInit):
            raise ValueError(
                "HiBaCaMLBackpropRunner requires FeedforwardStateInit on the graph structure"
            )

    def clone(self) -> "HiBaCaMLBackpropRunner":
        ps = self.persistent_state
        saved_ps_params = ps.params
        saved_ps_opt = ps.opt_state
        ps.params = None
        ps.opt_state = None
        try:
            ps_clone = copy.deepcopy(ps)
        finally:
            ps.params = saved_ps_params
            ps.opt_state = saved_ps_opt

        cloned = HiBaCaMLBackpropRunner(
            cfg=self.cfg,
            structure=self.structure,
            params=jax.tree_util.tree_map(lambda x: x, self.params),
            tasks=list(self.tasks.values()),
            optimizer=self.optimizer,
            rng_key=self.rng_key,
            persistent_state=ps_clone,
            create_run_logger=False,
        )
        cloned.opt_state = jax.tree_util.tree_map(lambda x: x, self.opt_state)
        cloned.persistent_state.params = cloned.params
        cloned.persistent_state.opt_state = cloned.opt_state
        cloned.current_phi = self.current_phi
        cloned.current_nonshared = tuple(self.current_nonshared)
        cloned._jit_run_batch_inference = self._jit_run_batch_inference
        cloned._jit_compute_pc_gradients = self._jit_compute_pc_gradients
        cloned._jit_eval_batch = self._jit_eval_batch
        cloned._eval_cache = {}
        return cloned

    def _build_clamps(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        *,
        params=None,
        refresh_certificates: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        meta = self.structure.config["hibacaml"]
        batch_size = int(batch["x"].shape[0])
        params = params if params is not None else self.params
        support_mask = self.build_support_mask(nonshared)
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        cert_vectors = self.shell_controller.certificate_matrix(
            self.persistent_state,
            support_mask,
        )

        clamps = {
            self.structure.task_map["x"]: jnp.asarray(batch["x"], dtype=jnp.float32),
            meta["support_mask_node"]: jnp.broadcast_to(
                support_mask,
                (batch_size, support_mask.shape[0]),
            ),
            meta["task_query_node"]: self._batch_query(task, batch_size),
        }
        for column_index, node_name in meta["cert_input_names"].items():
            vec = cert_vectors[column_index]
            clamps[node_name] = jnp.broadcast_to(vec, (batch_size, vec.shape[0]))
        return clamps

    def _build_multi_support_clamps(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        supports: Sequence[Sequence[int]],
        *,
        params=None,
        refresh_certificates: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        batch_size = int(batch["x"].shape[0])
        params = params if params is not None else self.params
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        support_masks = [self.build_support_mask(nonshared) for nonshared in supports]
        cert_matrices = [
            self.shell_controller.certificate_matrix(self.persistent_state, mask)
            for mask in support_masks
        ]
        meta = self.structure.config["hibacaml"]
        total_batch = batch_size * len(supports)
        clamps: Dict[str, jnp.ndarray] = {
            self.structure.task_map["x"]: jnp.concatenate(
                [jnp.asarray(batch["x"], dtype=jnp.float32)] * len(supports),
                axis=0,
            ),
            meta["support_mask_node"]: jnp.concatenate(
                [
                    jnp.broadcast_to(mask, (batch_size, mask.shape[0]))
                    for mask in support_masks
                ],
                axis=0,
            ),
            meta["task_query_node"]: self._batch_query(task, total_batch),
        }
        for col_idx, node_name in meta["cert_input_names"].items():
            clamps[node_name] = jnp.concatenate(
                [
                    jnp.broadcast_to(matrix[col_idx], (batch_size, matrix[col_idx].shape[0]))
                    for matrix in cert_matrices
                ],
                axis=0,
            )
        return clamps

    def _per_sample_total(
        self,
        final_state,
        stacked_batch: Dict[str, jnp.ndarray],
        params,
        clamps: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        output_name = self.structure.task_map["y"]
        hier_mid_name = self.structure.task_map["hier_mid"]
        hier_global_name = self.structure.task_map["hier_global"]
        eps = 1e-7

        def _ce_per_sample(probs: jnp.ndarray, targets: jnp.ndarray, weight: float) -> jnp.ndarray:
            safe = jnp.clip(probs, eps, 1.0)
            axes = tuple(range(1, targets.ndim))
            return weight * (-jnp.sum(targets * jnp.log(safe), axis=axes))

        task = _ce_per_sample(
            final_state.nodes[output_name].z_mu,
            jnp.asarray(stacked_batch["y"], dtype=jnp.float32),
            1.0,
        )
        hier_mid = _ce_per_sample(
            final_state.nodes[hier_mid_name].z_mu,
            jnp.asarray(stacked_batch["hier_mid"], dtype=jnp.float32),
            self.cfg.hierarchy.mid_loss_weight,
        )
        hier_global = _ce_per_sample(
            final_state.nodes[hier_global_name].z_mu,
            jnp.asarray(stacked_batch["hier_global"], dtype=jnp.float32),
            self.cfg.hierarchy.global_loss_weight,
        )
        composer_details = _composer_details_from_runtime(params, final_state, clamps, self.structure)
        parent_child = _hierarchy_parent_child_penalty(
            final_state,
            self.structure,
            self.cfg.hierarchy.parent_child_loss_weight,
        )
        return task + hier_mid + hier_global + composer_details["aux_penalty"] + parent_child

    def _forward_state(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        params,
        rng_key: jax.Array,
        *,
        update_cache: bool,
        refresh_certificates: bool = True,
    ):
        batch_size = int(batch["x"].shape[0])
        clamps = self._build_clamps(
            batch,
            task,
            nonshared,
            params=params,
            refresh_certificates=refresh_certificates,
        )
        state = initialize_graph_state(
            self.structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=params,
        )
        if update_cache:
            self._last_graph_state = state
            self._last_clamps = clamps
            self._last_task_id = task.task_id
            self._last_targets = {
                "y": jnp.asarray(batch["y"], dtype=jnp.float32),
                "hier_mid": jnp.asarray(batch["hier_mid"], dtype=jnp.float32),
                "hier_global": jnp.asarray(batch["hier_global"], dtype=jnp.float32),
            }
        return state, clamps

    def run_batch_inference(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        params=None,
    ):
        params = params if params is not None else self.params
        self.rng_key, state_key = jax.random.split(self.rng_key)
        return self._forward_state(
            batch,
            task,
            nonshared,
            params,
            state_key,
            update_cache=True,
            refresh_certificates=True,
        )

    def _loss_components(
        self,
        final_state,
        batch: Dict[str, jnp.ndarray],
        params,
        clamps: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        output_name = self.structure.task_map["y"]
        hier_mid_name = self.structure.task_map["hier_mid"]
        hier_global_name = self.structure.task_map["hier_global"]

        task_loss = _cross_entropy_from_probs(
            final_state.nodes[output_name].z_mu,
            jnp.asarray(batch["y"], dtype=jnp.float32),
            weight=1.0,
        )
        hier_mid = _cross_entropy_from_probs(
            final_state.nodes[hier_mid_name].z_mu,
            jnp.asarray(batch["hier_mid"], dtype=jnp.float32),
            weight=self.cfg.hierarchy.mid_loss_weight,
        )
        hier_global = _cross_entropy_from_probs(
            final_state.nodes[hier_global_name].z_mu,
            jnp.asarray(batch["hier_global"], dtype=jnp.float32),
            weight=self.cfg.hierarchy.global_loss_weight,
        )
        composer_details = _composer_details_from_runtime(params, final_state, clamps, self.structure)
        composer = jnp.mean(composer_details["aux_penalty"])
        parent_child = jnp.mean(
            _hierarchy_parent_child_penalty(
                final_state,
                self.structure,
                self.cfg.hierarchy.parent_child_loss_weight,
            )
        )
        return {
            "task": task_loss,
            "hier_mid": hier_mid,
            "hier_global": hier_global,
            "parent_child": parent_child,
            "composer": composer,
            "total": task_loss + hier_mid + hier_global + parent_child + composer,
        }

    def _supervised_losses(self, final_state) -> Dict[str, float]:
        targets = getattr(self, "_last_targets", None)
        if targets is None:
            raise RuntimeError("Backprop loss requested before any batch forward pass")
        losses = self._loss_components(final_state, targets, self.params, self._last_clamps)
        return {name: float(value) for name, value in losses.items()}

    def compute_pc_gradients(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        params=None,
    ):
        params = params if params is not None else self.params
        self.rng_key, state_key = jax.random.split(self.rng_key)

        def loss_fn(p):
            final_state, clamps = self._forward_state(
                batch,
                task,
                nonshared,
                p,
                state_key,
                update_cache=False,
                refresh_certificates=False,
            )
            losses = self._loss_components(final_state, batch, p, clamps)
            return losses["total"], final_state

        (total_loss, final_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        self._last_graph_state = final_state
        self._last_clamps = self._build_clamps(
            batch,
            task,
            nonshared,
            params=params,
            refresh_certificates=False,
        )
        self._last_targets = {
            "y": jnp.asarray(batch["y"], dtype=jnp.float32),
            "hier_mid": jnp.asarray(batch["hier_mid"], dtype=jnp.float32),
            "hier_global": jnp.asarray(batch["hier_global"], dtype=jnp.float32),
        }
        self._last_task_id = task.task_id
        losses = self._loss_components(final_state, batch, params, self._last_clamps)
        losses = {name: float(value) for name, value in losses.items()}
        losses["total"] = float(total_loss)
        return grads, losses, final_state

    def evaluate_batch_loss(
        self,
        task,
        batch: Dict[str, jnp.ndarray],
        nonshared: Sequence[int],
        params=None,
        *,
        refresh_certificates: bool = True,
    ) -> float:
        params = params if params is not None else self.params
        self.rng_key, state_key = jax.random.split(self.rng_key)
        final_state, clamps = self._forward_state(
            batch,
            task,
            nonshared,
            params,
            state_key,
            update_cache=False,
            refresh_certificates=refresh_certificates,
        )
        losses = self._loss_components(final_state, batch, params, clamps)
        return float(losses["total"])

    def evaluate_batch_losses(
        self,
        task,
        batch: Dict[str, jnp.ndarray],
        supports: Sequence[Sequence[int]],
        params=None,
        *,
        refresh_certificates: bool = True,
    ) -> List[float]:
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
        )
        self.rng_key, state_key = jax.random.split(self.rng_key)
        batch_size = int(batch["x"].shape[0])
        total_size = batch_size * len(support_list)
        state = initialize_graph_state(
            self.structure,
            total_size,
            state_key,
            clamps=clamps,
            params=params,
        )
        stacked_batch = {
            "y": jnp.concatenate(
                [jnp.asarray(batch["y"], dtype=jnp.float32)] * len(support_list),
                axis=0,
            ),
            "hier_mid": jnp.concatenate(
                [jnp.asarray(batch["hier_mid"], dtype=jnp.float32)] * len(support_list),
                axis=0,
            ),
            "hier_global": jnp.concatenate(
                [jnp.asarray(batch["hier_global"], dtype=jnp.float32)] * len(support_list),
                axis=0,
            ),
        }
        per_sample = self._per_sample_total(state, stacked_batch, params, clamps)
        jax.block_until_ready(per_sample)
        losses = per_sample.reshape(len(support_list), batch_size).mean(axis=1)
        return [float(value) for value in losses]
