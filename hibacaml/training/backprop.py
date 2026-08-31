"""Backprop runner for the HiBaCaML subsystem."""

from __future__ import annotations

from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp

from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from hibacaml.training.trainer import (
    HiBaCaMLTrainer,
    _composer_details_from_runtime,
    _cross_entropy_per_sample,
    _hierarchy_parent_child_penalty,
)


class HiBaCaMLBackpropRunner(HiBaCaMLTrainer):
    """End-to-end autodiff runner that preserves HiBaCaML control semantics."""

    def _build_clamps(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        *,
        params=None,
        refresh_certificates: bool = True,
        include_targets: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        # Backprop losses are external, so targets are never graph clamps.
        del include_targets
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
        include_targets: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        # Backprop losses are external, so targets are never graph clamps.
        del include_targets
        batch_size = int(batch["x"].shape[0])
        params = params if params is not None else self.params
        if refresh_certificates:
            self._refresh_certificates(batch_size, params=params)
        support_masks = [self.build_support_mask(nonshared) for nonshared in supports]
        # The unmasked vectors are identical for every support; only the mask differs.
        cert_base = self.shell_controller.certificate_vectors(self.persistent_state)
        cert_matrices = [
            self.shell_controller.mask_certificate_vectors(cert_base, mask)
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
        task = _cross_entropy_per_sample(
            final_state.nodes[output_name].z_mu,
            jnp.asarray(stacked_batch["y"], dtype=jnp.float32),
            1.0,
        )
        hier_mid = _cross_entropy_per_sample(
            final_state.nodes[hier_mid_name].z_mu,
            jnp.asarray(stacked_batch["hier_mid"], dtype=jnp.float32),
            self.cfg.hierarchy.mid_loss_weight,
        )
        hier_global = _cross_entropy_per_sample(
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

    def run_batch_evaluation_inference(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        params=None,
    ):
        """Run target-free evaluation with a single feedforward pass."""
        return self.run_batch_inference(
            batch,
            task,
            nonshared,
            params=params,
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

        task_loss = jnp.mean(
            _cross_entropy_per_sample(
                final_state.nodes[output_name].z_mu,
                jnp.asarray(batch["y"], dtype=jnp.float32),
                weight=1.0,
            )
        )
        hier_mid = jnp.mean(
            _cross_entropy_per_sample(
                final_state.nodes[hier_mid_name].z_mu,
                jnp.asarray(batch["hier_mid"], dtype=jnp.float32),
                weight=self.cfg.hierarchy.mid_loss_weight,
            )
        )
        hier_global = jnp.mean(
            _cross_entropy_per_sample(
                final_state.nodes[hier_global_name].z_mu,
                jnp.asarray(batch["hier_global"], dtype=jnp.float32),
                weight=self.cfg.hierarchy.global_loss_weight,
            )
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

    def compute_training_gradients(
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
            # Report the components computed here rather than recomputing them
            # eagerly afterwards on the same state and params.
            return losses["total"], (final_state, losses)

        (total_loss, (final_state, losses)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        self._last_graph_state = final_state
        losses = {name: float(value) for name, value in losses.items()}
        losses["total"] = float(total_loss)
        return grads, losses, final_state

    def compute_pc_gradients(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
        params=None,
    ):
        """Compatibility wrapper for the former polymorphic method name."""
        return self.compute_training_gradients(
            batch,
            task,
            nonshared,
            params=params,
        )

    def evaluate_batch_outputs(
        self,
        task,
        batch: Dict[str, jnp.ndarray],
        nonshared: Sequence[int],
        params=None,
        *,
        refresh_certificates: bool = True,
    ):
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
        per_sample_total = self._per_sample_total(
            final_state,
            batch,
            params,
            clamps,
        )
        logits = final_state.nodes[self.structure.task_map["y"]].z_mu
        return logits, per_sample_total

    def evaluate_batch_loss(
        self,
        task,
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
