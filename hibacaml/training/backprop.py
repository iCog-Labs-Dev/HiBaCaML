"""Backpropagation learner for HiBaCaML."""

from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from fabricpc.graph_initialization.state_initializer import initialize_graph_state

from hibacaml.training.shared import (
    composer_details_from_runtime,
    cross_entropy_per_sample,
    evaluation_details_from_state,
    hierarchy_parent_child_penalty,
)
from hibacaml.training.trainer import HiBaCaMLTrainer


class HiBaCaMLBackpropTrainer(HiBaCaMLTrainer):
    """End-to-end autodiff learner that preserves HiBaCaML control semantics."""

    COMPOSER_BEFORE_PARENT = True       # Preserve the established learner-specific objective addition order.
    CLAMPS_MAY_INCLUDE_TARGETS = False      # Targets are scored externally and must not be clamped into the graph.

    def _run_evaluation_inference(self, params, clamps, rng_key):
        # FabricPC's feedforward initialization is this learner's whole forward
        # pass; there is no relaxation after it.
        batch_size = next(iter(clamps.values())).shape[0]
        return initialize_graph_state(
            self.structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=params,
        )

    def _evaluate_prepared_batch(self, params, clamps, targets, rng_key):
        final_state = self._run_evaluation_inference(params, clamps, rng_key)
        return evaluation_details_from_state(
            params,
            final_state,
            clamps,
            targets,
            self.structure,
            composer_before_parent=self.COMPOSER_BEFORE_PARENT,
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
            cross_entropy_per_sample(
                final_state.nodes[output_name].z_mu,
                jnp.asarray(batch["y"], dtype=jnp.float32),
                weight=1.0,
            )
        )
        hier_mid = jnp.mean(
            cross_entropy_per_sample(
                final_state.nodes[hier_mid_name].z_mu,
                jnp.asarray(batch["hier_mid"], dtype=jnp.float32),
                weight=self.cfg.hierarchy.mid_loss_weight,
            )
        )
        hier_global = jnp.mean(
            cross_entropy_per_sample(
                final_state.nodes[hier_global_name].z_mu,
                jnp.asarray(batch["hier_global"], dtype=jnp.float32),
                weight=self.cfg.hierarchy.global_loss_weight,
            )
        )
        composer_details = composer_details_from_runtime(params, final_state, clamps, self.structure)
        composer = jnp.mean(composer_details["aux_penalty"])
        parent_child = jnp.mean(
            hierarchy_parent_child_penalty(
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

    def _training_inputs(
        self,
        batch: Dict[str, jnp.ndarray],
        task,
        nonshared: Sequence[int],
    ) -> Dict[str, object]:
        # Targets stay outside the graph; without certificate refresh, clamps are
        # parameter-independent and can be built outside `_gradients`.
        return {
            "clamps": self._build_clamps(
                batch,
                task,
                nonshared,
                refresh_certificates=False,
            ),
            "targets": batch,
        }

    def _gradients(self, params, inputs, rng_key):
        clamps = inputs["clamps"]
        batch = inputs["targets"]

        batch_size = next(iter(clamps.values())).shape[0]

        def loss_fn(p):
            final_state = initialize_graph_state(
                self.structure,
                batch_size,
                rng_key,
                clamps=clamps,
                params=p,
            )
            losses = self._loss_components(final_state, batch, p, clamps)
            return losses["total"], (final_state, losses)

        (total_loss, (final_state, losses)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        loss_vector = jnp.stack(
            [
                losses["task"],
                losses["hier_mid"],
                losses["hier_global"],
                losses["parent_child"],
                losses["composer"],
                # The primal, matching the value the previous code reported.
                total_loss,
            ]
        )
        return grads, loss_vector, final_state
