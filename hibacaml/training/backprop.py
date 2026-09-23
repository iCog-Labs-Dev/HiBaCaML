"""Backpropagation learner for HiBaCaML."""

from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from fabricpc.graph_initialization.state_initializer import initialize_graph_state

from hibacaml.training.shared import (
    batch_targets,
    composer_details_from_runtime,
    cross_entropy_per_sample,
    evaluation_details_from_state,
    hierarchy_parent_child_penalty,
)
from hibacaml.training.trainer import (
    HiBaCaMLTrainer,
    build_evaluation_programs,
    build_update_programs,
)


def _bp_loss_components(params, final_state, targets, clamps, structure, cfg):
    output_name = structure.task_map["y"]
    hier_mid_name = structure.task_map["hier_mid"]
    hier_global_name = structure.task_map["hier_global"]

    task_loss = jnp.mean(
        cross_entropy_per_sample(
            final_state.nodes[output_name].z_mu,
            jnp.asarray(targets["y"], dtype=jnp.float32),
            weight=1.0,
        )
    )
    hier_mid = jnp.mean(
        cross_entropy_per_sample(
            final_state.nodes[hier_mid_name].z_mu,
            jnp.asarray(targets["hier_mid"], dtype=jnp.float32),
            weight=cfg.hierarchy.mid_loss_weight,
        )
    )
    hier_global = jnp.mean(
        cross_entropy_per_sample(
            final_state.nodes[hier_global_name].z_mu,
            jnp.asarray(targets["hier_global"], dtype=jnp.float32),
            weight=cfg.hierarchy.global_loss_weight,
        )
    )
    composer_details = composer_details_from_runtime(
        params,
        final_state,
        clamps,
        structure,
    )
    composer = jnp.mean(composer_details["aux_penalty"])
    parent_child = jnp.mean(
        hierarchy_parent_child_penalty(
            final_state,
            structure,
            cfg.hierarchy.parent_child_loss_weight,
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


def _bp_gradient_step(params, inputs, structure, cfg, rng_key):
    clamps = inputs["clamps"]
    targets = inputs["targets"]
    batch_size = next(iter(clamps.values())).shape[0]

    def loss_fn(p):
        final_state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=p,
        )
        losses = _bp_loss_components(
            p,
            final_state,
            targets,
            clamps,
            structure,
            cfg,
        )
        return losses["total"], (final_state, losses)

    (total_loss, (final_state, losses)), grads = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(params)
    loss_vector = jnp.stack(
        [
            losses["task"],
            losses["hier_mid"],
            losses["hier_global"],
            losses["parent_child"],
            losses["composer"],
            total_loss,
        ]
    )
    return grads, loss_vector, final_state


def _bp_inference_step(params, clamps, structure, rng_key):
    batch_size = next(iter(clamps.values())).shape[0]
    return initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )


def _bp_evaluation_step(
    params,
    clamps,
    targets,
    structure,
    rng_key,
    *,
    composer_before_parent,
):
    final_state = _bp_inference_step(params, clamps, structure, rng_key)
    return evaluation_details_from_state(
        params,
        final_state,
        clamps,
        targets,
        structure,
        composer_before_parent=composer_before_parent,
    )


class HiBaCaMLBackpropTrainer(HiBaCaMLTrainer):
    """End-to-end autodiff learner that preserves HiBaCaML control semantics."""

    COMPOSER_BEFORE_PARENT = True
    CLAMPS_MAY_INCLUDE_TARGETS = False

    @classmethod
    def build_programs(cls, structure, optimizer, cfg):
        programs = build_update_programs(
            lambda params, inputs, rng_key: _bp_gradient_step(
                params,
                inputs,
                structure,
                cfg,
                rng_key,
            ),
            structure,
            optimizer,
            cfg,
            allow_lean=True,
        )
        programs.update(
            build_evaluation_programs(
                lambda params, clamps, rng_key: _bp_inference_step(
                    params,
                    clamps,
                    structure,
                    rng_key,
                ),
                lambda params, clamps, targets, rng_key: _bp_evaluation_step(
                    params,
                    clamps,
                    targets,
                    structure,
                    rng_key,
                    composer_before_parent=cls.COMPOSER_BEFORE_PARENT,
                ),
            )
        )
        return programs

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
            "targets": batch_targets(batch),
        }

    def _gradients(self, params, inputs, rng_key):
        return _bp_gradient_step(
            params,
            inputs,
            self.structure,
            self.cfg,
            rng_key,
        )
