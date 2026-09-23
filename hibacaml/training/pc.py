"""Full-native predictive-coding learner and inference factors for HiBaCaML."""

from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from fabricpc.core.inference import (
    InferenceBase,
    InferenceSGD,
    gather_inputs,
    run_inference,
)
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.scaling import scale_input_grads, scale_inputs, scale_weight_grads
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.graph_initialization.state_initializer import initialize_graph_state

from hibacaml.nodes.composer import composer_details
from hibacaml.training.shared import (
    composer_context,
    composer_details_from_runtime,
    composer_feature_gate_names,
    evaluation_details_from_state,
    loss_vector_from_state,
    parent_child_energy,
    per_sample_parent_child_from_state,
)
from hibacaml.training.trainer import (
    HiBaCaMLTrainer,
    build_evaluation_programs,
    build_update_programs,
)
from hibacaml.types import MnistTask


def _targets_are_clamped(clamps, structure) -> bool:
    """True when every supervised target is clamped, False when none is."""
    names = (
        structure.task_map["y"],
        structure.task_map["hier_mid"],
        structure.task_map["hier_global"],
    )
    present = tuple(name in clamps for name in names)
    if any(present) and not all(present):
        missing = ", ".join(name for name, ok in zip(names, present) if not ok)
        raise ValueError(
            "predictive coding requires all supervised targets clamped or none; "
            f"missing: {missing}"
        )
    return all(present)


def _auxiliary_weights(structure):
    return structure.config["hibacaml"]["cfg"]


def _parent_child_weight(structure) -> float:
    return float(_auxiliary_weights(structure).hierarchy.parent_child_loss_weight)


def _composer_is_inert(structure) -> bool:
    composer = _auxiliary_weights(structure).composer
    return (
        composer.prior_kl_weight <= 0.0
        and composer.gate_entropy_ceiling_weight <= 0.0
        and composer.gate_dev_weight <= 0.0
    )


def _scaled_inputs(node_name, state, structure):
    node_info = structure.nodes[node_name].node_info
    return scale_inputs(
        gather_inputs(node_info, structure, state),
        node_info.scaling_config,
    )


def _predict_from_state(node_name, node_params, state, structure):
    """One node's z_mu, recomputed from the current state of its inputs."""
    node_info = structure.nodes[node_name].node_info
    return node_info.node_class.forward(
        node_params,
        _scaled_inputs(node_name, state, structure),
        state.nodes[node_name],
        node_info,
    ).z_mu


def _push_prediction_cotangent(state, *, node_name, cotangent, params, structure):
    """Send dE/dz_mu for one node back to the latents that produced it."""

    node_info = structure.nodes[node_name].node_info
    node_params = params.nodes[node_name]
    node_state = state.nodes[node_name]
    node_class = node_info.node_class

    def predict(inputs):
        return node_class.forward(node_params, inputs, node_state, node_info).z_mu

    _, pullback = jax.vjp(predict, _scaled_inputs(node_name, state, structure))
    (input_grads,) = pullback(cotangent)
    input_grads = scale_input_grads(input_grads, node_info.scaling_config)
    for edge_key, grad in input_grads.items():
        source = structure.edges[edge_key].source
        state = update_node_in_state(
            state,
            source,
            latent_grad=state.nodes[source].latent_grad + grad,
        )
    return state


def _add_parent_child_latent_gradients(params, state, structure):
    weight = _parent_child_weight(structure)
    if weight <= 0.0:
        return state
    mid_name = structure.task_map["hier_mid"]
    global_name = structure.task_map["hier_global"]

    # Batch sum, never mean: FabricPC differentiates jnp.sum(energy), so a mean
    # would deliver this factor 1/batch too weak against every other gradient.
    def energy(mid_prediction, global_prediction):
        return jnp.sum(parent_child_energy(mid_prediction, global_prediction, weight))

    # Operands are z_mu. The hierarchy nodes are clamped during training, so their z_latent is the label.
    mid_cotangent, global_cotangent = jax.grad(energy, argnums=(0, 1))(
        state.nodes[mid_name].z_mu,
        state.nodes[global_name].z_mu,
    )
    state = _push_prediction_cotangent(
        state,
        node_name=mid_name,
        cotangent=mid_cotangent,
        params=params,
        structure=structure,
    )
    return _push_prediction_cotangent(
        state,
        node_name=global_name,
        cotangent=global_cotangent,
        params=params,
        structure=structure,
    )


def _add_composer_latent_gradients(params, state, clamps, structure):
    if _composer_is_inert(structure):
        return state
    node_params, features, certs, query, node_config = composer_context(
        params, state, clamps, structure
    )

    def energy(feature_predictions):
        details = composer_details(
            node_params, feature_predictions, certs, query, node_config
        )
        return jnp.sum(details["aux_penalty"])

    cotangents = jax.grad(energy)(features)
    for column, node_name in enumerate(composer_feature_gate_names(structure)):
        state = _push_prediction_cotangent(
            state,
            node_name=node_name,
            cotangent=cotangents[:, column],
            params=params,
            structure=structure,
        )
    return state


def _add_node_grads(grads, node_name, delta):
    merged = jax.tree_util.tree_map(jnp.add, grads.nodes[node_name], delta)
    return grads._replace(nodes={**grads.nodes, node_name: merged})


def _add_parent_child_weight_gradients(params, final_state, structure, grads):
    weight = _parent_child_weight(structure)
    if weight <= 0.0:
        return grads
    mid_name = structure.task_map["hier_mid"]
    global_name = structure.task_map["hier_global"]

    def energy(mid_params, global_params):
        return jnp.sum(
            parent_child_energy(
                _predict_from_state(mid_name, mid_params, final_state, structure),
                _predict_from_state(global_name, global_params, final_state, structure),
                weight,
            )
        )

    deltas = jax.grad(energy, argnums=(0, 1))(
        params.nodes[mid_name], params.nodes[global_name]
    )
    for node_name, delta in zip((mid_name, global_name), deltas):
        scaling = structure.nodes[node_name].node_info.scaling_config
        grads = _add_node_grads(grads, node_name, scale_weight_grads(delta, scaling))
    return grads


def _add_composer_weight_gradients(params, final_state, clamps, structure, grads):
    if _composer_is_inert(structure):
        return grads
    composer_name = structure.config["hibacaml"]["composer_node"]
    features = jnp.stack(
        [
            _predict_from_state(name, params.nodes[name], final_state, structure)
            for name in composer_feature_gate_names(structure)
        ],
        axis=1,
    )
    _, features, certs, query, node_config = composer_context(
        params, final_state, clamps, structure, feature_predictions=features
    )

    def energy(node_params):
        details = composer_details(
            node_params, features, certs, query, node_config
        )
        return jnp.sum(details["aux_penalty"])

    delta = jax.grad(energy)(params.nodes[composer_name])
    scaling = structure.nodes[composer_name].node_info.scaling_config
    return _add_node_grads(grads, composer_name, scale_weight_grads(delta, scaling))


def _add_full_native_weight_gradients(params, final_state, clamps, structure, grads):
    """Add the two factors' local parameter gradients to FabricPC's own."""

    grads = _add_parent_child_weight_gradients(params, final_state, structure, grads)
    return _add_composer_weight_gradients(
        params, final_state, clamps, structure, grads
    )


class HiBaCaMLPCInference(InferenceSGD):
    """FabricPC SGD inference extended with HiBaCaML's two auxiliary factors."""

    @staticmethod
    def forward_value_and_grad(params, state, clamps, structure):
        training = _targets_are_clamped(clamps, structure)
        state = InferenceBase.forward_value_and_grad(params, state, clamps, structure)
        if not training:
            return state
        state = _add_parent_child_latent_gradients(params, state, structure)
        return _add_composer_latent_gradients(params, state, clamps, structure)


def _run_inference_step(params, clamps, structure, rng_key):
    # Feedforward initialization seeds the latents; the relaxation follows it.
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
    grads = _add_full_native_weight_gradients(
        params, final_state, clamps, structure, grads
    )
    # Reporting keeps reading the stored z_mu, which FabricPC leaves one update
    # behind the settled z_latent. The gradients above recompute; these do not.
    composer_details = composer_details_from_runtime(params, final_state, clamps, structure)
    parent_child = per_sample_parent_child_from_state(final_state, structure)
    losses = loss_vector_from_state(
        final_state,
        structure,
        composer_aux_mean=jnp.mean(composer_details["aux_penalty"]),
        parent_child_mean=jnp.mean(parent_child),
    )
    return grads, losses, final_state


def _eval_batch_step(
    params,
    clamps,
    targets,
    structure,
    rng_key,
    *,
    composer_before_parent: bool,
):
    """Run target-free inference, then score predictions against held-out labels."""
    final_state = _run_inference_step(params, clamps, structure, rng_key)
    return evaluation_details_from_state(
        params,
        final_state,
        clamps,
        targets,
        structure,
        composer_before_parent=composer_before_parent,
    )


class HiBaCaMLPCTrainer(HiBaCaMLTrainer):
    """Sequential trainer using FabricPC settling and local weight gradients."""

    COMPOSER_BEFORE_PARENT = False
    CLAMPS_MAY_INCLUDE_TARGETS = True

    @classmethod
    def build_programs(cls, structure, optimizer, cfg):
        # Without this inference algorithm the auxiliary factors never reach
        # settling, which is the report-only behavior this learner replaced.
        inference = structure.config["inference"]
        if not isinstance(inference, HiBaCaMLPCInference):
            raise ValueError(
                f"{cls.__name__} requires a graph built with HiBaCaMLPCInference; "
                f"got {type(inference).__name__}"
            )
        programs = build_update_programs(
            lambda params, inputs, rng_key: _pc_gradient_step(
                params,
                inputs["clamps"],
                structure,
                rng_key,
            ),
            structure,
            optimizer,
            cfg,
            allow_lean=False,
        )
        programs["gradients"] = jax.jit(
            lambda params, clamps, rng_key: _pc_gradient_step(
                params,
                clamps,
                structure,
                rng_key,
            )
        )
        programs.update(
            build_evaluation_programs(
                lambda params, clamps, rng_key: _run_inference_step(
                    params,
                    clamps,
                    structure,
                    rng_key,
                ),
                lambda params, clamps, targets, rng_key: _eval_batch_step(
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
        task: MnistTask,
        nonshared: Sequence[int],
    ) -> Dict[str, object]:
        # Supervised targets are graph clamps during training, and certificates
        # refresh while the clamps are built. Both are scientific state.
        return {
            "clamps": self._build_clamps(
                batch,
                task,
                nonshared,
                include_targets=True,
            )
        }

    def _gradients(self, params, inputs, rng_key):
        return self._programs["gradients"](params, inputs["clamps"], rng_key)
