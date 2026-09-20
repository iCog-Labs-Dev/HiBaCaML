"""Trainer-independent computation shared by the PC and backprop learners."""

from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp

from hibacaml.nodes.core import composer_stage2_details
from hibacaml.types import MnistTask

# --------------------------------------------------------------------------
# Clamp assembly
# --------------------------------------------------------------------------

def batch_query(task: MnistTask, batch_size: int) -> jnp.ndarray:
    """Broadcast a task's one-hot query across a batch."""
    
    return jnp.broadcast_to(
        jnp.asarray(task.task_query, dtype=jnp.float32),
        (batch_size, task.task_query.shape[0]),
    )


def build_single_support_clamps(
    structure,
    batch: Dict[str, jnp.ndarray],
    task: MnistTask,
    support_mask: jnp.ndarray,
    cert_vectors: Dict[int, jnp.ndarray],
    *,
    include_targets: bool,
) -> Dict[str, jnp.ndarray]:
    """Assemble clamps for one active support."""

    meta = structure.config["hibacaml"]
    batch_size = int(batch["x"].shape[0])
    clamps = {
        structure.task_map["x"]: jnp.asarray(batch["x"], dtype=jnp.float32),
        meta["support_mask_node"]: jnp.broadcast_to(support_mask, (batch_size, support_mask.shape[0])),
        meta["task_query_node"]: batch_query(task, batch_size),
    }
    if include_targets:
        clamps.update(
            {
                structure.task_map["y"]: jnp.asarray(
                    batch["y"], dtype=jnp.float32
                ),
                structure.task_map["hier_mid"]: jnp.asarray(
                    batch["hier_mid"], dtype=jnp.float32
                ),
                structure.task_map["hier_global"]: jnp.asarray(
                    batch["hier_global"], dtype=jnp.float32
                ),
            }
        )
    for column_index, node_name in meta["cert_input_names"].items():
        vec = cert_vectors[column_index]
        clamps[node_name] = jnp.broadcast_to(vec, (batch_size, vec.shape[0]))
    return clamps


def build_multi_support_clamps(
    structure,
    batch: Dict[str, jnp.ndarray],
    task: MnistTask,
    support_masks: Sequence[jnp.ndarray],
    cert_matrices: Sequence[Dict[int, jnp.ndarray]],
    *,
    include_targets: bool,
) -> Dict[str, jnp.ndarray]:
    """Assemble one stacked clamp set scoring several candidate supports at once."""

    meta = structure.config["hibacaml"]
    batch_size = int(batch["x"].shape[0])
    supports_count = len(support_masks)
    total_batch = batch_size * supports_count
    clamps = {
        structure.task_map["x"]: jnp.concatenate(
            [jnp.asarray(batch["x"], dtype=jnp.float32)] * supports_count,
            axis=0,
        ),
        meta["support_mask_node"]: jnp.concatenate(
            [jnp.broadcast_to(mask, (batch_size, mask.shape[0])) for mask in support_masks],
            axis=0,
        ),
        meta["task_query_node"]: batch_query(task, total_batch),
    }
    if include_targets:
        clamps.update(
            {
                structure.task_map["y"]: jnp.concatenate(
                    [jnp.asarray(batch["y"], dtype=jnp.float32)]
                    * supports_count,
                    axis=0,
                ),
                structure.task_map["hier_mid"]: jnp.concatenate(
                    [jnp.asarray(batch["hier_mid"], dtype=jnp.float32)]
                    * supports_count,
                    axis=0,
                ),
                structure.task_map["hier_global"]: jnp.concatenate(
                    [jnp.asarray(batch["hier_global"], dtype=jnp.float32)]
                    * supports_count,
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

# --------------------------------------------------------------------------
# Objective mathematics
# --------------------------------------------------------------------------

def composer_feature_gate_names(structure):
    """Feature-gate node names in column order."""
    gates = structure.config["hibacaml"]["feature_gate_names"]
    return [gates[idx] for idx in sorted(gates)]


def composer_context(params, state, clamps, structure, *, feature_predictions=None):
    """Assemble the composer's inputs as ``composer_stage2_details`` takes them.

    One builder for every caller -- backprop, evaluation, reporting, and the PC
    auxiliary factor -- so they cannot drift apart. ``feature_predictions``
    overrides the stacked feature-gate predictions; only the PC weight-gradient
    phase supplies it, recomputing them from settled inputs.
    """
    meta = structure.config["hibacaml"]
    composer_name = meta["composer2_node"]
    if feature_predictions is None:
        feature_predictions = jnp.stack(
            [state.nodes[name].z_mu for name in composer_feature_gate_names(structure)],
            axis=1,
        )
    cert_names = [meta["cert_input_names"][idx] for idx in sorted(meta["cert_input_names"])]
    certs = jnp.stack(
        [clamps.get(name, state.nodes[name].z_mu) for name in cert_names],
        axis=1,
    )
    return (
        params.nodes[composer_name],
        feature_predictions,
        certs,
        clamps[meta["task_query_node"]],
        structure.nodes[composer_name].node_info.node_config,
    )


def composer_details_from_runtime(params, final_state, clamps, structure):
    return composer_stage2_details(
        *composer_context(params, final_state, clamps, structure)
    )


def parent_child_energy(
    mid_prediction: jnp.ndarray,
    global_prediction: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Per-example agreement between the two hierarchy predictions."""
    if weight <= 0.0:
        return jnp.zeros((mid_prediction.shape[0],), dtype=jnp.float32)
    mid_parent = jnp.mean(mid_prediction, axis=1)
    return weight * jnp.mean(jnp.square(mid_parent - global_prediction), axis=-1)


def hierarchy_parent_child_penalty(final_state, structure, weight: float) -> jnp.ndarray:
    # Both operands are z_mu. Under supervised PC training the hierarchy nodes are
    # clamped, so their z_latent is the label -- comparing those would measure
    # agreement between targets rather than between predictions.
    return parent_child_energy(
        final_state.nodes[structure.task_map["hier_mid"]].z_mu,
        final_state.nodes[structure.task_map["hier_global"]].z_mu,
        weight,
    )


def loss_vector_from_state(
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


def loss_dict_from_vector(losses: jnp.ndarray) -> Dict[str, float]:
    return {
        "task": float(losses[0]),
        "hier_mid": float(losses[1]),
        "hier_global": float(losses[2]),
        "parent_child": float(losses[3]),
        "composer": float(losses[4]),
        "total": float(losses[5]),
    }


def per_sample_parent_child_from_state(final_state, structure) -> jnp.ndarray:
    cfg = structure.config["hibacaml"]["cfg"]
    return hierarchy_parent_child_penalty(
        final_state,
        structure,
        cfg.hierarchy.parent_child_loss_weight,
    )


def cross_entropy_per_sample(
    probs: jnp.ndarray,
    targets: jnp.ndarray,
    weight: float = 1.0,
) -> jnp.ndarray:
    """Return externally supervised cross-entropy for each example."""
    safe = jnp.clip(probs, 1e-7, 1.0)
    axes = tuple(range(1, targets.ndim))
    return weight * (-jnp.sum(targets * jnp.log(safe), axis=axes))

# --------------------------------------------------------------------------
# Evaluation scoring and accumulation
# --------------------------------------------------------------------------

LOSS_NAMES = (
    "task",
    "hier_mid",
    "hier_global",
    "parent_child",
    "composer",
    "total",
)

COMPOSER_NAMES = (
    "gate_entropy",
    "top1_mass",
    "effective_k",
    "gate_dev",
    "prior_kl",
)


def batch_targets(
    batch: Dict[str, jnp.ndarray],
    *,
    repeats: int = 1,
) -> Dict[str, jnp.ndarray]:
    """Build external evaluation targets in support-major order."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
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


def _per_sample_supervised_total_from_state(
    final_state,
    structure,
    targets: Dict[str, jnp.ndarray],
    *,
    composer_aux: jnp.ndarray,
    parent_child: jnp.ndarray,
    composer_before_parent: bool,
):
    """Compute label-based losses without feeding labels into inference."""
    cfg = structure.config["hibacaml"]["cfg"]
    task = cross_entropy_per_sample(
        final_state.nodes[structure.task_map["y"]].z_mu,
        targets["y"],
    )
    hier_mid = cross_entropy_per_sample(
        final_state.nodes[structure.task_map["hier_mid"]].z_mu,
        targets["hier_mid"],
        cfg.hierarchy.mid_loss_weight,
    )
    hier_global = cross_entropy_per_sample(
        final_state.nodes[structure.task_map["hier_global"]].z_mu,
        targets["hier_global"],
        cfg.hierarchy.global_loss_weight,
    )
    if composer_before_parent:
        # Preserve backprop's established summation order.
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


def evaluation_details_from_state(
    params,
    final_state,
    clamps,
    targets,
    structure,
    *,
    composer_before_parent: bool,
):
    """Score one target-free graph state and return the existing detail tuple."""
    composer_details = composer_details_from_runtime(
        params,
        final_state,
        clamps,
        structure,
    )
    parent_child = per_sample_parent_child_from_state(final_state, structure)
    per_sample_total, losses = _per_sample_supervised_total_from_state(
        final_state,
        structure,
        targets,
        composer_aux=composer_details["aux_penalty"],
        parent_child=parent_child,
        composer_before_parent=composer_before_parent,
    )
    logits = final_state.nodes[structure.task_map["y"]].z_mu
    composer = {
        "gate_probs": composer_details["gate_probs"],
        "gate_entropy": composer_details["gate_entropy"],
        "top1_mass": composer_details["top1_mass"],
        "effective_k": composer_details["effective_k"],
        "gate_dev": composer_details["gate_dev"],
        "prior_kl": composer_details["prior_kl"],
    }
    return logits, per_sample_total, losses, composer


def support_mean_losses(
    per_sample_total,
    support_count: int,
    batch_size: int,
):
    """Reduce flattened support-major losses while preserving support order."""
    if support_count < 1:
        raise ValueError("support_count must be at least 1")
    expected = support_count * batch_size
    actual = int(per_sample_total.size)
    if actual != expected:
        raise ValueError(
            "per_sample_total has "
            f"{actual} entries; expected {support_count} * {batch_size} = {expected}"
        )
    means = per_sample_total.reshape(support_count, batch_size).mean(axis=1)
    return [float(value) for value in means]


def per_class_metrics_from_confusion(confusion: jnp.ndarray):
    """Return precision and recall for ``confusion[target, prediction]``."""
    row_totals = jnp.sum(confusion, axis=1)
    column_totals = jnp.sum(confusion, axis=0)
    diagonal = jnp.diag(confusion).astype(jnp.float32)
    recall = jnp.where(row_totals > 0, diagonal / row_totals, 0.0)
    precision = jnp.where(column_totals > 0, diagonal / column_totals, 0.0)
    return precision, recall


class EvaluationAccumulator:
    """Accumulate batch details into the stable public evaluation report."""

    def __init__(self, output_dim: int, total_columns: int):
        self.loss_sums = jnp.zeros((len(LOSS_NAMES),), dtype=jnp.float32)
        self.gate_sums = jnp.zeros((total_columns,), dtype=jnp.float32)
        self.selection_counts = jnp.zeros_like(self.gate_sums)
        self.confusion = jnp.zeros((output_dim, output_dim), dtype=jnp.int32)
        self.composer_sums = dict.fromkeys(COMPOSER_NAMES, 0.0)
        self.correct = 0
        self.example_count = 0
        self.max_selected_columns = 0

    def add(self, batch, details) -> None:
        logits, _, losses, composer = details
        batch_size = int(logits.shape[0])
        predictions = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(jnp.asarray(batch["y"]), axis=-1)

        self.example_count += batch_size
        self.confusion = self.confusion.at[targets, predictions].add(1)
        self.loss_sums = self.loss_sums + losses * batch_size

        gate_probs = composer["gate_probs"]
        selected = gate_probs > 0.0
        self.gate_sums = self.gate_sums + jnp.sum(gate_probs, axis=0)
        self.selection_counts = self.selection_counts + jnp.sum(selected, axis=0)

        batch_scalars = jax.device_get(
            tuple(jnp.sum(composer[name]) for name in COMPOSER_NAMES)
            + (
                jnp.sum(predictions == targets),
                jnp.max(jnp.sum(selected, axis=-1)),
            )
        )
        for index, name in enumerate(COMPOSER_NAMES):
            self.composer_sums[name] += float(batch_scalars[index])
        self.correct += int(batch_scalars[-2])
        self.max_selected_columns = max(
            self.max_selected_columns,
            int(batch_scalars[-1]),
        )

    def finalize(
        self,
        split_name: str,
        mode: str,
        refresh_certificates: bool,
    ) -> Dict[str, object]:
        denominator = max(self.example_count, 1)
        component_means = self.loss_sums / denominator
        precision, recall = per_class_metrics_from_confusion(self.confusion)
        loss_components = {
            name: float(component_means[index])
            for index, name in enumerate(LOSS_NAMES)
        }
        return {
            "split_name": split_name,
            "mode": mode,
            "accuracy": self.correct / denominator,
            "cross_entropy": loss_components["task"],
            "mean_loss": loss_components["total"],
            "loss_components": loss_components,
            "num_examples": int(self.example_count),
            "correct_examples": int(self.correct),
            "confusion_matrix": self.confusion.tolist(),
            "per_class_recall": recall.tolist(),
            "per_class_precision": precision.tolist(),
            "column_gate_mean": (self.gate_sums / denominator).tolist(),
            "column_selection_fraction": (
                self.selection_counts / denominator
            ).tolist(),
            "gate_entropy": self.composer_sums["gate_entropy"] / denominator,
            "top1_mass": self.composer_sums["top1_mass"] / denominator,
            "effective_k": self.composer_sums["effective_k"] / denominator,
            "gate_deviation": self.composer_sums["gate_dev"] / denominator,
            "prior_kl": self.composer_sums["prior_kl"] / denominator,
            "max_selected_columns": int(self.max_selected_columns),
            "refresh_certificates": bool(refresh_certificates),
        }
