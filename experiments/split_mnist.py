"""Split-MNIST experiment runner for HiBaCaML."""

from __future__ import annotations
import ast
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(".")

from hibacaml.reporting.logger import log_progress
from hibacaml import (
    build_split_mnist_tasks,
    build_trainer,
    export_run_artifacts,
    make_hibacaml_config,
    override,
    prepare_run_root,
)
from hibacaml.experiment import log_runtime
from hibacaml.control.router import Router, canonical_support
from hibacaml.data.mnist import (
    DESKEWED_EXTENDED_TANGENT, SHIFT_ROTATION_TANGENT, ROUTER_HISTORIES, _normalize_images, router_profile,
)
from hibacaml.reporting.integrity import fingerprint
from hibacaml.training.shared import build_single_support_clamps
from hibacaml.reporting import (
    best_swap_gains,
    build_run_snapshot,
    export_task_artifacts,
    mean_forgetting,
    mean_seen_accuracy,
    phi_trajectory,
    plot_sequential_run,
    save_checkpoint,
    support_entropy_trajectory,
    support_sequence,
    task_metric_rows,
    task_summary_row,
    write_run_records,
)


MODE = "default"
LEARNING = "backprop"                  # "pc" or "backprop"
# Both modes use 20 centroids per support. The default deskews digits and adds
# scale, shear and stroke thickness to the fast control's shift and rotation matching.
ROUTER_MODE = DESKEWED_EXTENDED_TANGENT  # use SHIFT_ROTATION_TANGENT for the fast control, or None
TASKS_LIMIT = None                     # e.g. 2, 5, or None for all
VALIDATION_FRACTION = 0.10             # per task, carved from its training data; audits read it

# Any HiBaCaMLConfig field to override for this run.
OVERRIDES = {
    "batch_size": 1024,
    "patch_size": (28, 28),            # flattened (28, 28), or sixteen patches (7, 7)
    "exact_search__static_support_batch_size": 32,
    "exact_search__neighbor_support_batch_size": 32,
    "exact_search__boundary_current_data_batch_size": 128,
    "exact_search__rollout_train_data_batch_size": 128,
    "exact_search__boundary_worst_old_data_batch_size": 128,
    "exact_search__boundary_mixed_old_data_batch_size": 128,
    "exact_search__local_swap_audit_data_batch_size": 128,
    "exact_search__demotion_audit_data_batch_size": 128,
}


def load_reference_router(study_root, *, mode, patch_size, history, supports):
    """Restore a frozen MNIST prefix from its original count/refinement study."""
    profile = router_profile(mode, patch_size)
    root = Path(study_root)
    model, _ = ROUTER_HISTORIES[tuple(patch_size)]
    summary = json.loads((root / "run_summary.json").read_text())
    if (summary["status"] != "complete" or summary["split"] != "validation"
            or Path(summary["histories"][model]["source"]).name != history):
        raise ValueError("reference study and requested history disagree")
    # These keys belong to the original study archives.
    if profile.identity["mode"] == SHIFT_ROTATION_TANGENT:
        folder, key = "centroids", "k20"
    else:
        folder, key = "router_state", "D_centroids_k20"
    with (root / "fit.csv").open(newline="") as stream:
        rows = {int(r["task"]): r for r in csv.DictReader(stream)
                if r["model"] == model and int(r["count"]) == profile.centroids_per_support}
    entries = []
    for boundary, expected in enumerate(supports):
        row = rows[boundary]
        with np.load(root / model / folder / f"entry_{boundary}.npz", allow_pickle=False) as entry:
            support, centers = canonical_support(entry["support"]), entry[key]
        if support != canonical_support(expected) or support != canonical_support(ast.literal_eval(row["support"])):
            raise ValueError(f"reference support mismatch at boundary {boundary}")
        record = dict(support=support, seed=profile.fit_seed(boundary), examples=int(row["examples"]),
                      iterations=int(row["iterations"]), inertia=float(row["inertia"]),
                      fit_rows_sha256=row["fit_rows_sha256"], centroid_sha256=row["centroid_sha256"])
        entries.append((centers, record))
    return Router.from_entries(profile, history, entries)


def _predict_support(trainer, images, support, key):
    """Predict from original normalized pixels and saved certificates with a zero query."""
    cfg = trainer.cfg
    mask = trainer.build_support_mask(support)
    certs = trainer.certificate_controller.certificate_matrix(trainer.persistent_state, mask)
    zero_query = jnp.zeros(cfg.composer.query_dim, dtype=jnp.float32)
    # One compiled batch shape even when only one image routes to this support.
    zeros = jnp.zeros((cfg.batch_size, cfg.output_dim), dtype=jnp.float32)
    targets = {"y": zeros, "hier_global": zeros,
               "hier_mid": jnp.zeros((cfg.batch_size, cfg.hierarchy.mid_targets, cfg.output_dim), dtype=jnp.float32)}
    parts = []
    for start in range(0, len(images), cfg.batch_size):
        batch = images[start:start + cfg.batch_size]
        size = len(batch)
        if size < cfg.batch_size:
            batch = np.pad(batch, ((0, cfg.batch_size - size), (0, 0), (0, 0), (0, 0)), mode="edge")
        clamps = build_single_support_clamps(trainer.structure, {"x": batch}, zero_query, mask,
                                             certs, include_targets=False)
        probabilities = trainer._evaluate_prepared_batch(trainer.params, clamps, targets, key)[0]
        parts.append(np.asarray(probabilities)[:size])
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite routed prediction")
    return result


def _group_predictions(trainer, images, decision, key, *, known_support=None, known_probs=None):
    probabilities = np.empty((len(images), trainer.cfg.output_dim), dtype=np.float32)
    for index, support in enumerate(decision.catalogue):
        positions = np.flatnonzero(decision.choices == index)
        if len(positions):
            probabilities[positions] = (known_probs[positions] if support == known_support else
                                        _predict_support(trainer, images[positions], support, key))
    return probabilities


def predict_routed(trainer, router, raw_images, key):
    """Image-only prediction. Return probabilities in input order and the routing decision."""
    snapshots = trainer.persistent_state.task_support_snapshots
    identity = router.profile.identity
    mode = identity.get("mode")
    if (tuple(snapshots[t].nonshared for t in sorted(snapshots)) != router.supports
            or mode not in (DESKEWED_EXTENDED_TANGENT, SHIFT_ROTATION_TANGENT)
            or identity != router_profile(mode, trainer.cfg.patch_size).identity
            or (trainer.run_id and trainer.run_id != router.history)):
        raise ValueError("router bank does not match this model and its saved supports")
    decision = router.route(raw_images)
    images = _normalize_images(np.asarray(raw_images))
    return _group_predictions(trainer, images, decision, key), decision


def _macro(rows):
    return {name: float(np.mean([row[name] for row in rows]))
            for name in ("accuracy", "ce", "own_support_rate", "oracle_accuracy", "oracle_ce", "oracle_gap")}


def _routing_validation(trainer, router, tasks):
    """Matched zero-query measurements. Only the evaluator knows the supplied support."""
    reports = {}
    key = jax.random.PRNGKey(0)
    for task in tasks:
        loader = task.validation_loader
        n = loader.examples
        if not n:
            raise ValueError(f"no validation examples for task {task.task_id}")
        supplied = trainer.persistent_state.task_support_snapshots[task.task_id].nonshared
        choices, routed, oracle = [], [], []
        for start in range(0, n, loader.batch_size):
            raw = loader.images[start:min(start + loader.batch_size, n)]
            decision = router.route(raw)
            images = _normalize_images(raw)
            batch_key = jax.random.fold_in(key, start // loader.batch_size)
            own = _predict_support(trainer, images, supplied, batch_key)
            chosen = _group_predictions(trainer, images, decision, batch_key,
                                        known_support=supplied, known_probs=own)
            choices.append(decision.choices)
            routed.append(chosen)
            oracle.append(own)

        labels = loader.targets[:n].argmax(axis=1)
        indices = np.concatenate(choices).astype(np.int32)
        record = {"examples": n, "choices_sha256": fingerprint(indices),
                  "own_support_rate": float(np.mean(indices == decision.catalogue.index(supplied)))}
        for prefix, parts in (("", routed), ("oracle_", oracle)):
            probs = np.concatenate(parts)
            record[prefix + "accuracy"] = float(np.mean(probs.argmax(axis=1) == labels))
            true_probs = probs[np.arange(n), labels].astype(np.float64)
            record[prefix + "ce"] = float(-np.log(np.clip(true_probs, 1e-7, 1.)).mean())
        record["oracle_gap"] = record["oracle_accuracy"] - record["accuracy"]
        reports[task.task_id] = record
    return {"tasks": reports, "macro": _macro(reports.values()),
            "examples": sum(row["examples"] for row in reports.values())}


def _observe_routing(trainer, router, seen, row_digests, checkpoint, run_root):
    task = seen[-1]
    boundary = task.task_id
    snapshots = trainer.persistent_state.task_support_snapshots
    support = snapshots[boundary].nonshared
    if router.supports != tuple(snapshots[i].nonshared for i in range(boundary)):
        raise ValueError("previous saved supports changed after router acquisition")
    pool = trainer.cfg.column_pool
    eligible = range(pool.shared_count, pool.total_columns)
    if len(support) != pool.topk_nonshared or any(i not in eligible for i in support):
        raise ValueError(f"invalid saved nonshared support {support}")
    router.acquire(boundary, support, task.train_loader.images, row_digests[boundary])
    bank_path = checkpoint.parent.parent / "router_bank.npz"
    with checkpoint.open("rb") as stream:
        router.save(bank_path, hashlib.file_digest(stream, "sha256").hexdigest())
    report = _routing_validation(trainer, router, seen)
    log_progress(f"task={boundary} {router.profile.identity['mode']} validation={report['macro']['accuracy']:.4f} "
                 f"oracle={report['macro']['oracle_accuracy']:.4f}", component="runner")
    return dict(report, boundary=boundary, bank=str(bank_path.relative_to(run_root)))


def _run_tasks(
    trainer,
    tasks,
    learning: str,
    run_root: Path,
    *,
    router=None,
    row_digests=None,
) -> dict:
    """Train and evaluate every task in sequence for one learning mode."""

    accuracy_matrix: Dict[int, Dict[int, float]] = {}
    task_summaries: List[Dict[str, object]] = []
    artifact_roots: Dict[int, str] = {}
    checkpoint_paths: Dict[int, str] = {}
    routed_boundaries = []

    for task in tasks:
        log_progress(f"task={task.task_id} train start", component="runner")
        summary = trainer.train_task(task)
        # Reporting observes the run, so it never refreshes certificates.
        eval_results = trainer.evaluate_all_saved_supports(refresh_certificates=False)

        accuracy_matrix[task.task_id] = {
            eval_task_id: metrics["accuracy"]
            for eval_task_id, metrics in eval_results.items()
        }

        task_summaries.append(task_summary_row(summary))

        artifact_root = export_task_artifacts(trainer, task.task_id)
        checkpoint_root = run_root / f"task_{task.task_id}" / "checkpoints"
        checkpoint_path = save_checkpoint(
            trainer,
            task.task_id,
            root=checkpoint_root,
        )

        artifact_roots[task.task_id] = str(artifact_root)
        checkpoint_paths[task.task_id] = str(checkpoint_path)

        if router is not None:
            seen = tasks[:task.task_id + 1]
            routed_boundaries.append(_observe_routing(trainer, router, seen, row_digests, checkpoint_path, run_root))

        log_progress(
            f"task={task.task_id} train done accuracy={summary.accuracy:.4f} "
            f"support={summary.support_indices}",
            component="runner",
        )
        gc.collect()

    snapshot = build_run_snapshot(trainer)
    seen_accuracy = mean_seen_accuracy(accuracy_matrix)
    forgetting = mean_forgetting(accuracy_matrix)
    supports = support_sequence(trainer)
    swap_gains = best_swap_gains(trainer)
    rows = task_metric_rows(
        sorted(accuracy_matrix), seen_accuracy, forgetting, supports, swap_gains, task_summaries
    )

    export_run_artifacts(snapshot, run_root / "final")

    results = {
        "learning": learning,
        "accuracy_matrix": accuracy_matrix,
        "mean_seen_accuracy": seen_accuracy,
        "mean_forgetting": forgetting,
        "support_sequence": supports,
        "phi_trajectory": phi_trajectory(trainer),
        "best_swap_gains": swap_gains,
        "support_entropy_trajectory": support_entropy_trajectory(task_summaries),
        "task_summaries": task_summaries,
        "artifact_roots": artifact_roots,
        "checkpoint_paths": checkpoint_paths,
    }
    if router is not None:
        macros = [row["macro"] for row in routed_boundaries]
        results["routing"] = {
            "mode": router.profile.identity["mode"], "patch_size": trainer.cfg.patch_size,
            "centroids_per_support": router.profile.centroids_per_support,
            "split": "validation", "query": "zero",
            "boundaries": routed_boundaries, "sequence": _macro(macros[1:]) if len(macros) > 1 else None,
            "final": macros[-1] if macros else None,
        }

    write_run_records(run_root, results, rows)

    return results


## Main entry point
def run_experiment(
    cfg,
    *,
    learning: str = LEARNING,
    tasks_limit: int | None = TASKS_LIMIT,
    plots_dir: str | None = None,
    run_id: str | None = None,
    validation_fraction: float = VALIDATION_FRACTION,
):
    if learning not in ("pc", "backprop"):
        raise ValueError("learning must be 'pc' or 'backprop'")

    log_runtime()

    if run_id is None:
        from datetime import datetime, timezone
        run_id = f"SMNIST_{learning}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    mode = ROUTER_MODE if learning == "backprop" else None
    router = Router(router_profile(mode, cfg.patch_size), run_id) if mode is not None else None
    row_digests = {} if router is not None else None
    tasks = build_split_mnist_tasks(
        cfg, limit=tasks_limit, validation_fraction=validation_fraction, row_digests=row_digests,
    )
    log_progress(f"loaded {len(tasks)} Split-MNIST tasks", component="runner")

    cfg, run_output_root = prepare_run_root(cfg, run_id)
    run_plots_dir = Path(plots_dir) if plots_dir else run_output_root / "plots"

    _, trainer = build_trainer(
        cfg,
        tasks,
        learning,
        run_id=run_id,
        experiment_metadata={"validation_fraction": validation_fraction},
    )

    results = _run_tasks(
        trainer,
        tasks,
        learning,
        run_output_root,
        router=router,
        row_digests=row_digests,
    )
    plot_sequential_run(results, run_plots_dir)

    log_progress(
        f"run complete id={run_id} dir={run_output_root}", component="runner"
    )

    return {
        "cfg": cfg,
        "results": results,
        "run_id": run_id,
        "run_output_root": run_output_root,
        "selector_state_root": cfg.selector_state_path(),
        "plots_dir": run_plots_dir,
        "summary_path": run_output_root / "run_summary.json",
        "table_path": run_output_root / "task_summary.csv",
    }


if __name__ == "__main__":
    cfg = make_hibacaml_config(MODE)
    cfg = override(cfg, **OVERRIDES)
    run_output = run_experiment(cfg)
