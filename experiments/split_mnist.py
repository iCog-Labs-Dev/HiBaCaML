"""Split-MNIST experiment runner for HiBaCaML."""

from __future__ import annotations
import gc
import sys
from pathlib import Path
from typing import Dict, List

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
TASKS_LIMIT = None                     # e.g. 2, 5, or None for all
VALIDATION_FRACTION = 0.10             # per task, carved from its training data; audits read it

# Any HiBaCaMLConfig field to override for this run.
OVERRIDES = {
    "batch_size": 1024,
    "patch_size": (28, 28),            # flattened tokens: the Stage 0 best configuration
    "exact_search__static_support_batch_size": 32,
    "exact_search__neighbor_support_batch_size": 32,
    "exact_search__boundary_current_data_batch_size": 128,
    "exact_search__rollout_train_data_batch_size": 128,
    "exact_search__boundary_worst_old_data_batch_size": 128,
    "exact_search__boundary_mixed_old_data_batch_size": 128,
    "exact_search__local_swap_audit_data_batch_size": 128,
    "exact_search__demotion_audit_data_batch_size": 128,
}


# Core experiment loop
def _run_tasks(
    trainer,
    tasks,
    learning: str,
    run_root: Path,
) -> dict:
    """Train and evaluate every task in sequence for one learning mode."""

    accuracy_matrix: Dict[int, Dict[int, float]] = {}
    task_summaries: List[Dict[str, object]] = []
    artifact_roots: Dict[int, str] = {}
    checkpoint_paths: Dict[int, str] = {}

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

    tasks = build_split_mnist_tasks(
        cfg, limit=tasks_limit, validation_fraction=validation_fraction
    )

    log_progress(f"loaded {len(tasks)} Split-MNIST tasks", component="runner")

    if run_id is None:
        from datetime import datetime, timezone
        run_id = f"SMNIST_{learning}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
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
