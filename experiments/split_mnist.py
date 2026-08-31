"""Split-MNIST experiment runner for HiBaCaML."""

from __future__ import annotations
import csv
import dataclasses
import gc
import json
import sys
import jax
from pathlib import Path
from typing import Dict, List, Sequence

from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

sys.path.append(".")

from hibacaml.debug import log_progress
from hibacaml import (
    HiBaCaMLBackpropRunner,
    HiBaCaMLTrainer,
    build_split_mnist_tasks,
    create_hibacaml_structure,
    export_run_artifacts,
    make_hibacaml_config,
    override,
)

from hibacaml.reporting import (
    plot_accuracy_forgetting,
    plot_support_table,
    plot_swap_gains,
    print_pre_run_review,
)


MODE = "default"                       # "default" or "smoke"
LEARNING = "backprop"                  # "pc" or "backprop"
TASKS_LIMIT = None                     # e.g. 2, 5, or None for all

# Any HiBaCaMLConfig field to override for this run.
OVERRIDES = {
    "batch_size": 768,
    "exact_search__static_support_batch_size": 16,
    "exact_search__neighbor_support_batch_size": 16,
    "exact_search__boundary_current_data_batch_size": 128,
    "exact_search__rollout_train_data_batch_size": 128,
    "exact_search__boundary_worst_old_data_batch_size": 128,
    "exact_search__boundary_mixed_old_data_batch_size": 128,
    "exact_search__local_swap_audit_data_batch_size": 128,
    "exact_search__demotion_audit_data_batch_size": 128,
}


# Helpers
def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(val) for key, val in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


# Trainer construction
def build_trainer(cfg, tasks, learning: str, *, run_id: str = ""):
    """Construct the HiBaCaML trainer for one learning mode."""
    inference = InferenceSGD(eta_infer=cfg.eta_infer, infer_steps=cfg.infer_steps)
    graph_state_initializer = FeedforwardStateInit()
    structure = create_hibacaml_structure(
        cfg,
        inference,
        graph_state_initializer=graph_state_initializer,
    )
    log_progress(f"graph ready with {len(structure.nodes)} nodes", component="runner")

    params = initialize_params(structure, jax.random.PRNGKey(cfg.seed))
    log_progress("parameter initialization complete", component="runner")

    trainer_cls = HiBaCaMLBackpropRunner if learning == "backprop" else HiBaCaMLTrainer
    trainer = trainer_cls(cfg, structure, params, tasks=tasks, run_id=run_id)
    log_progress(f"trainer constructed class={type(trainer).__name__}", component="runner")
    return structure, trainer


# Metric helpers
def _mean_seen_accuracy(accuracy_matrix: Dict[int, Dict[int, float]]) -> List[float]:
    curve: List[float] = []
    for task_id in sorted(accuracy_matrix):
        row = accuracy_matrix[task_id]
        curve.append(sum(row.values()) / max(len(row), 1))
    return curve


def _mean_forgetting_curve(accuracy_matrix: Dict[int, Dict[int, float]]) -> List[float]:
    curve: List[float] = []
    task_ids = sorted(accuracy_matrix)
    for boundary_id in task_ids:
        row = accuracy_matrix[boundary_id]
        forgetting = []
        for eval_task_id in sorted(row):
            best_so_far = max(
                accuracy_matrix[past_id][eval_task_id]
                for past_id in task_ids
                if past_id <= boundary_id and eval_task_id in accuracy_matrix[past_id]
            )
            forgetting.append(best_so_far - row[eval_task_id])
        curve.append(sum(forgetting) / max(len(forgetting), 1))
    return curve


def _support_sequence(trainer) -> Dict[int, List[int]]:
    return {
        task_id: list(snapshot.full_support)
        for task_id, snapshot in sorted(trainer.persistent_state.task_support_snapshots.items())
    }


def _phi_trajectory(trainer) -> Dict[int, Dict[str, float]]:
    return {
        task_id: {
            "outer_quantile": snapshot.phi.outer_quantile,
            "middle_quantile": snapshot.phi.middle_quantile,
            "replacement_margin_base": snapshot.phi.replacement_margin_base,
            "demotion_min_role_gain": snapshot.phi.demotion_min_role_gain,
        }
        for task_id, snapshot in sorted(trainer.persistent_state.task_support_snapshots.items())
    }


def _support_entropy_trajectory(task_summaries: List[Dict[str, object]]) -> List[float]:
    """Per-task support-usage entropy trajectory using the trainer's own natural-log metric."""
    return [
        float(row["support_entropy"])
        for row in sorted(task_summaries, key=lambda r: r["task_id"])
        if "support_entropy" in row
    ]


def _best_swap_gains(trainer) -> Dict[int, float]:
    gains = {}
    for task_id, rows in sorted(trainer.persistent_state.local_swap_tables.items()):
        gains[task_id] = max((row.gain for row in rows), default=0.0)
    return gains


def _task_metric_rows(
    task_ids: Sequence[int],
    mean_seen_accuracy: Sequence[float],
    mean_forgetting: Sequence[float],
    support_sequence: Dict[int, List[int]],
    best_swap_gains: Dict[int, float],
    task_summaries: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    summary_by_task = {row["task_id"]: row for row in task_summaries}
    rows = []
    for offset, task_id in enumerate(task_ids):
        summary = summary_by_task[task_id]
        rows.append(
            {
                "task_id": task_id,
                "classes": summary["classes"],
                "accuracy": summary["accuracy"],
                "mean_loss": summary["mean_loss"],
                "mean_seen_accuracy": mean_seen_accuracy[offset],
                "mean_forgetting": mean_forgetting[offset],
                "support": support_sequence.get(task_id, []),
                "best_swap_gain": best_swap_gains.get(task_id, 0.0),
            }
        )
    return rows


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
        eval_results = trainer.evaluate_all_saved_supports()

        accuracy_matrix[task.task_id] = {
            eval_task_id: metrics["accuracy"]
            for eval_task_id, metrics in eval_results.items()
        }

        task_summaries.append(
            {
                "task_id": summary.task_id,
                "classes": list(summary.classes),
                "support_indices": list(summary.support_indices),
                "accuracy": summary.accuracy,
                "mean_loss": summary.mean_loss,
                "best_old_accuracy": summary.best_old_accuracy,
                "support_entropy": summary.support_entropy,
            }
        )

        artifact_root = trainer.export_task_artifacts(task.task_id)
        checkpoint_root = run_root / f"task_{task.task_id}" / "checkpoints"
        checkpoint_path = trainer.save_checkpoint(task.task_id, root=checkpoint_root)

        artifact_roots[task.task_id] = str(artifact_root)
        checkpoint_paths[task.task_id] = str(checkpoint_path)

        log_progress(
            f"task={task.task_id} train done accuracy={summary.accuracy:.4f} "
            f"support={summary.support_indices}",
            component="runner",
        )
        gc.collect()

    snapshot = trainer.snapshot()
    mean_seen_accuracy = _mean_seen_accuracy(accuracy_matrix)
    mean_forgetting = _mean_forgetting_curve(accuracy_matrix)
    support_sequence = _support_sequence(trainer)
    phi_trajectory = _phi_trajectory(trainer)
    best_swap_gains = _best_swap_gains(trainer)
    support_entropy_trajectory = _support_entropy_trajectory(task_summaries)
    task_ids = sorted(accuracy_matrix)
    task_metric_rows = _task_metric_rows(
        task_ids,
        mean_seen_accuracy,
        mean_forgetting,
        support_sequence,
        best_swap_gains,
        task_summaries,
    )

    export_run_artifacts(snapshot, run_root / "final")

    results = {
        "learning": learning,
        "accuracy_matrix": accuracy_matrix,
        "mean_seen_accuracy": mean_seen_accuracy,
        "mean_forgetting": mean_forgetting,
        "support_sequence": support_sequence,
        "phi_trajectory": phi_trajectory,
        "best_swap_gains": best_swap_gains,
        "support_entropy_trajectory": support_entropy_trajectory,
        "task_summaries": task_summaries,
        "artifact_roots": artifact_roots,
        "checkpoint_paths": checkpoint_paths,
    }

    _write_json(run_root / "run_summary.json", results)
    _write_csv(run_root / "task_summary.csv", task_metric_rows)

    return results


# Plot generation
def generate_plots(results: dict, plots_dir: Path) -> None:
    """Generate accuracy, support, and swap-gain figures."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_forgetting(
        results["mean_seen_accuracy"],
        results["mean_forgetting"],
        plots_dir / "figure3_accuracy_forgetting.png",
    )
    plot_support_table(
        results["support_sequence"],
        plots_dir / "chosen_support_table.png",
    )
    plot_swap_gains(
        results["best_swap_gains"],
        plots_dir / "one_swap_gains.png",
    )

    log_progress(f"plots saved to {plots_dir}", component="runner")



## Main entry point
def run_experiment(
    cfg,
    *,
    learning: str = LEARNING,
    tasks_limit: int | None = TASKS_LIMIT,
    plots_dir: str | None = None,
    run_id: str | None = None,
):
    if learning not in ("pc", "backprop"):
        raise ValueError("learning must be 'pc' or 'backprop'")

    print("JAX backend:", jax.default_backend(), flush=True)
    print("JAX devices:", jax.devices(), flush=True)
    print("starting HiBaCaML experiment.")

    tasks = build_split_mnist_tasks(cfg, limit=tasks_limit)

    log_progress(f"loaded {len(tasks)} Split-MNIST tasks", component="runner")

    if run_id is None:
        from datetime import datetime, timezone
        run_id = f"SMNIST_{learning}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_output_root = cfg.experiment_root_path() / run_id
    if run_output_root.exists():
        raise FileExistsError(
            f"run directory already exists: {run_output_root} "
            "(choose a different run_id, or remove the existing directory)"
        )
    run_output_root.mkdir(parents=True)
    run_plots_dir = Path(plots_dir) if plots_dir else run_output_root / "plots"

    selector_state_root = cfg.reporting.selector_state_root or str(run_output_root / "selector_state")
    cfg = dataclasses.replace(
        cfg,
        reporting=dataclasses.replace(
            cfg.reporting,
            experiment_root=str(run_output_root),
            selector_state_root=selector_state_root,
        ),
    )

    structure, trainer = build_trainer(cfg, tasks, learning, run_id=run_id)
    print_pre_run_review(cfg, structure, tasks, learning)

    results = _run_tasks(
        trainer,
        tasks,
        learning,
        run_output_root,
    )
    generate_plots(results, run_plots_dir)

    print("\nRun complete.")
    print(f"  Run ID  : {run_id}")
    print(f"  Run dir : {run_output_root}")
    print(f"  Summary : {run_output_root / 'run_summary.json'}")
    print(f"  Table   : {run_output_root / 'task_summary.csv'}")
    print(f"  Plots   : {run_plots_dir}")
    print(f"  Selector: {cfg.selector_state_path()}")

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
