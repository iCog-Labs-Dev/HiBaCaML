"""Split-MNIST experiment runner for HiBaCaML."""

from __future__ import annotations

# ------------------------------------------------------------
# Script inputs: edit these directly, or import and call
# run_experiment_notebook(...) from another script.
# ------------------------------------------------------------
MODE = "paper_faithful"               # "paper_faithful", "full", "smoke"
TASKS = None                           # e.g. 2, 5, or None for all
SHORTLIST = None                       # e.g. 8 or None
EPOCHS = None                          # e.g. 1, 5, or None
INFER_STEPS = None                     # e.g. 20 or None
ROLLOUT_STEPS = None                   # e.g. 4 or None
STATIC_SUPPORT_BATCH_SIZE = 16       # e.g. 2 or None
NEIGHBOR_SUPPORT_BATCH_SIZE = 16     # e.g. 32 or None
CURRENT_BOUNDARY_BATCHES = None        # e.g. 2 or None
OLD_BOUNDARY_BATCHES = None            # e.g. 2 or None
TRAIN_BATCHES_LIMIT = None             # e.g. 10 or None
TEST_BATCHES_LIMIT = None              # e.g. 10 or None
EXACT_SEARCH = None                    # True / False / None; leave None for config default
LEARNING = "backprop"                  # "pc" or "backprop"
CONFIRM = False                        # False is best for non-interactive runs
CPU = False                            # True => force JAX CPU backend
PLOTS_DIR = None                       
EXPERIMENT_ROOT = None               
SELECTOR_STATE_ROOT = None             
RUN_ID = None                         

# Resume controls
RESUME_FROM_TASK = None                # e.g. 4 — next task to train; loads task_(N-1)/checkpoints/...
RESUME_RUN_ID = None                   # optional explicit run_id; auto-detected if None

# Memory-related inputs
LOW_MEMORY_BATCH_SIZE_CAP = None       # e.g. 64, or None to disable
BATCH_SIZE = 768                        # direct batch_size override, e.g. 64


# ------------------------------------------------------------
# Early environment setup: must happen before `import jax`
# ------------------------------------------------------------
import csv
import dataclasses
import gc
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

if "." not in sys.path:
    sys.path.insert(0, ".")

importlib.invalidate_caches()

if CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax(
    jax_platforms=os.environ.get("JAX_PLATFORMS", "cuda")
)

import jax

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(".mplconfig")
    mpl_cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache.resolve())


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit
from hibacaml import (
    HiBaCaMLBackpropRunner,
    HiBaCaMLTrainer,
    build_split_mnist_tasks,
    create_hibacaml_structure,
    export_run_artifacts,
    make_hibacaml_config,
)
from hibacaml.debug import log_progress
from hibacaml.reporting import (
    plot_accuracy_forgetting,
    plot_support_table,
    plot_swap_gains,
    print_pre_run_review,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Resume helpers
# ------------------------------------------------------------
def _load_task_evaluations(run_root: Path, task_id: int) -> Dict[int, Dict[str, float]]:
    path = run_root / f"task_{task_id}" / "task_evaluations.json"
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def _reconstruct_task_summary(task, evaluations: Dict[int, Dict[str, float]], snapshot) -> Dict[str, object]:
    own = evaluations[task.task_id]
    return {
        "task_id": task.task_id,
        "classes": list(task.classes),
        "support_indices": list(snapshot.full_support),
        "accuracy": own["accuracy"],
        "mean_loss": own["mean_loss"],
        "best_old_accuracy": own["best_old_accuracy"],
        "support_entropy": own["support_entropy"],
    }


def _infer_resume_run_id(experiment_root: Path, prev_task_id: int, checkpoint_filename: str) -> str:
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"experiment_root not found: {experiment_root}")
    candidates = [
        d for d in experiment_root.iterdir()
        if d.is_dir() and (d / f"task_{prev_task_id}" / "checkpoints" / checkpoint_filename).is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run directory under {experiment_root} contains "
            f"task_{prev_task_id}/checkpoints/{checkpoint_filename}"
        )
    if len(candidates) > 1:
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        log_progress(
            f"multiple resume candidates found; picking most recent: {candidates[0].name}",
            component="runner",
        )
    return candidates[0].name


def _resolve_resume_checkpoint(cfg, resume_from_task: int, explicit_run_id: str | None) -> tuple[str, Path]:
    prev_task_id = resume_from_task - 1
    if prev_task_id < 0:
        raise ValueError(f"resume_from_task must be >= 1, got {resume_from_task}")
    experiment_root = cfg.experiment_root_path()
    checkpoint_filename = cfg.reporting.checkpoint_filename
    run_id = explicit_run_id or _infer_resume_run_id(experiment_root, prev_task_id, checkpoint_filename)
    checkpoint_path = experiment_root / run_id / f"task_{prev_task_id}" / "checkpoints" / checkpoint_filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return run_id, checkpoint_path


# ------------------------------------------------------------
# Trainer construction
# ------------------------------------------------------------
def build_trainer(cfg, tasks, learning: str, *, run_id: str = ""):
    """Construct the HiBaCaML trainer for one learning mode."""
    inference = InferenceSGD(eta_infer=cfg.eta_infer, infer_steps=cfg.infer_steps)
    graph_state_initializer = FeedforwardStateInit() if learning == "backprop" else None
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


# ------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Core experiment loop
# ------------------------------------------------------------
def run_experiment(
    cfg,
    tasks,
    learning: str,
    run_root: Path,
    *,
    run_id: str = "",
    resume_from: str | Path | None = None,
) -> dict:
    """Run the HiBaCaML experiment for one learning mode."""
    trainer_cfg = dataclasses.replace(
        cfg,
        reporting=dataclasses.replace(
            cfg.reporting,
            experiment_root=str(run_root),
        ),
    )
    _, trainer = build_trainer(trainer_cfg, tasks, learning, run_id=run_id)

    accuracy_matrix: Dict[int, Dict[int, float]] = {}
    task_summaries: List[Dict[str, object]] = []
    artifact_roots: Dict[int, str] = {}
    checkpoint_paths: Dict[int, str] = {}
    completed_ids: set = set()

    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
        snapshots = trainer.persistent_state.task_support_snapshots
        completed_ids = set(snapshots.keys())
        if snapshots:
            # load_checkpoint restores persistent_state/params/opt_state but leaves
            # current_phi/current_nonshared at their constructor defaults. The next
            # boundary_search builds phi_candidates() around trainer.current_phi, so
            # without this the resumed controller would search a different neighborhood
            # than an uninterrupted run.
            last = snapshots[max(snapshots)]
            trainer.current_phi = last.phi
            trainer.current_nonshared = tuple(last.nonshared)
            log_progress(
                f"restored controller state from task={max(snapshots)} "
                f"nonshared={trainer.current_nonshared}",
                component="runner",
            )
        log_progress(
            f"resumed from {resume_from}; completed tasks={sorted(completed_ids)}",
            component="runner",
        )
        for task in tasks:
            if task.task_id not in completed_ids:
                continue
            evals = _load_task_evaluations(run_root, task.task_id)
            accuracy_matrix[task.task_id] = {
                eval_id: metrics["accuracy"] for eval_id, metrics in evals.items()
            }
            task_summaries.append(
                _reconstruct_task_summary(
                    task,
                    evals,
                    trainer.persistent_state.task_support_snapshots[task.task_id],
                )
            )
            artifact_roots[task.task_id] = str(run_root / f"task_{task.task_id}")
            checkpoint_paths[task.task_id] = str(
                run_root / f"task_{task.task_id}" / "checkpoints" / cfg.reporting.checkpoint_filename
            )

    for task in tasks:
        if task.task_id in completed_ids:
            continue
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


# ------------------------------------------------------------
# Config preparation
# ------------------------------------------------------------
def apply_notebook_overrides(
    cfg,
    *,
    shortlist: int | None = SHORTLIST,
    epochs: int | None = EPOCHS,
    infer_steps: int | None = INFER_STEPS,
    rollout_steps: int | None = ROLLOUT_STEPS,
    static_support_batch_size: int | None = STATIC_SUPPORT_BATCH_SIZE,
    neighbor_support_batch_size: int | None = NEIGHBOR_SUPPORT_BATCH_SIZE,
    current_boundary_batches: int | None = CURRENT_BOUNDARY_BATCHES,
    old_boundary_batches: int | None = OLD_BOUNDARY_BATCHES,
    train_batches_limit: int | None = TRAIN_BATCHES_LIMIT,
    test_batches_limit: int | None = TEST_BATCHES_LIMIT,
    exact_search: bool | None = EXACT_SEARCH,
    experiment_root: str | None = EXPERIMENT_ROOT,
    selector_state_root: str | None = SELECTOR_STATE_ROOT,
    low_memory_batch_size_cap: int | None = LOW_MEMORY_BATCH_SIZE_CAP,
    batch_size: int | None = BATCH_SIZE,
):
    """Apply script overrides in one place."""
    if shortlist is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                boundary_shortlist=shortlist,
            ),
        )

    if current_boundary_batches is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                boundary_current_batches=current_boundary_batches,
            ),
        )

    if old_boundary_batches is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                boundary_old_batches=old_boundary_batches,
            ),
        )

    if rollout_steps is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                boundary_rollout_steps=rollout_steps,
            ),
        )

    if static_support_batch_size is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                static_support_batch_size=static_support_batch_size,
            ),
        )

    if neighbor_support_batch_size is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                neighbor_support_batch_size=neighbor_support_batch_size,
            ),
        )

    if exact_search is not None:
        cfg = dataclasses.replace(
            cfg,
            exact_search=dataclasses.replace(
                cfg.exact_search,
                enable_exact_search=exact_search,
            ),
        )

    if epochs is not None:
        cfg = dataclasses.replace(cfg, epochs_per_task=epochs)

    if infer_steps is not None:
        cfg = dataclasses.replace(cfg, infer_steps=infer_steps)

    if train_batches_limit is not None:
        cfg = dataclasses.replace(cfg, train_batches_limit=train_batches_limit)

    if test_batches_limit is not None:
        cfg = dataclasses.replace(cfg, test_batches_limit=test_batches_limit)

    if experiment_root is not None:
        cfg = dataclasses.replace(
            cfg,
            reporting=dataclasses.replace(cfg.reporting, experiment_root=experiment_root),
        )

    if selector_state_root is not None:
        cfg = dataclasses.replace(
            cfg,
            reporting=dataclasses.replace(cfg.reporting, selector_state_root=selector_state_root),
        )

    if low_memory_batch_size_cap is not None:
        cfg = dataclasses.replace(
            cfg,
            batch_size=min(cfg.batch_size, low_memory_batch_size_cap),
        )
        log_progress(
            f"low-memory batch cap active: batch_size <= {low_memory_batch_size_cap}",
            component="runner",
        )

    if batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=batch_size)
        log_progress(f"batch_size override: {batch_size}", component="runner")

    return cfg


# ------------------------------------------------------------
# Plot generation
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def run_experiment_notebook(
    mode: str = MODE,
    tasks_limit: int | None = TASKS,
    learning: str = LEARNING,
    confirm: bool = CONFIRM,
    plots_dir: str | None = PLOTS_DIR,
    experiment_root: str | None = EXPERIMENT_ROOT,
    selector_state_root: str | None = SELECTOR_STATE_ROOT,
    run_id: str | None = RUN_ID,
    shortlist: int | None = SHORTLIST,
    epochs: int | None = EPOCHS,
    infer_steps: int | None = INFER_STEPS,
    rollout_steps: int | None = ROLLOUT_STEPS,
    static_support_batch_size: int | None = STATIC_SUPPORT_BATCH_SIZE,
    neighbor_support_batch_size: int | None = NEIGHBOR_SUPPORT_BATCH_SIZE,
    current_boundary_batches: int | None = CURRENT_BOUNDARY_BATCHES,
    old_boundary_batches: int | None = OLD_BOUNDARY_BATCHES,
    train_batches_limit: int | None = TRAIN_BATCHES_LIMIT,
    test_batches_limit: int | None = TEST_BATCHES_LIMIT,
    exact_search: bool | None = EXACT_SEARCH,
    low_memory_batch_size_cap: int | None = LOW_MEMORY_BATCH_SIZE_CAP,
    batch_size: int | None = BATCH_SIZE,
    resume_from_task: int | None = RESUME_FROM_TASK,
    resume_run_id: str | None = RESUME_RUN_ID,
):
    if learning not in ("pc", "backprop"):
        raise ValueError("learning must be 'pc' or 'backprop'")
    if mode not in ("paper_faithful", "smoke"):
        raise ValueError("mode must be 'paper_faithful' or 'smoke'")

    print("JAX backend:", jax.default_backend(), flush=True)
    print("JAX devices:", jax.devices(), flush=True)
    print("starting HiBaCaML experiment.")

    cfg = make_hibacaml_config(mode)
    cfg = apply_notebook_overrides(
        cfg,
        shortlist=shortlist,
        epochs=epochs,
        infer_steps=infer_steps,
        rollout_steps=rollout_steps,
        static_support_batch_size=static_support_batch_size,
        neighbor_support_batch_size=neighbor_support_batch_size,
        current_boundary_batches=current_boundary_batches,
        old_boundary_batches=old_boundary_batches,
        train_batches_limit=train_batches_limit,
        test_batches_limit=test_batches_limit,
        exact_search=exact_search,
        experiment_root=experiment_root,
        selector_state_root=selector_state_root,
        low_memory_batch_size_cap=low_memory_batch_size_cap,
        batch_size=batch_size,
    )

    tasks = build_split_mnist_tasks(cfg)
    if tasks_limit is not None:
        tasks = tasks[:tasks_limit]

    log_progress(f"loaded {len(tasks)} Split-MNIST tasks", component="runner")

    review_inference = InferenceSGD(
        eta_infer=cfg.eta_infer,
        infer_steps=cfg.infer_steps,
    )
    review_structure = create_hibacaml_structure(cfg, review_inference)

    confirm_flag = bool(confirm) and sys.stdin.isatty()
    print_pre_run_review(
        cfg,
        review_structure,
        tasks,
        learning,
        confirm=confirm_flag,
    )

    del review_structure
    gc.collect()

    resume_checkpoint_path: Path | None = None
    if resume_from_task is not None:
        resolved_run_id, resume_checkpoint_path = _resolve_resume_checkpoint(
            cfg, resume_from_task, resume_run_id
        )
        run_id = resolved_run_id
        log_progress(
            f"resume mode: run_id={run_id} checkpoint={resume_checkpoint_path}",
            component="runner",
        )
    elif run_id is None:
        from datetime import datetime, timezone
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_seed{cfg.seed}"
    run_output_root = cfg.experiment_root_path() / run_id
    run_output_root.mkdir(parents=True, exist_ok=True)
    run_plots_dir = Path(plots_dir) if plots_dir else run_output_root / "plots"

    results = run_experiment(
        cfg,
        tasks,
        learning,
        run_output_root,
        run_id=run_id,
        resume_from=resume_checkpoint_path,
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


# ------------------------------------------------------------
# Execute only when run as a script
# ------------------------------------------------------------
if __name__ == "__main__":
    run_output = run_experiment_notebook(
        mode=MODE,
        tasks_limit=TASKS,
        learning=LEARNING,
        confirm=CONFIRM,
        plots_dir=PLOTS_DIR,
        experiment_root=EXPERIMENT_ROOT,
        selector_state_root=SELECTOR_STATE_ROOT,
        run_id=RUN_ID,
        shortlist=SHORTLIST,
        epochs=EPOCHS,
        infer_steps=INFER_STEPS,
        rollout_steps=ROLLOUT_STEPS,
        static_support_batch_size=STATIC_SUPPORT_BATCH_SIZE,
        neighbor_support_batch_size=NEIGHBOR_SUPPORT_BATCH_SIZE,
        current_boundary_batches=CURRENT_BOUNDARY_BATCHES,
        old_boundary_batches=OLD_BOUNDARY_BATCHES,
        train_batches_limit=TRAIN_BATCHES_LIMIT,
        test_batches_limit=TEST_BATCHES_LIMIT,
        exact_search=EXACT_SEARCH,
        low_memory_batch_size_cap=LOW_MEMORY_BATCH_SIZE_CAP,
        batch_size=BATCH_SIZE,
        resume_from_task=RESUME_FROM_TASK,
        resume_run_id=RESUME_RUN_ID,
    )
