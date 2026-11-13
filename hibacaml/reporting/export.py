"""Artifact writers for HiBaCaML runs.

Covers both the final bundle (`export_run_artifacts`) and the two artifacts
written incrementally while a run is in progress: the `events.jsonl` stream and
the current-state document.
"""

from __future__ import annotations

import csv
import json
import pickle
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import jax.numpy as jnp

from hibacaml.reporting.logger import log_progress, rss_mb

_RUN_STARTED_AT = time.time()
_RUN_STATE: Dict[str, Any] = {}


def start_run(root: str | Path, /, **fields: Any) -> None:
    """Begin a run: reset the elapsed clock and the sticky state.

    Without this a second experiment in the same process would inherit the
    first one's clock and its last-known task.
    """
    global _RUN_STARTED_AT
    _RUN_STARTED_AT = time.time()
    _RUN_STATE.clear()
    Path(root).mkdir(parents=True, exist_ok=True)
    write_run_state(root, **fields)


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, mappings, and arrays into JSON-serializable values."""
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, jnp.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def _rows_from_mapping(mapping: Dict[Any, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, value in mapping.items():
        base = {"key": key}
        if isinstance(value, list):
            for item in value:
                if is_dataclass(item):
                    rows.append({**base, **asdict(item)})
                elif isinstance(item, dict):
                    rows.append({**base, **item})
                else:
                    rows.append({**base, "value": item})
        elif is_dataclass(value):
            rows.append({**base, **asdict(value)})
        elif isinstance(value, dict):
            rows.append({**base, **value})
        else:
            rows.append({**base, "value": value})
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(row.get(k)) for k in fieldnames})

def append_event(root: str | Path, phase: str, /, **fields: Any) -> None:
    """Append one structured record to the run's `events.jsonl`.

    `root` and `phase` are positional-only so an event may carry fields of
    those names -- "export_start" legitimately records a `root`.

    Rollout clones pass no root, which is what keeps trial events out of the
    audit stream.
    """
    root = Path(root)
    record = {
        "wall_time": time.time(),
        "elapsed_seconds": time.time() - _RUN_STARTED_AT,
        "phase": phase,
        "rss_mb": rss_mb(),
        **fields,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
    write_run_state(root, phase=phase)


def write_run_state(root: str | Path, /, **fields: Any) -> None:
    """Rewrite the current-state document used to watch a run in flight."""
    _RUN_STATE.update({k: v for k, v in fields.items() if v is not None})
    write_json(
        Path(root) / "heartbeat.json",
        {
            "wall_time": time.time(),
            "elapsed_seconds": time.time() - _RUN_STARTED_AT,
            "rss_mb": rss_mb(),
            **_RUN_STATE,
        },
    )


def build_run_snapshot(trainer) -> Dict[str, object]:
    """Build a run snapshot, evaluating saved supports and advancing evaluation state."""
    return {
        "cfg": trainer.cfg.to_dict(),
        "current_support": trainer.persistent_state.current_support,
        "boundary_support": trainer.persistent_state.boundary_support,
        "support_tables": trainer.persistent_state.support_tables,
        "support_posterior_tables": trainer.persistent_state.support_posterior_tables,
        "reserve_recruitment_tables": (
            trainer.persistent_state.reserve_recruitment_tables
        ),
        "controller_tables": trainer.persistent_state.controller_tables,
        "local_swap_tables": trainer.persistent_state.local_swap_tables,
        "demotion_swap_tables": trainer.persistent_state.demotion_swap_tables,
        "replay_proposals": trainer.persistent_state.replay_proposals,
        "recently_demoted": trainer.persistent_state.recently_demoted,
        "certificates": trainer.persistent_state.certificates,
        "task_support_snapshots": trainer.persistent_state.task_support_snapshots,
        "support_sequence": {
            task_id: snapshot.full_support
            for task_id, snapshot in sorted(
                trainer.persistent_state.task_support_snapshots.items()
            )
        },
        "phi_trajectory": {
            task_id: snapshot.phi
            for task_id, snapshot in sorted(
                trainer.persistent_state.task_support_snapshots.items()
            )
        },
        # Reporting observes the run, so it never refreshes certificates.
        "evaluations": trainer.evaluate_all_saved_supports(refresh_certificates=False),
        "global_step": trainer.persistent_state.global_step,
        "params_revision": trainer.persistent_state.params_revision,
        "timing_summaries": trainer.persistent_state.timing_summaries,
        "composer_diagnostics": trainer.persistent_state.composer_diagnostics,
        "epoch_evaluations": trainer.persistent_state.epoch_evaluations,
        "selector_bank_summary": trainer.selector_bank.summary(),
        "run_id": trainer.run_id,
        "experiment_metadata": trainer.experiment_metadata,
        "evaluation_protocol": "target_free_inference_external_supervision_v1",
    }


def export_run_artifacts(snapshot: Dict[str, object], root: str | Path) -> None:
    """Export the HiBaCaML artifact bundle (V20.2b)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # 1. Define JSON exports
    json_exports = {
        "run_metadata.json": {
            "cfg": snapshot["cfg"],
            "global_step": snapshot["global_step"],
            "params_revision": snapshot["params_revision"],
            "run_id": snapshot.get("run_id"),
            "experiment_metadata": snapshot.get("experiment_metadata", {}),
            "evaluation_protocol": snapshot.get("evaluation_protocol"),
        },
        "support_provenance.json": {
            "current_support": snapshot["current_support"],
            "boundary_support": snapshot["boundary_support"],
            "task_support_snapshots": snapshot["task_support_snapshots"],
        },
        "task_evaluations.json": snapshot["evaluations"],
        "support_sequence.json": snapshot["support_sequence"],
        "phi_trajectory.json": snapshot["phi_trajectory"],
        "timing_summary.json": snapshot["timing_summaries"],
        "epoch_evaluations.json": snapshot.get("epoch_evaluations", []),
        "recently_demoted.json": snapshot.get("recently_demoted", {}),
        "selector_bank_summary.json": snapshot.get("selector_bank_summary", {}),
    }

    # 2. Define CSV exports
    csv_exports = {
        "support_posterior.csv": _rows_from_mapping(snapshot.get("support_posterior_tables", {})),
        "reserve_recruitment.csv": _rows_from_mapping(snapshot.get("reserve_recruitment_tables", {})),
        "exact_support_search.csv": _rows_from_mapping(snapshot["support_tables"]),
        "controller_search.csv": _rows_from_mapping(snapshot["controller_tables"]),
        "local_swap_audit.csv": _rows_from_mapping(snapshot["local_swap_tables"]),
        "demotion_swap_audit.csv": _rows_from_mapping(snapshot.get("demotion_swap_tables", {})),
        "replay_proposals.csv": _rows_from_mapping(snapshot.get("replay_proposals", {})),
        "column_certificates.csv": _rows_from_mapping(snapshot["certificates"]),
        "composer_diagnostics.csv": _rows_from_mapping(snapshot.get("composer_diagnostics", {})),
    }

    # 3. Write them to disk
    for filename, payload in json_exports.items():
        write_json(root / filename, payload)

    for filename, payload in csv_exports.items():
        write_csv(root / filename, payload)


def export_task_artifacts(
    trainer,
    task_id: int,
    root: str | Path | None = None,
) -> Path:
    """Build and export one task's run artifact bundle."""
    run_root = (
        Path(root)
        if root is not None
        else trainer.cfg.experiment_root_path() / f"task_{task_id}"
    )
    log_progress(f"export task={task_id} start root={run_root}", component="report")
    if trainer.run_root is not None:
        append_event(
            trainer.run_root,
            "export_start",
            task_id=task_id,
            root=str(run_root),
        )
    run_root.mkdir(parents=True, exist_ok=True)
    export_run_artifacts(build_run_snapshot(trainer), run_root)
    log_progress(f"export task={task_id} done root={run_root}", component="report")
    if trainer.run_root is not None:
        append_event(
            trainer.run_root,
            "export_done",
            task_id=task_id,
            root=str(run_root),
        )
    return run_root


def save_checkpoint(
    trainer,
    task_id: int,
    root: str | Path | None = None,
) -> Path:
    """Serialize the trainer's current checkpoint payload."""
    checkpoint_root = (
        Path(root) if root is not None else trainer.cfg.experiment_root_path()
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    path = checkpoint_root / trainer.cfg.reporting.checkpoint_filename
    payload = {
        "task_id": task_id,
        "persistent_state": trainer.persistent_state,
        "params": trainer.params,
        "opt_state": trainer.opt_state,
        "experiment_metadata": trainer.experiment_metadata,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    if trainer.run_root is not None:
        append_event(
            trainer.run_root,
            "checkpoint_saved",
            task_id=task_id,
            path=str(path),
        )
    return path


# --------------------------------------------------------------------------
# Run records shared by the MNIST and CIFAR runners
# --------------------------------------------------------------------------

def mean_seen_accuracy(accuracy_matrix: Dict[int, Dict[int, float]]) -> List[float]:
    curve: List[float] = []
    for task_id in sorted(accuracy_matrix):
        row = accuracy_matrix[task_id]
        curve.append(sum(row.values()) / max(len(row), 1))
    return curve


def mean_forgetting(accuracy_matrix: Dict[int, Dict[int, float]]) -> List[float]:
    """Best-so-far minus current accuracy, averaged over every seen task (the current one adds 0)."""
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


def continual_extras(accuracy_matrix: Dict[int, Dict[int, float]]) -> Dict[str, List[float]]:
    """Per boundary b: acquisition A[b][b], BWT over old tasks, and forgetting averaged over old tasks only."""
    acquisition, bwt, old_forgetting = [], [], []
    for b in sorted(accuracy_matrix):
        old = [j for j in sorted(accuracy_matrix[b]) if j < b]
        acquisition.append(accuracy_matrix[b][b])
        bwt.append(sum(accuracy_matrix[b][j] - accuracy_matrix[j][j] for j in old) / len(old) if old else 0.0)
        drops = [
            max(accuracy_matrix[p][j] for p in range(j, b + 1)) - accuracy_matrix[b][j] for j in old
        ]
        old_forgetting.append(sum(drops) / len(drops) if drops else 0.0)
    return {"acquisition": acquisition, "bwt": bwt, "mean_old_forgetting": old_forgetting}


def support_sequence(trainer) -> Dict[int, List[int]]:
    return {
        task_id: list(snapshot.full_support)
        for task_id, snapshot in sorted(trainer.persistent_state.task_support_snapshots.items())
    }


def phi_trajectory(trainer) -> Dict[int, Dict[str, float]]:
    return {
        task_id: {
            "outer_quantile": snapshot.phi.outer_quantile,
            "middle_quantile": snapshot.phi.middle_quantile,
            "replacement_margin_base": snapshot.phi.replacement_margin_base,
            "demotion_min_role_gain": snapshot.phi.demotion_min_role_gain,
        }
        for task_id, snapshot in sorted(trainer.persistent_state.task_support_snapshots.items())
    }


def support_entropy_trajectory(task_summaries: List[Dict[str, object]]) -> List[float]:
    """Per-task support-usage entropy trajectory using the trainer's own natural-log metric."""
    return [
        float(row["support_entropy"])
        for row in sorted(task_summaries, key=lambda r: r["task_id"])
        if "support_entropy" in row
    ]


def best_swap_gains(trainer) -> Dict[int, float]:
    gains = {}
    for task_id, rows in sorted(trainer.persistent_state.local_swap_tables.items()):
        gains[task_id] = max((row.gain for row in rows), default=0.0)
    return gains


def task_summary_row(summary) -> Dict[str, object]:
    """One sequential-run summary row from a `TaskSummary`."""
    return {
        "task_id": summary.task_id,
        "classes": list(summary.classes),
        "support_indices": list(summary.support_indices),
        "accuracy": summary.accuracy,
        "mean_loss": summary.mean_loss,
        "best_old_accuracy": summary.best_old_accuracy,
        "support_entropy": summary.support_entropy,
    }


def task_metric_rows(
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


def static_task_row(summary, test_metrics, validation_metrics, completed_updates: int) -> Dict[str, object]:
    """The single row of a static (one-task) run's task summary."""
    return {
        "task_id": summary.task_id,
        "classes": list(summary.classes),
        "support": list(summary.support_indices),
        "test_accuracy": test_metrics["accuracy"],
        "test_cross_entropy": test_metrics["cross_entropy"],
        "test_mean_loss": test_metrics["mean_loss"],
        "validation_accuracy": validation_metrics.get("accuracy"),
        "validation_cross_entropy": validation_metrics.get("cross_entropy"),
        "completed_updates": completed_updates,
    }


def write_run_records(root: Path, results: Dict[str, Any], rows: Iterable[Dict[str, Any]], **tables) -> None:
    """`run_summary.json`, `task_summary.csv`, then one `<name>.csv` per extra table."""
    root = Path(root)
    write_json(root / "run_summary.json", results)
    write_csv(root / "task_summary.csv", rows)
    for name, table in tables.items():
        write_csv(root / f"{name}.csv", table)
