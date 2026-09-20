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
from typing import Any, Dict, Iterable, List

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


def build_run_snapshot(
    trainer,
    *,
    refresh_evaluation_certificates: bool = True,
) -> Dict[str, object]:
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
        "evaluations": trainer.evaluate_all_saved_supports(
            refresh_certificates=refresh_evaluation_certificates,
        ),
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
    *,
    refresh_evaluation_certificates: bool = True,
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
    export_run_artifacts(
        build_run_snapshot(
            trainer,
            refresh_evaluation_certificates=refresh_evaluation_certificates,
        ),
        run_root,
    )
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
