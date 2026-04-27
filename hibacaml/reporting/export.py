"""Artifact export helpers for HiBaCaML experiment snapshots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    try:
        import jax.numpy as jnp

        if isinstance(value, jnp.ndarray):
            return value.tolist()
    except Exception:
        pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return value
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))


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


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_jsonable(row.get(k)) for k in fieldnames})


def export_run_artifacts(snapshot: Dict[str, object], root: str | Path) -> None:
    """Export the HiBaCaML artifact bundle."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    _write_json(
        root / "run_metadata.json",
        {
            "cfg": snapshot["cfg"],
            "global_step": snapshot["global_step"],
            "params_revision": snapshot["params_revision"],
        },
    )
    _write_json(root / "task_evaluations.json", snapshot["evaluations"])
    _write_json(
        root / "support_provenance.json",
        {
            "current_support": snapshot["current_support"],
            "boundary_support": snapshot["boundary_support"],
            "task_support_snapshots": snapshot["task_support_snapshots"],
        },
    )
    _write_json(root / "support_sequence.json", snapshot["support_sequence"])
    _write_json(root / "phi_trajectory.json", snapshot["phi_trajectory"])
    _write_json(root / "timing_summary.json", snapshot["timing_summaries"])
    _write_json(root / "support_diagnostics.json", snapshot["support_diagnostics"])
    _write_json(root / "composer_diagnostics.json", snapshot.get("composer_diagnostics", {}))
    _write_json(root / "task_support_snapshots.json", snapshot["task_support_snapshots"])
    _write_csv(root / "exact_support_search.csv", _rows_from_mapping(snapshot["support_tables"]))
    _write_csv(root / "controller_search.csv", _rows_from_mapping(snapshot["controller_tables"]))
    _write_csv(root / "local_swap_audit.csv", _rows_from_mapping(snapshot["local_swap_tables"]))
    _write_csv(root / "column_certificates.csv", _rows_from_mapping(snapshot["certificates"]))
    _write_csv(root / "composer_diagnostics.csv", _rows_from_mapping(snapshot.get("composer_diagnostics", {})))
    _write_csv(
        root / "support_overlap.csv",
        snapshot["support_diagnostics"].get("pairwise_jaccard", []),
    )
