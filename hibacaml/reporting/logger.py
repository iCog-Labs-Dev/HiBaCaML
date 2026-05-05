"""Structured run logging for HiBaCaML experiments."""

from __future__ import annotations

import csv
import json
import platform
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hibacaml.reporting.export import _to_jsonable

try:
    import resource
except Exception:  # pragma: no cover - non-Unix fallback.
    resource = None


def rss_mb() -> float | None:
    """Return peak resident set size in MiB with platform-correct units."""
    if resource is None:
        return None
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if platform.system().lower() == "linux":
        return value / 1024.0
    return value / (1024.0 * 1024.0)


class HiBaCaMLRunLogger:
    """Small JSONL/JSON/CSV logger for HiBaCaML experiments."""

    def __init__(self, cfg, root: Optional[Path] = None):
        self.cfg = cfg
        self.root = Path(root) if root is not None else Path(cfg.reporting.experiment_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()
        self._lock = threading.Lock()
        self._event_path = self.root / "events.jsonl"
        self._memory_path = self.root / "memory_samples.csv"
        self._phase_timings: Dict[str, float] = {}
        self._task_summaries: Dict[str, Dict[str, Any]] = {}
        self._progress: Dict[str, Any] = {
            "mode": cfg.mode,
            "global_step": 0,
            "phase": "created",
        }
        if not self._memory_path.exists():
            with self._memory_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["wall_time", "elapsed_seconds", "phase", "task_id", "step", "rss_mb"],
                )
                writer.writeheader()

    def event(self, phase: str, payload: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        record = {
            "wall_time": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
            "phase": phase,
            "rss_mb": rss_mb(),
        }
        if payload:
            record.update(payload)
        record.update(extra)
        with self._lock:
            with self._event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_to_jsonable(record), sort_keys=True) + "\n")
        self.heartbeat(phase=phase, global_step=record.get("global_step"))

    def progress(self, **payload: Any) -> None:
        self._progress.update(payload)
        self._write_json(self.root / self.cfg.reporting.progress_filename, self._progress)

    def heartbeat(self, **payload: Any) -> None:
        heartbeat = {
            "wall_time": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
            "rss_mb": rss_mb(),
            **payload,
        }
        self._write_json(self.root / self.cfg.reporting.heartbeat_filename, heartbeat)

    def memory_sample(
        self,
        *,
        phase: str,
        task_id: int | None = None,
        step: int | None = None,
    ) -> None:
        row = {
            "wall_time": time.time(),
            "elapsed_seconds": time.time() - self.started_at,
            "phase": phase,
            "task_id": task_id,
            "step": step,
            "rss_mb": rss_mb(),
        }
        with self._lock:
            with self._memory_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["wall_time", "elapsed_seconds", "phase", "task_id", "step", "rss_mb"],
                )
                writer.writerow(_to_jsonable(row))

    def phase_timing(self, phase: str, seconds: float) -> None:
        self._phase_timings[phase] = self._phase_timings.get(phase, 0.0) + float(seconds)
        self._write_json(
            self.root / "phase_timings.json",
            self._phase_timings,
        )

    def task_summary(self, task_id: int, summary: Dict[str, Any]) -> None:
        self._task_summaries[str(task_id)] = summary
        self._write_json(
            self.root / "task_summaries.json",
            self._task_summaries,
        )

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            path.write_text(
                json.dumps(_to_jsonable(payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
