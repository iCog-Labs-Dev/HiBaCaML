"""Small progress logging helpers for HiBaCaML debugging."""

from __future__ import annotations

from pathlib import Path
import os
import platform
import threading
import time

_START_TIME = time.perf_counter()
_LOG_LOCK = threading.Lock()
_LOG_HANDLE = None
_LOG_PATH_ENV = "FABRICPC_HIBACAML_LOG_PATH"


def _rss_mb() -> float | None:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None

    # Linux reports KiB; macOS/BSD report bytes.
    if platform.system().lower() == "linux":
        return rss / 1024.0
    return rss / (1024.0 * 1024.0)


def _default_log_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "hibacaml_debug.log"


def get_log_path() -> Path:
    """Return the resolved HiBaCaML debug log path."""
    override = os.environ.get(_LOG_PATH_ENV)
    return Path(override).expanduser() if override else _default_log_path()


def _ensure_log_handle():
    global _LOG_HANDLE
    if _LOG_HANDLE is not None and not _LOG_HANDLE.closed:
        return _LOG_HANDLE

    path = get_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_HANDLE = path.open("a", encoding="utf-8", buffering=1)
    return _LOG_HANDLE


def log_progress(message: str, *, component: str = "hibacaml") -> None:
    """Emit a flushed progress line to stdout and the debug log file."""
    elapsed = time.perf_counter() - _START_TIME
    peak_rss = _rss_mb()
    prefix = f"[{component} t+{elapsed:7.1f}s"
    if peak_rss is not None:
        prefix += f" rss={peak_rss:8.1f}MB"
    prefix += "]"
    line = f"{prefix} {message}"
    print(line, flush=True)

    try:
        with _LOG_LOCK:
            handle = _ensure_log_handle()
            handle.write(line + "\n")
            handle.flush()
    except Exception:
        # Logging should never interfere with the main execution path.
        pass
