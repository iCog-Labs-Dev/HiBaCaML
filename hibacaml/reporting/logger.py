"""Human-readable progress logging for HiBaCaML."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import platform
import sys
import time
from pathlib import Path

try:
    import resource
except Exception:
    resource = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_START_TIME = time.perf_counter()
_LOGGER_NAME = "hibacaml"
_FORMAT = "[%(component)s t+%(elapsed)7.1fs%(rss)s] %(message)s"
_ROLLOUT_DEPTH = contextvars.ContextVar("hibacaml_rollout_depth", default=0)
_configured = False

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


@contextlib.contextmanager
def rollout_logging():
    """Demote progress lines to DEBUG while a rollout clone is running.

    Rollout trials re-run training on a clone, so their progress lines are
    repetitive and vastly outnumber the parent's. They stay recoverable with
    `logging.getLogger("hibacaml").setLevel(logging.DEBUG)`.
    """
    token = _ROLLOUT_DEPTH.set(_ROLLOUT_DEPTH.get() + 1)
    try:
        yield
    finally:
        # Reset by token so an exception cannot leave the level demoted.
        _ROLLOUT_DEPTH.reset(token)


class _ContextFilter(logging.Filter):
    """Attach elapsed time, peak RSS, and the short component to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.elapsed = time.perf_counter() - _START_TIME
        value = rss_mb()
        record.rss = f" rss={value:8.1f}MB" if value is not None else ""
        record.component = record.name.rpartition(".")[2]
        return True


def _configure() -> None:
    """Attach handlers to the `hibacaml` logger once. Never touches the root."""
    global _configured
    if _configured:
        return
    _configured = True

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        # An application configured its own logging; it owns level and sinks.
        return

    logger.setLevel(logging.INFO)
    logging.raiseExceptions = False  # logging must never interrupt a run

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        path: Path = _REPO_ROOT / "hibacaml_debug.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8", delay=True))
    except OSError:
        pass

    formatter = logging.Formatter(_FORMAT)
    context = _ContextFilter()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context)
        logger.addHandler(handler)
    # Keep records inside the hibacaml tree so an application's own root
    # configuration cannot print them a second time.
    logger.propagate = False


def log_progress(message: str, *, component: str = _LOGGER_NAME) -> None:
    """Emit a progress line to stdout and the debug log file."""
    _configure()
    name = (
        _LOGGER_NAME
        if component == _LOGGER_NAME
        else f"{_LOGGER_NAME}.{component}"
    )
    level = logging.DEBUG if _ROLLOUT_DEPTH.get() else logging.INFO
    logging.getLogger(name).log(level, message)
