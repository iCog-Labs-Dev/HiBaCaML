"""Byte-exact state digests for protection, isolation and provenance checks."""

from __future__ import annotations

import dataclasses
import hashlib
from contextlib import contextmanager
from typing import Dict, Iterable

import jax
import numpy as np


def fingerprint(value) -> str:
    """SHA-256 over structure, shapes, dtypes and raw bytes (signed zeros included)."""
    digest = hashlib.sha256()

    def visit(v):
        if dataclasses.is_dataclass(v):
            visit({f.name: getattr(v, f.name) for f in dataclasses.fields(v)})
        elif isinstance(v, dict):
            digest.update(b"dict")
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(v[key])
        elif isinstance(v, (tuple, list)):
            digest.update(type(v).__name__.encode())
            for item in v:
                visit(item)
        elif isinstance(v, (np.ndarray, jax.Array, np.generic)):
            array = np.asarray(v)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        else:
            digest.update(repr(v).encode())

    visit(value)
    return digest.hexdigest()


_LEARNING_STATE_FIELDS = (
    "certificates", "shell_stats", "shell_roles", "global_step", "params_revision",
    "certificates_revision", "roles_revision", "current_support", "boundary_support",
    "task_support_snapshots",
)


_STREAM_FIELDS = ("passes", "seed", "shuffle", "reshuffle_each_pass", "drop_last", "batch_size", "max_batches")


def _stream(loader):
    """Everything that decides a loader's next batches, apart from its fixed data arrays."""
    augment = getattr(loader, "augment", None)
    return {
        **{name: getattr(loader, name, None) for name in _STREAM_FIELDS},
        "augment": None if augment is None else f"{augment.__module__}.{augment.__qualname__}",
    }


def learning_state_signature(trainer) -> str:
    """Digest of the state later backprop training consumes.

    Excluded: timings and reports; task data contents (fixed inputs, hashed in run
    evidence); and `_last_graph_state`, which backprop training never reads (its
    scheduled refreshes use the update's own graph state). Only certificate refreshes
    requested through `_build_clamps(refresh_certificates=True)` read that cache;
    use `resume_clone` when a continuation must keep it.
    """
    state = trainer.persistent_state
    return fingerprint({
        "params": trainer.params,
        "opt_state": trainer.opt_state,
        "rng_key": trainer.rng_key,
        "persistent": {name: getattr(state, name) for name in _LEARNING_STATE_FIELDS},
        "support": tuple(trainer.current_nonshared),
        "phi": trainer.current_phi,
        "write_policy": trainer.write_policy,
        "policy_inputs": trainer._policy_inputs,
        "loader_streams": {
            (task_id, split): _stream(task.loader(split))
            for task_id, task in sorted(trainer.tasks.items())
            for split in ("train", "validation", "test")
            if getattr(task, f"{split}_loader") is not None
        },
    })


def column_fingerprints(params, columns: Iterable[int]) -> Dict[int, str]:
    """`{column: digest}` over every leaf of every `col<N>/` node of the requested columns."""
    wanted = {int(c) for c in columns}
    grouped: Dict[int, dict] = {c: {} for c in wanted}
    for name, node in params.nodes.items():
        head = name.split("/")[0]
        if head.startswith("col") and int(head[3:]) in wanted:
            grouped[int(head[3:])][name] = node
    return {c: fingerprint(nodes) for c, nodes in grouped.items()}


@contextmanager
def preserve_reporting_state(trainer):
    """Restore the RNG key and cached graph state after observational evaluation."""
    rng_key = trainer.rng_key
    had_state = hasattr(trainer, "_last_graph_state")
    cached = getattr(trainer, "_last_graph_state", None)
    try:
        yield
    finally:
        trainer.rng_key = rng_key
        if had_state:
            trainer._last_graph_state = cached
        elif hasattr(trainer, "_last_graph_state"):
            del trainer._last_graph_state
