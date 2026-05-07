"""V20.2b cross-run replay bank for support proposal.

The bank persists `(context, support, observed_quality)` rows across runs.
At every boundary and accepted local one-swap, the trainer appends a row.
At reselection time, `query()` returns top-k cosine-similar contexts whose
supports become candidate edits.

The bank is intentionally small and non-JIT: pickle round-trip plus numpy
cosine similarity. `save()` writes atomically (tmp file + replace) so
concurrent reads never observe a half-written bank.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hibacaml.config import HiBaCaMLConfig
    from hibacaml.types import PersistentHiBaCaMLState, PhiLike


_BANK_SCHEMA_VERSION = "v20.2b-bank-1"


@dataclass(frozen=True)
class ReplayRow:
    """One row of cross-run support history."""

    row_id: str
    run_id: str
    task_id: int
    global_step: int
    context: np.ndarray
    nonshared: Tuple[int, ...]
    full_support: Tuple[int, ...]
    phi: "PhiLike"
    observed_total: float
    observed_old_worst: float
    observed_old_mix: float
    provenance: str  # "boundary" | "local_swap" | "demotion_swap"


@dataclass
class _BankPayload:
    schema_version: str
    rows: List[ReplayRow] = field(default_factory=list)
    contributing_run_ids: List[str] = field(default_factory=list)
    last_write: float = 0.0


class SelectorBank:
    """File-backed bank of replay rows shared across runs."""

    def __init__(
        self,
        root: Path,
        *,
        bank_filename: str = "bank.pkl",
        metadata_filename: str = "bank_metadata.json",
    ):
        self.root = Path(root)
        self.bank_path = self.root / bank_filename
        self.metadata_path = self.root / metadata_filename
        self._payload = _BankPayload(schema_version=_BANK_SCHEMA_VERSION)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.bank_path.exists():
            self._payload = _BankPayload(schema_version=_BANK_SCHEMA_VERSION)
            return
        with self.bank_path.open("rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, _BankPayload):
            raise ValueError(
                f"Selector bank at {self.bank_path} is not a V20.2b _BankPayload"
            )
        if payload.schema_version != _BANK_SCHEMA_VERSION:
            raise ValueError(
                f"Selector bank schema {payload.schema_version!r} != "
                f"{_BANK_SCHEMA_VERSION!r}"
            )
        self._payload = payload

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._payload.last_write = time.time()
        tmp_path = self.bank_path.with_suffix(self.bank_path.suffix + ".tmp")
        with tmp_path.open("wb") as fh:
            pickle.dump(self._payload, fh)
        os.replace(tmp_path, self.bank_path)
        self.metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": self._payload.schema_version,
                    "row_count": len(self._payload.rows),
                    "last_write": self._payload.last_write,
                    "contributing_run_ids": self._payload.contributing_run_ids,
                },
                indent=2,
                sort_keys=True,
            )
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, row: ReplayRow) -> None:
        self._payload.rows.append(row)
        if row.run_id and row.run_id not in self._payload.contributing_run_ids:
            self._payload.contributing_run_ids.append(row.run_id)

    def extend(self, rows: Sequence[ReplayRow]) -> None:
        for row in rows:
            self.add(row)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        context: np.ndarray,
        *,
        k: int = 8,
        current_nonshared: Optional[Tuple[int, ...]] = None,
    ) -> List[ReplayRow]:
        """Return up to k bank rows ranked by cosine similarity to context.

        Deduplicates by `nonshared` so the caller never sees the same support
        twice. Optionally filters out rows whose `nonshared` equals
        `current_nonshared` (no point proposing what we already have).
        """
        rows = self._payload.rows
        if not rows:
            return []
        ctx = np.asarray(context, dtype=np.float32).reshape(-1)
        ctx_norm = float(np.linalg.norm(ctx)) + 1e-8
        scored: List[Tuple[float, ReplayRow]] = []
        for row in rows:
            if current_nonshared is not None and row.nonshared == current_nonshared:
                continue
            other = np.asarray(row.context, dtype=np.float32).reshape(-1)
            if other.shape != ctx.shape:
                continue
            sim = float(np.dot(ctx, other) / (ctx_norm * (np.linalg.norm(other) + 1e-8)))
            scored.append((sim, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        seen: set = set()
        unique: List[ReplayRow] = []
        for _, row in scored:
            if row.nonshared in seen:
                continue
            seen.add(row.nonshared)
            unique.append(row)
            if len(unique) >= k:
                break
        return unique

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def rows(self) -> Tuple[ReplayRow, ...]:
        return tuple(self._payload.rows)

    def __len__(self) -> int:
        return len(self._payload.rows)

    def summary(self) -> dict:
        rows = self._payload.rows
        if not rows:
            return {
                "total_rows": 0,
                "rows_per_task": {},
                "mean_pairwise_overlap": 0.0,
                "support_diversity_index": 0.0,
                "contributing_run_ids": list(self._payload.contributing_run_ids),
            }
        per_task: dict = {}
        for row in rows:
            per_task[int(row.task_id)] = per_task.get(int(row.task_id), 0) + 1
        unique_supports = {row.nonshared for row in rows}
        # Pairwise jaccard on at most the first 64 unique supports (cost cap).
        sample = list(unique_supports)[:64]
        jaccard_sum = 0.0
        pair_count = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                left = set(sample[i])
                right = set(sample[j])
                jaccard_sum += len(left & right) / max(len(left | right), 1)
                pair_count += 1
        return {
            "total_rows": len(rows),
            "rows_per_task": per_task,
            "mean_pairwise_overlap": (jaccard_sum / pair_count) if pair_count else 0.0,
            "support_diversity_index": float(len(unique_supports) / max(len(rows), 1)),
            "contributing_run_ids": list(self._payload.contributing_run_ids),
        }


# ----------------------------------------------------------------------
# Context construction
# ----------------------------------------------------------------------
def build_context(
    task_id: int,
    persistent_state: "PersistentHiBaCaMLState",
    cfg: "HiBaCaMLConfig",
    full_support: Sequence[int],
) -> np.ndarray:
    """Compose the 66-dim retrieval-key vector described in the V20.2b plan.

    Layout:
        [0 : num_tasks)                               one-hot task id
        [num_tasks : num_tasks + total_columns)        per-column saturation
        [.. + total_columns : .. + 2*total_columns)    per-column q_mean
        [.. + 2*total_columns : .. + 3*total_columns)  full-support mask
        [last]                                         recent gate_entropy EMA
    """
    total_cols = cfg.column_pool.total_columns
    num_tasks = max(1, int(cfg.num_tasks))

    task_onehot = np.zeros(num_tasks, dtype=np.float32)
    if 0 <= task_id < num_tasks:
        task_onehot[task_id] = 1.0

    saturation = np.zeros(total_cols, dtype=np.float32)
    q_mean = np.zeros(total_cols, dtype=np.float32)
    certs = persistent_state.certificates or {}
    for idx in range(total_cols):
        cert = certs.get(idx)
        if cert is None:
            continue
        saturation[idx] = float(cert.saturation)
        q_mean[idx] = float(cert.q_mean)

    support_mask = np.zeros(total_cols, dtype=np.float32)
    for col in full_support:
        if 0 <= int(col) < total_cols:
            support_mask[int(col)] = 1.0

    diagnostics = persistent_state.composer_diagnostics or {}
    recent_gate_entropy = 0.0
    if diagnostics:
        latest_step = max(diagnostics)
        recent_gate_entropy = float(diagnostics[latest_step].get("gate_entropy", 0.0))

    return np.concatenate(
        [
            task_onehot,
            saturation,
            q_mean,
            support_mask,
            np.asarray([recent_gate_entropy], dtype=np.float32),
        ],
        axis=0,
    )


def make_replay_row(
    *,
    run_id: str,
    task_id: int,
    global_step: int,
    context: np.ndarray,
    nonshared: Sequence[int],
    full_support: Sequence[int],
    phi: "PhiLike",
    observed_total: float,
    observed_old_worst: float,
    observed_old_mix: float,
    provenance: str,
) -> ReplayRow:
    """Convenience constructor that fills in row_id and tuple-coerces supports."""
    return ReplayRow(
        row_id=str(uuid.uuid4()),
        run_id=str(run_id),
        task_id=int(task_id),
        global_step=int(global_step),
        context=np.asarray(context, dtype=np.float32),
        nonshared=tuple(int(c) for c in sorted(nonshared)),
        full_support=tuple(int(c) for c in sorted(full_support)),
        phi=phi,
        observed_total=float(observed_total),
        observed_old_worst=float(observed_old_worst),
        observed_old_mix=float(observed_old_mix),
        provenance=str(provenance),
    )
