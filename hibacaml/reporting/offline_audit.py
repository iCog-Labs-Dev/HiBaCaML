from __future__ import annotations
import os as _os

_os.environ.setdefault("JAX_PLATFORMS", "cpu")
from fabricpc import setup_jax as _setup_jax  # noqa: E402

_setup_jax(platform=_os.environ.get("JAX_PLATFORMS", "cpu"))

import argparse
import csv
import dataclasses
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from hibacaml.config.defaults import HiBaCaMLConfig
from hibacaml.control.search import ExactSearchService
from hibacaml.types import BoundaryBundle, SupportSearchRow


# Candidate enumeration
def all_nonshared_candidates(n_adaptive: int, k: int) -> List[Tuple[int, ...]]:
    return list(itertools.combinations(range(n_adaptive), k))


def one_swap_neighbors(
    support: Sequence[int], all_adaptive: Sequence[int]
) -> List[Tuple[int, ...]]:
    support = tuple(sorted(support))
    inactive = [c for c in all_adaptive if c not in support]
    neighbors = []
    for out_col in support:
        remaining = tuple(c for c in support if c != out_col)
        for in_col in inactive:
            neighbors.append(tuple(sorted(remaining + (in_col,))))
    return neighbors

# Per-context audit result
@dataclass(frozen=True)
class OfflineAuditRow:
    task_id: int
    checkpoint_label: str
    chosen_support: Tuple[int, ...]
    chosen_rank: int  # 1-indexed rank among all n_candidates, lower is better
    chosen_is_best: bool
    best_support: Tuple[int, ...]
    best_total: float
    chosen_total: float
    improvement: float  # chosen_total - best_total; 0.0 iff chosen_is_best
    n_candidates: int
    within_eps_counts: Dict[str, int]  # e.g. {"0.01": 15, "0.05": 120}
    best_one_swap_support: Tuple[int, ...]
    best_one_swap_total: float
    one_swap_recovery_fraction: float  # 1.0 if chosen already best; else
    # (chosen_total - best_one_swap_total) / (chosen_total - best_total)
    has_improving_one_swap: bool
    all_rows: Tuple[SupportSearchRow, ...] = field(repr=False, compare=False)

    def to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("all_rows")  # too large / not needed for the summary export
        d["chosen_support"] = list(self.chosen_support)
        d["best_support"] = list(self.best_support)
        d["best_one_swap_support"] = list(self.best_one_swap_support)
        return d


def run_offline_audit(
    search_service: ExactSearchService,
    task_id: int,
    bundle: BoundaryBundle,
    chosen_support: Sequence[int],
    adaptive_column_ids: Sequence[int],
    *,
    k: int,
    checkpoint_label: str = "",
    eps_bands: Sequence[float] = (0.01, 0.05),
) -> OfflineAuditRow:
    
    n_adaptive = len(adaptive_column_ids)
    candidate_offsets = all_nonshared_candidates(n_adaptive, k)
    candidates = [
        tuple(sorted(adaptive_column_ids[i] for i in offsets))
        for offsets in candidate_offsets
    ]

    rows = search_service.static_support_scores_batched(
        task_id, candidates, bundle
    )
    rows_by_support = {row.nonshared: row for row in rows}

    ranked = sorted(rows, key=lambda r: r.static_total)
    rank_of = {row.nonshared: i + 1 for i, row in enumerate(ranked)}

    chosen_key = tuple(sorted(chosen_support))
    if chosen_key not in rows_by_support: 
        chosen_row = search_service.static_support_score(task_id, chosen_key, bundle)
        rows_by_support[chosen_key] = chosen_row
        ranked = sorted(list(ranked) + [chosen_row], key=lambda r: r.static_total)
        rank_of = {row.nonshared: i + 1 for i, row in enumerate(ranked)}

    best_row = ranked[0]
    chosen_row = rows_by_support[chosen_key]

    within_eps_counts = {
        str(eps): sum(
            1 for row in rows if row.static_total <= best_row.static_total + eps
        )
        for eps in eps_bands
    }

    neighbor_supports = one_swap_neighbors(
        [adaptive_column_ids.index(c) for c in chosen_key], list(range(n_adaptive))
    )
    neighbor_candidates = [
        tuple(sorted(adaptive_column_ids[i] for i in offsets))
        for offsets in neighbor_supports
    ]
    neighbor_rows = [
        rows_by_support[c] for c in neighbor_candidates if c in rows_by_support
    ]
    if neighbor_rows:
        best_neighbor = min(neighbor_rows, key=lambda r: r.static_total)
    else:
        best_neighbor = chosen_row

    total_gap = chosen_row.static_total - best_row.static_total
    if total_gap <= 0.0:
        recovery_fraction = 1.0
    else:
        recovered = chosen_row.static_total - best_neighbor.static_total
        recovery_fraction = max(0.0, min(1.0, recovered / total_gap))

    return OfflineAuditRow(
        task_id=task_id,
        checkpoint_label=checkpoint_label,
        chosen_support=chosen_key,
        chosen_rank=rank_of[chosen_key],
        chosen_is_best=(chosen_key == best_row.nonshared),
        best_support=best_row.nonshared,
        best_total=best_row.static_total,
        chosen_total=chosen_row.static_total,
        improvement=max(0.0, total_gap),
        n_candidates=len(rows),
        within_eps_counts=within_eps_counts,
        best_one_swap_support=best_neighbor.nonshared,
        best_one_swap_total=best_neighbor.static_total,
        one_swap_recovery_fraction=recovery_fraction,
        has_improving_one_swap=(best_neighbor.static_total < chosen_row.static_total),
        all_rows=tuple(rows),
    )


# Multi-context aggregation (Table 1 style summary)
def aggregate_offline_audit(rows: Sequence[OfflineAuditRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {}

    def _mean(xs):
        return sum(xs) / len(xs)

    ranks = [r.chosen_rank for r in rows]
    improvements = [r.improvement for r in rows]
    eps_keys = sorted(rows[0].within_eps_counts.keys(), key=float)

    return {
        "n_contexts": n,
        "contexts_chosen_is_best": sum(r.chosen_is_best for r in rows),
        "mean_rank_of_chosen": _mean(ranks),
        "median_rank_of_chosen": sorted(ranks)[n // 2],
        "worst_rank_of_chosen": max(ranks),
        "mean_improvement": _mean(improvements),
        "median_improvement": sorted(improvements)[n // 2],
        "max_improvement": max(improvements),
        **{
            f"mean_n_within_{eps}_of_best": _mean(
                [r.within_eps_counts[eps] for r in rows]
            )
            for eps in eps_keys
        },
        "contexts_with_improving_one_swap": sum(
            r.has_improving_one_swap for r in rows
        ),
        "contexts_best_one_swap_reaches_global_best": sum(
            r.best_one_swap_support == r.best_support for r in rows
        ),
        "mean_one_swap_recovery_fraction": _mean(
            [r.one_swap_recovery_fraction for r in rows]
        ),
    }


def column_usage_bias(
    rows: Sequence[OfflineAuditRow], adaptive_column_ids: Sequence[int]
) -> Dict[int, Dict[str, int]]:
    counts = {
        col: {"chosen_count": 0, "best_count": 0} for col in adaptive_column_ids
    }
    for row in rows:
        for col in row.chosen_support:
            if col in counts:
                counts[col]["chosen_count"] += 1
        for col in row.best_support:
            if col in counts:
                counts[col]["best_count"] += 1
    return counts

# Multi-checkpoint driver
def run_offline_audit_over_checkpoints(
    cfg: HiBaCaMLConfig,
    tasks,
    checkpoint_specs: Sequence[Tuple[int, str, Path]],
    *,
    learning: str = "backprop",
    eps_bands: Sequence[float] = (0.01, 0.05),
) -> List[OfflineAuditRow]:
    # Local import to avoid a hard dependency of this module on the
    # experiment script when only the library functions above are used
    # (e.g. from a notebook with an already-constructed trainer).
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
    from split_mnist import build_trainer  # noqa: E402

    all_rows: List[OfflineAuditRow] = []

    for task_id, label, checkpoint_path in checkpoint_specs:
        _, trainer = build_trainer(cfg, tasks, learning, run_id=f"audit_{label}")
        trainer.load_checkpoint(checkpoint_path)

        snapshot = trainer.persistent_state.task_support_snapshots.get(task_id)
        if snapshot is None:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no recorded support "
                f"snapshot for task_id={task_id}; cannot audit this context."
            )
        chosen_support = tuple(sorted(snapshot.nonshared))

        search_service = trainer.exact_search
        bundle = search_service.make_bundle(task_id)

        adaptive_column_ids = list(cfg.column_pool.adaptive_indices)
        row = run_offline_audit(
            search_service,
            task_id,
            bundle,
            chosen_support,
            adaptive_column_ids,
            k=cfg.column_pool.topk_nonshared,
            checkpoint_label=label,
            eps_bands=eps_bands,
        )
        all_rows.append(row)

    return all_rows

# Export
def export_offline_audit(
    rows: Sequence[OfflineAuditRow],
    adaptive_column_ids: Sequence[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-context CSV
    csv_path = out_dir / "offline_selector_audit.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        eps_keys = sorted(rows[0].within_eps_counts.keys(), key=float) if rows else []
        writer.writerow(
            [
                "task_id",
                "checkpoint_label",
                "chosen_support",
                "chosen_rank",
                "chosen_is_best",
                "best_support",
                "best_total",
                "chosen_total",
                "improvement",
                "n_candidates",
                *[f"n_within_{eps}_of_best" for eps in eps_keys],
                "best_one_swap_support",
                "best_one_swap_total",
                "one_swap_recovery_fraction",
                "has_improving_one_swap",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.task_id,
                    r.checkpoint_label,
                    "|".join(map(str, r.chosen_support)),
                    r.chosen_rank,
                    r.chosen_is_best,
                    "|".join(map(str, r.best_support)),
                    r.best_total,
                    r.chosen_total,
                    r.improvement,
                    r.n_candidates,
                    *[r.within_eps_counts[eps] for eps in eps_keys],
                    "|".join(map(str, r.best_one_swap_support)),
                    r.best_one_swap_total,
                    r.one_swap_recovery_fraction,
                    r.has_improving_one_swap,
                ]
            )

    # Table-1-style aggregate summary
    summary = aggregate_offline_audit(rows)
    with (out_dir / "offline_selector_audit_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    # Column usage bias
    bias = column_usage_bias(rows, adaptive_column_ids)
    bias_path = out_dir / "column_usage_bias.csv"
    with bias_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["column_id", "chosen_count", "best_count"])
        for col, counts in sorted(bias.items()):
            writer.writerow([col, counts["chosen_count"], counts["best_count"]])


# CLI
def _parse_checkpoint_specs(run_dir: Path) -> List[Tuple[int, str, Path]]:
    """Discover (task_id, label, checkpoint_path) triples from a run
    directory laid out like experiments/split_mnist.py produces:
    <run_dir>/task_<N>/checkpoints/<checkpoint_filename>.
    """
    specs = []
    for task_dir in sorted(run_dir.glob("task_*")):
        try:
            task_id = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        checkpoint_dir = task_dir / "checkpoints"
        if not checkpoint_dir.is_dir():
            continue
        for ckpt in sorted(checkpoint_dir.glob("*")):
            specs.append((task_id, f"task_{task_id}", ckpt))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        default="paper_faithful",
        choices=["paper_faithful", "smoke", "full"],
        help=(
            "HiBaCaMLConfig mode used to produce the run being audited "
            "(e.g. --mode smoke for a run started with MODE='smoke' in "
            "experiments/split_mnist.py). This MUST match the mode the "
            "checkpoint was saved under -- run_summary.json does not "
            "currently record which mode a run used, so this cannot be "
            "auto-detected. A mismatch loads checkpointed params into a "
            "differently-shaped graph and fails with a JAX shape-mismatch "
            "error deep inside node forward passes, not a clear config error."
        ),
    )
    args = parser.parse_args()

    from hibacaml.config.defaults import make_hibacaml_config
    from hibacaml.data.split_mnist import build_split_mnist_tasks

    cfg = make_hibacaml_config(mode=args.mode)
    tasks = build_split_mnist_tasks(cfg)

    checkpoint_specs = _parse_checkpoint_specs(args.run_dir)
    if not checkpoint_specs:
        raise SystemExit(f"No checkpoints found under {args.run_dir}")

    rows = run_offline_audit_over_checkpoints(cfg, tasks, checkpoint_specs)

    out_dir = args.out_dir or (args.run_dir / "offline_audit")
    export_offline_audit(rows, list(cfg.column_pool.adaptive_indices), out_dir)
    print(f"Offline audit written to {out_dir}")


if __name__ == "__main__":
    main()
