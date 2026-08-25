"""Accuracy, support, and swap-gain plots for HiBaCaML experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import matplotlib.pyplot as plt

def _save_fig(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_forgetting(
    mean_seen_accuracy: Sequence[float],
    mean_forgetting: Sequence[float],
    output_path: Path,
    *,
    title: str = "Accuracy And Forgetting",
) -> None:
    """Plot mean seen accuracy and mean forgetting curves across task boundaries."""
    if not mean_seen_accuracy and not mean_forgetting:
        return

    count = max(len(mean_seen_accuracy), len(mean_forgetting))
    xs = list(range(count))

    fig, axes = plt.subplots(2, 1, figsize=(max(6, count + 2), 7), sharex=True)

    if mean_seen_accuracy:
        axes[0].plot(
            range(len(mean_seen_accuracy)),
            mean_seen_accuracy,
            marker="o",
            color="#1f77b4",
            linewidth=2,
        )
        axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Mean Seen Accuracy")
    axes[0].set_title(title)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    if mean_forgetting:
        axes[1].plot(
            range(len(mean_forgetting)),
            mean_forgetting,
            marker="s",
            color="#d62728",
            linewidth=2,
        )
        axes[1].set_ylim(bottom=min(-0.02, min(mean_forgetting) - 0.02))
    axes[1].set_ylabel("Mean Forgetting")
    axes[1].set_xlabel("Task Boundary")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"T{idx}" for idx in xs])

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_support_table(
    support_sequence: Mapping[int, Sequence[int]],
    output_path: Path,
    *,
    title: str = "Chosen Support Sequence",
) -> None:
    """Render the chosen full-support sequence as a compact table."""

    if not support_sequence:
        return

    task_ids = sorted(support_sequence)
    rows = [
        [f"T{task_id}", ", ".join(str(col) for col in support_sequence[task_id])]
        for task_id in task_ids
    ]

    fig_height = max(2.5, 0.55 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=["Task", "Full Support"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e8eef8")
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_swap_gains(
    best_swap_gains: Mapping[int, float],
    output_path: Path,
    *,
    title: str = "Best One-Swap Gain By Task",
) -> None:
    """Plot the best accepted one-swap gain per task."""

    if not best_swap_gains:
        return

    task_ids = sorted(best_swap_gains)
    values = [best_swap_gains[task_id] for task_id in task_ids]
    colors = ["#2ca02c" if value > 0.0 else "#b0b0b0" for value in values]

    fig, ax = plt.subplots(figsize=(max(6, len(task_ids) * 1.4), 4.5))
    bars = ax.bar([f"T{task_id}" for task_id in task_ids], values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Best Gain")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    _save_fig(fig, output_path)
