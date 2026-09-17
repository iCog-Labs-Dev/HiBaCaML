"""Static full-bank Full-MNIST experiment for HiBaCaML."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict
import jax

sys.path.append(".")

from hibacaml.reporting.logger import log_progress
from hibacaml.control.support import default_nonshared_support
from hibacaml import (
    build_full_mnist_task,
    build_trainer,
    export_run_artifacts,
    make_hibacaml_config,
    override,
    prepare_run_root,
)
from hibacaml.reporting import (
    plot_composer_usage,
    plot_confusion_matrix,
    plot_learning_curve,
    write_csv,
    write_json,
)


MODE = "default"
LEARNING = "backprop"  # "backprop" or "pc"
VALIDATION_FRACTION = 0.10
RUN_ID = None

# TODO: Clarify the full-bank phrasing: BANK=17 leaves three physical reserve columns inactive.
BANK = 17

_SHARED_COUNT = 2
_TOTAL_COLUMNS = 20


def _bank_overrides(bank: int) -> Dict[str, int]:
    """Config fields that make `bank` columns active, with none held in reserve."""
    if isinstance(bank, bool) or not isinstance(bank, int):
        raise TypeError(f"BANK must be an integer, got {type(bank).__name__}")
    if not _SHARED_COUNT < bank <= _TOTAL_COLUMNS:
        raise ValueError(
            f"BANK must be in ({_SHARED_COUNT}, {_TOTAL_COLUMNS}]: the graph has "
            f"{_TOTAL_COLUMNS} columns, {_SHARED_COUNT} of them shared"
        )
    adaptive = bank - _SHARED_COUNT
    return {
        "column_pool__adaptive_count": adaptive,
        "column_pool__reserve_count": _TOTAL_COLUMNS - bank,
        "column_pool__topk_nonshared": adaptive,
    }


OVERRIDES = {
    "seed": 0,
    "batch_size": 256,
    "epochs_per_task": 5,
    "task_local_heads": False,
    "num_tasks": 1,
    "infer_steps": 16,
    "eta_infer": 0.05,
    **_bank_overrides(BANK),
    "composer__topk": 3,
    "exact_search__enable_exact_search": False,
    "exact_search__enable_structural_edits": False,
    "exact_search__enable_precision_update_resistance": False,
    "exact_search__enable_demotion_swap_audit": False,
    "reporting__write_selector_state": False,
    "reporting__experiment_root": "runs/experiments/full_mnist_architecture",
}


def _select(source: Dict[str, object], *names: str) -> Dict[str, object]:
    """Pick named fields, failing loudly if one is missing."""
    return {name: source[name] for name in names}


def _arm_name(learning: str, bank: int) -> str:
    learner = {
        "backprop": "backprop",
        "pc": "pc_local",
    }.get(learning)
    if learner is None:
        raise ValueError("learning must be 'pc' or 'backprop'")
    return f"full_mnist_{learner}_bank{bank}"


def _architecture_metadata(cfg) -> Dict[str, object]:
    pool = cfg.column_pool
    active_support = tuple(pool.shared_indices + pool.adaptive_indices)
    aggregation_scale = 1.0 / pool.active_support_size
    return {
        "total_columns": pool.total_columns,
        "shared_columns": pool.shared_indices,
        "adaptive_columns": pool.adaptive_indices,
        "reserve_columns": pool.reserve_indices,
        "topk_nonshared": pool.topk_nonshared,
        "active_support_size": pool.active_support_size,
        "aggregation_scale": aggregation_scale,
        "composer_topk": cfg.composer.topk,
        "initial_nonshared_support": pool.adaptive_indices,
        "initial_full_support": active_support,
        "inactive_reserve_columns": pool.reserve_indices,
    }


def build_tasks(cfg, *, validation_fraction: float = VALIDATION_FRACTION):
    """Build the dedicated one-task Full-MNIST experiment input."""
    return (
        build_full_mnist_task(
            cfg,
            validation_fraction=validation_fraction,
            split_seed=cfg.seed,
        ),
    )


def _generate_plots(epoch_records, test_metrics, plots_dir: Path) -> None:
    """Write the three figures relevant to the static Full-MNIST study."""
    plot_learning_curve(epoch_records, plots_dir / "learning_curve.png")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        plots_dir / "test_confusion_matrix.png",
    )
    plot_composer_usage(
        test_metrics["column_gate_mean"],
        test_metrics["column_selection_fraction"],
        plots_dir / "composer_usage.png",
    )


def run_experiment(
    cfg,
    *,
    learning: str = LEARNING,
    validation_fraction: float = VALIDATION_FRACTION,
    run_id: str | None = RUN_ID,
):
    """Run the full-bank Full-MNIST protocol and write its complete artifacts.

    The bank size is read from `cfg`, not passed in, so the configuration stays
    the single source of truth.
    """
    if learning not in ("pc", "backprop"):
        raise ValueError("learning must be 'pc' or 'backprop'")
    if cfg.task_local_heads:
        raise ValueError("Full-MNIST requires task_local_heads=False")
    if cfg.composer.topk != 3:
        raise ValueError("Full-MNIST Phase 1A requires composer.topk == 3")

    log_progress(
        f"backend={jax.default_backend()} devices={jax.devices()}",
        component="runner",
    )
    tasks = build_tasks(cfg, validation_fraction=validation_fraction)
    task = tasks[0]
    bank = cfg.column_pool.active_support_size
    arm = _arm_name(learning, bank)
    relative_run_id = run_id or f"{arm}/seed_{cfg.seed}"
    cfg, run_output_root = prepare_run_root(
        cfg,
        relative_run_id,
        selector_state_root=str(
            cfg.experiment_root_path() / relative_run_id / "selector_state"
        ),
    )

    architecture = _architecture_metadata(cfg)
    expected_updates = len(task.train_loader) * cfg.epochs_per_task
    experiment_metadata = {
        "experiment": "full_mnist_architecture",
        "arm": arm,
        "learner_contract": "backprop_full" if learning == "backprop" else "pc_local",
        "bank": bank,
        "seed": cfg.seed,
        "validation_fraction": validation_fraction,
        "split_seed": cfg.seed,
        "fit_examples": task.train_loader.examples,
        "validation_examples": task.validation_loader.examples,
        "test_examples": task.test_loader.examples,
        "updates_per_epoch": len(task.train_loader),
        "expected_total_updates": expected_updates,
        "heldout_refresh_certificates": False,
        "cert_refresh_interval": cfg.cert_refresh_interval,
        "training_certificate_refresh_policy": (
            "pc_clamp_build_plus_scheduled_interval"
            if learning == "pc"
            else "scheduled_interval"
        ),
        "architecture": architecture,
    }
    structure, trainer = build_trainer(
        cfg,
        tasks,
        learning,
        run_id=relative_run_id,
        experiment_metadata=experiment_metadata,
    )

    pool = cfg.column_pool
    if pool.topk_nonshared != pool.adaptive_count:
        raise AssertionError(
            "Full-MNIST requires a full bank: "
            f"topk_nonshared={pool.topk_nonshared} != adaptive_count={pool.adaptive_count}"
        )
    expected_nonshared = tuple(pool.adaptive_indices)
    expected_support = tuple(range(bank))
    if default_nonshared_support(cfg) != expected_nonshared:
        raise AssertionError("default support is not the full adaptive bank")
    summary = trainer.train_task(
        task,
        final_evaluation_refresh_certificates=False,
        evaluate_train_each_epoch=True,
    )
    persisted_nonshared = trainer.persistent_state.current_support[task.task_id]
    full_support = trainer.active_full_support(persisted_nonshared)
    if tuple(persisted_nonshared) != expected_nonshared:
        raise AssertionError("Full-MNIST support changed during training")
    if full_support != expected_support:
        raise AssertionError(
            f"Full-MNIST support must be columns 0..{bank - 1}"
        )
    if set(full_support) & set(pool.reserve_indices):
        raise AssertionError("reserve columns must remain inactive")
    if trainer.persistent_state.global_step != expected_updates:
        raise AssertionError(
            "completed update count does not match the declared schedule: "
            f"{trainer.persistent_state.global_step} != {expected_updates}"
        )

    test_metrics = trainer.evaluate_task(
        task.task_id,
        refresh_certificates=False,
    )
    if test_metrics["max_selected_columns"] > 3:
        raise AssertionError("composer selected more than three columns")
    epoch_records = list(trainer.persistent_state.epoch_evaluations)
    completed_examples = sum(
        int(record["epoch_examples"]) for record in epoch_records
    )
    completed_schedule = {
        "completed_total_updates": trainer.persistent_state.global_step,
        "examples_processed": completed_examples,
    }
    experiment_metadata.update(completed_schedule)
    trainer.experiment_metadata.update(completed_schedule)
    snapshot = trainer.snapshot(refresh_evaluation_certificates=False)
    export_run_artifacts(snapshot, run_output_root / "final")
    trainer.export_task_artifacts(
        task.task_id,
        root=run_output_root / "task_0",
        refresh_evaluation_certificates=False,
    )
    checkpoint_path = trainer.save_checkpoint(
        task.task_id,
        root=run_output_root / "checkpoints",
    )

    final_train_metrics = epoch_records[-1]["train_metrics"] if epoch_records else {}
    final_validation_metrics = (
        epoch_records[-1]["validation_metrics"] if epoch_records else {}
    )
    results = {
        "run_id": relative_run_id,
        "arm": arm,
        "learning": learning,
        "learner_contract": experiment_metadata["learner_contract"],
        "bank": bank,
        "architecture": architecture,
        # Both sub-dicts select from experiment_metadata rather than restating it,
        # so the two records cannot drift apart.
        "data": _select(
            experiment_metadata,
            "split_seed",
            "validation_fraction",
            "fit_examples",
            "validation_examples",
            "test_examples",
        ),
        "schedule": {
            "epochs": cfg.epochs_per_task,
            "batch_size": cfg.batch_size,
            **_select(
                experiment_metadata,
                "updates_per_epoch",
                "expected_total_updates",
                "completed_total_updates",
                "examples_processed",
            ),
        },
        "support": list(full_support),
        "epoch_evaluations": epoch_records,
        "final_fit": final_train_metrics,
        "final_validation": final_validation_metrics,
        "official_test": test_metrics,
        "task_summary": summary.__dict__,
        "checkpoint_path": str(checkpoint_path),
        "experiment_metadata": experiment_metadata,
    }
    write_json(run_output_root / "run_summary.json", results)
    write_csv(
        run_output_root / "task_summary.csv",
        [
            {
                "task_id": summary.task_id,
                "classes": list(summary.classes),
                "support": list(summary.support_indices),
                "test_accuracy": test_metrics["accuracy"],
                "test_cross_entropy": test_metrics["cross_entropy"],
                "test_mean_loss": test_metrics["mean_loss"],
                "validation_accuracy": final_validation_metrics.get("accuracy"),
                "validation_cross_entropy": final_validation_metrics.get("cross_entropy"),
                "completed_updates": trainer.persistent_state.global_step,
            }
        ],
    )
    _generate_plots(epoch_records, test_metrics, run_output_root / "plots")

    log_progress(
        f"run complete arm={arm} dir={run_output_root} "
        f"updates={trainer.persistent_state.global_step}/{expected_updates}",
        component="runner",
    )
    return {
        "cfg": cfg,
        "structure": structure,
        "trainer": trainer,
        "results": results,
        "run_output_root": run_output_root,
        "summary_path": run_output_root / "run_summary.json",
    }


if __name__ == "__main__":
    config = override(make_hibacaml_config(MODE), **OVERRIDES)
    run_experiment(config)
