"""Static Full-CIFAR-10 experiment (R6): joint ten-class capacity of the 4+5 active layout.

    python -m experiments.cifar [--run-id NAME]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from hibacaml import build_full_cifar_task, build_trainer, make_hibacaml_config, override, prepare_run_root
from hibacaml.experiment import (
    architecture_metadata,
    log_runtime,
    require,
    require_finite,
    require_production_runtime,
)
from hibacaml.reporting import log_progress, plot_static_run, static_task_row, write_run_records
from hibacaml.reporting.integrity import column_fingerprints
from hibacaml.training.trainer import adam_count

MODE = "cifar"
VALIDATION_FRACTION = 0.10
AUGMENT = True              # random crop + horizontal flip on the fit split
EVAL_BATCH_SIZE = 500       # one evaluation shape for the 5,000 validation and 10,000 test images
SHARED_K_CONNECTIVITY = "dense"

# 40 physical columns: 0-3 shared, 4-8 adaptive and always active, 9-39 inactive reserve.
OVERRIDES = {
    "seed": 0,
    "epochs_per_task": 60,
    "task_local_heads": False,
    "num_tasks": 1,
    "column_pool__total_columns": 40,
    "column_pool__shared_count": 4,
    "column_pool__adaptive_count": 5,
    "column_pool__reserve_count": 31,
    "column_pool__topk_nonshared": 5,
}


def _preflight(cfg) -> None:
    pool, search = cfg.column_pool, cfg.exact_search
    require(not cfg.task_local_heads, "Full-CIFAR-10 trains one ten-way head")
    require(cfg.stem is not None, "Full-CIFAR-10 requires the convolutional stem")
    require(pool.topk_nonshared == pool.adaptive_count, "the static bank keeps every adaptive column active")
    require(
        not (search.enable_exact_search or search.enable_structural_edits
             or search.enable_precision_update_resistance or search.enable_demotion_swap_audit),
        "legacy search, edits, precision resistance and demotion audits stay off",
    )


def run_experiment(cfg, *, run_id: str | None = None):
    """Run the static full-bank Full-CIFAR-10 protocol (backprop) and write its records."""
    _preflight(cfg)
    runtime = log_runtime()
    task = build_full_cifar_task(
        cfg, validation_fraction=VALIDATION_FRACTION, augment=AUGMENT, eval_batch_size=EVAL_BATCH_SIZE
    )
    bank = cfg.column_pool.active_support_size
    arm = f"full_cifar_backprop_bank{bank}"
    run_id = run_id or f"CIFAR10_backprop_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    cfg, run_output_root = prepare_run_root(cfg, run_id)
    expected_updates = len(task.train_loader) * cfg.epochs_per_task
    experiment_metadata = {
        "experiment": "full_cifar_r6",
        "arm": arm,
        "learner_contract": "backprop_full",
        "bank": bank,
        "seed": cfg.seed,
        "validation_fraction": VALIDATION_FRACTION,
        "split_seed": cfg.seed,
        "augment": AUGMENT,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "fit_examples": task.train_loader.examples,
        "validation_examples": task.validation_loader.examples,
        "test_examples": task.test_loader.examples,
        "updates_per_epoch": len(task.train_loader),
        "expected_total_updates": expected_updates,
        "heldout_refresh_certificates": False,
        "cert_refresh_interval": cfg.cert_refresh_interval,
        "training_certificate_refresh_policy": "scheduled_interval",
        "shared_k_connectivity": SHARED_K_CONNECTIVITY,
    }
    structure, trainer = build_trainer(
        cfg, (task,), "backprop", run_id=run_id, experiment_metadata=experiment_metadata,
        shared_k_connectivity=SHARED_K_CONNECTIVITY,
    )
    architecture = architecture_metadata(cfg, structure, trainer.params)
    reserves = cfg.column_pool.reserve_indices
    before = column_fingerprints(trainer.params, reserves)
    # Fit evaluation stays off: it would score augmented images and consume fit passes.
    summary = trainer.train_task(task, evaluate_train_each_epoch=False, collect_examples=True)

    state = trainer.persistent_state
    counts = dict(
        global_step=state.global_step,
        adam_count=adam_count(trainer.opt_state),
        params_revision=state.params_revision,
        train_passes=task.train_loader.passes,
    )
    require(counts == dict(global_step=expected_updates, adam_count=expected_updates,
                           params_revision=expected_updates, train_passes=cfg.epochs_per_task),
            "update counts", counts=counts)
    nonshared = state.task_support_snapshots[task.task_id].nonshared
    require(nonshared == cfg.column_pool.adaptive_indices, "the static support changed")
    require(column_fingerprints(trainer.params, reserves) == before, "an inactive reserve column changed")
    test_metrics = {k: v for k, v in trainer.completion_reports[task.task_id].items() if k != "examples"}
    test_metrics.update(best_old_accuracy=summary.best_old_accuracy, support_entropy=summary.support_entropy)
    require(test_metrics["max_selected_columns"] <= cfg.composer.topk, "composer selected more than top-k columns")
    log_progress(f"checks passed counts={counts} reserves_unchanged={len(reserves)}", component="runner")

    epoch_records = list(state.epoch_evaluations)
    final_validation_metrics = epoch_records[-1]["validation_metrics"] if epoch_records else {}
    results = {
        "run_id": run_id,
        "arm": arm,
        "learning": "backprop",
        "learner_contract": experiment_metadata["learner_contract"],
        "bank": bank,
        "architecture": architecture,
        "runtime": runtime,
        "data": {k: experiment_metadata[k] for k in (
            "split_seed", "validation_fraction", "augment", "fit_examples", "validation_examples", "test_examples",
        )},
        "schedule": {
            "epochs": cfg.epochs_per_task,
            "batch_size": cfg.batch_size,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "updates_per_epoch": len(task.train_loader),
            "expected_total_updates": expected_updates,
            "completed_total_updates": state.global_step,
            "examples_processed": sum(int(record["epoch_examples"]) for record in epoch_records),
        },
        "counts": counts,
        "support": list(trainer.active_full_support(nonshared)),
        "epoch_evaluations": epoch_records,
        "final_fit": {},
        "final_validation": final_validation_metrics,
        "official_test": test_metrics,
        "task_summary": summary.__dict__,
        "experiment_metadata": experiment_metadata,
    }
    require_finite((trainer.params, trainer.opt_state), label="model state")
    require_finite(results, label="run results")
    write_run_records(
        run_output_root,
        results,
        [static_task_row(summary, test_metrics, final_validation_metrics, state.global_step)],
    )
    plot_static_run(epoch_records, test_metrics, run_output_root / "plots")
    log_progress(
        f"run complete arm={arm} dir={run_output_root} updates={state.global_step}/{expected_updates}",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    require_production_runtime()
    return run_experiment(override(make_hibacaml_config(MODE), **OVERRIDES), run_id=args.run_id)


if __name__ == "__main__":
    main()
