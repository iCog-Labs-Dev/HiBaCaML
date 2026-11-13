"""Split-CIFAR-10 integrated reference A and its controls (Task-IL over five class pairs).

This module currently runs the sequential foundation: lowest-index fresh supports,
physical role labels from T0, post-T0 write protection, continuous training state
and Split-MNIST-style run records rewritten after every boundary. The support
teacher and the structural controller plug in through `on_epoch_end` later.

    XLA_FLAGS=--xla_gpu_deterministic_ops=true python -m experiments.split_cifar [--seed S] [--tasks-limit N]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, List

from hibacaml import build_split_cifar_tasks, build_trainer, make_hibacaml_config, override, prepare_run_root
from hibacaml.control.roles import initial_roles, unit_scale_table, validate_roles
from hibacaml.data.cifar import task_seeds
from hibacaml.experiment import (
    architecture_metadata,
    log_runtime,
    require,
    require_finite,
    require_production_runtime,
)
from hibacaml.reporting import (
    best_swap_gains,
    continual_extras,
    log_progress,
    mean_forgetting,
    mean_seen_accuracy,
    phi_trajectory,
    plot_sequential_run,
    support_entropy_trajectory,
    support_sequence,
    task_metric_rows,
    task_summary_row,
    write_run_records,
)
from hibacaml.reporting.integrity import column_fingerprints, fingerprint, preserve_reporting_state
from hibacaml.training.trainer import adam_count
from hibacaml.training.write_policy import PolicyLayout, WritePolicy, frozen_leaves

MODE = "cifar"
VALIDATION_FRACTION = 0.10  # per task, carved from its fit data; decisions read it, never test
AUGMENT = True              # random crop + horizontal flip on the fit split
EVAL_BATCH_SIZE = 500       # one evaluation shape for the 1,000 validation and 2,000 test images
SHARED_K_CONNECTIVITY = "kernel-only"  # shared K kernel units read only kernel coordinates

# 40 physical columns: 0-2 shared, 3-33 adaptive, 34-39 reserve; four nonshared active per task.
OVERRIDES = {
    "seed": 0,
    "epochs_per_task": 30,
    "task_local_heads": True,
    "num_tasks": 5,
    "column_pool__total_columns": 40,
    "column_pool__shared_count": 3,
    "column_pool__adaptive_count": 31,
    "column_pool__reserve_count": 6,
    "column_pool__topk_nonshared": 4,
    "reporting__log_gradient_clipping": True,
}


def fresh_supports(cfg, count: int):
    """Lowest-index fresh allocation: task t owns adaptive columns [t*k, (t+1)*k)."""
    adaptive, k = cfg.column_pool.adaptive_indices, cfg.column_pool.topk_nonshared
    require(len(adaptive) >= count * k, "not enough adaptive columns for fresh supports")
    return {t: tuple(adaptive[t * k:(t + 1) * k]) for t in range(count)}


def _preflight(cfg) -> None:
    search = cfg.exact_search
    require(cfg.task_local_heads, "Split-CIFAR-10 trains task-local two-way heads")
    require(cfg.stem is not None, "Split-CIFAR-10 requires the convolutional stem")
    require(cfg.reporting.log_gradient_clipping, "reference A records clipping through the instrumented optimizer")
    require(
        not (search.enable_exact_search or search.enable_structural_edits
             or search.enable_precision_update_resistance or search.enable_demotion_swap_audit),
        "legacy search, edits, precision resistance and demotion audits stay off",
    )


def _post_t0_policy(structure, cfg, trainer, own) -> WritePolicy:
    """Shared columns keep radial label rates and frozen pools; this task's columns write freely."""
    shared = cfg.column_pool.shared_indices
    table = unit_scale_table(structure, cfg, trainer.persistent_state.shell_roles, shared)
    return WritePolicy(tuple(sorted(shared + tuple(own))), shared, table)


def _frozen_digest(params, leaves) -> str:
    return fingerprint({leaf: getattr(params.nodes[leaf[0]], leaf[1])[leaf[2]] for leaf in leaves})


def _records(trainer, matrices, task_summaries, extra):
    """Split-MNIST's summary keys and task rows, plus the CIFAR-specific `extra` entries."""
    test = matrices["test"]
    seen_accuracy, forgetting = mean_seen_accuracy(test), mean_forgetting(test)
    supports, swap_gains = support_sequence(trainer), best_swap_gains(trainer)
    results = {
        "learning": "backprop",
        "accuracy_matrix": test,
        "validation_accuracy_matrix": matrices["validation"],
        "mean_seen_accuracy": seen_accuracy,
        "mean_forgetting": forgetting,
        "continual_metrics": {split: continual_extras(matrices[split]) for split in ("test", "validation")},
        "support_sequence": supports,
        "phi_trajectory": phi_trajectory(trainer),
        "best_swap_gains": swap_gains,
        "support_entropy_trajectory": support_entropy_trajectory(task_summaries),
        "task_summaries": task_summaries,
        **extra,
    }
    rows = task_metric_rows(sorted(test), seen_accuracy, forgetting, supports, swap_gains, task_summaries)
    return results, rows


def run_experiment(cfg, *, tasks_limit: int | None = None, run_id: str | None = None, on_epoch_end=None):
    """Train the task sequence with fresh supports and write the records after every boundary."""
    _preflight(cfg)
    runtime = log_runtime()
    tasks = build_split_cifar_tasks(
        cfg, limit=tasks_limit, validation_fraction=VALIDATION_FRACTION,
        augment=AUGMENT, eval_batch_size=EVAL_BATCH_SIZE,
    )
    require(all(task.train_loader.passes == 0 for task in tasks), "task construction consumed a fit pass")
    log_progress(f"loaded {len(tasks)} Split-CIFAR-10 tasks", component="runner")
    supports = fresh_supports(cfg, len(tasks))
    run_id = run_id or f"SCIFAR10_backprop_seed{cfg.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    cfg, run_output_root = prepare_run_root(cfg, run_id)
    structure, trainer = build_trainer(
        cfg, tasks, "backprop", run_id=run_id,
        experiment_metadata={
            "validation_fraction": VALIDATION_FRACTION,
            "augment": AUGMENT,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "shared_k_connectivity": SHARED_K_CONNECTIVITY,
            "task_seeds": {task.task_id: task_seeds(cfg.seed, task.task_id) for task in tasks},
        },
        shared_k_connectivity=SHARED_K_CONNECTIVITY,
    )
    # Kernel-only changes no shape or count, so assert it rather than trust the default.
    require(structure.config["hibacaml"]["shared_k_connectivity"] == SHARED_K_CONNECTIVITY,
            "shared K connectivity does not match the profile")
    # Roles before T0: certificates use role masks from the first refresh.
    trainer.persistent_state.shell_roles = validate_roles(structure, cfg, initial_roles(structure, cfg))
    layout = PolicyLayout.from_structure(structure, cfg)
    pool = cfg.column_pool
    references = column_fingerprints(trainer.params, pool.adaptive_indices + pool.reserve_indices)
    per_task = [len(task.train_loader) * cfg.epochs_per_task for task in tasks]
    extra = {
        "status": "running",
        "runtime": runtime,
        "architecture": architecture_metadata(cfg, structure, trainer.params),
        "supports": supports,
        "schedule": {
            "epochs_per_task": cfg.epochs_per_task,
            "batch_size": cfg.batch_size,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "updates_per_task": per_task,
            "expected_total_updates": sum(per_task),
        },
        "boundaries": [],
    }
    guard: Dict[str, object] = {}

    def hook(t, task, epoch):
        if guard:
            require(_frozen_digest(t.params, guard["leaves"]) == guard["digest"],
                    "a protected leaf changed", task_id=task.task_id, epoch=epoch)
        if on_epoch_end is not None:
            on_epoch_end(t, task, epoch)
            if guard:
                require(_frozen_digest(t.params, guard["leaves"]) == guard["digest"],
                        "a protected leaf changed", task_id=task.task_id, epoch=epoch)

    matrices = {"test": {}, "validation": {}}
    task_summaries: List[Dict[str, object]] = []
    completed = 0
    for task in tasks:
        tid, own = task.task_id, supports[task.task_id]
        log_progress(f"task={tid} train start support={own}", component="runner")
        if tid >= 1:
            policy = _post_t0_policy(structure, cfg, trainer, own)
            trainer.set_write_policy(policy)
            if not guard:  # T0 parameters, frozen from here on
                leaves = frozen_leaves(trainer.params, policy, layout)
                guard.update(leaves=leaves, digest=_frozen_digest(trainer.params, leaves))
                extra["protection_activation_step"] = trainer.persistent_state.global_step + 1
                log_progress(f"post-T0 protection active from step {extra['protection_activation_step']} "
                             f"protected_leaves={len(leaves)}", component="runner")
        phi = cfg.phi if tid == 0 else trainer.current_phi
        trainer.set_boundary_choice(tid, own, phi, record_dropped_columns=False)
        trainer.set_current_support(tid, own, phi)
        summary = trainer.train_task(task, collect_examples=True, on_epoch_end=hook)

        # Boundary checks: counts, supports, isolation and protection.
        completed += per_task[tid]
        state = trainer.persistent_state
        counts = dict(
            global_step=state.global_step,
            adam_count=adam_count(trainer.opt_state),
            params_revision=state.params_revision,
            roles_revision=state.roles_revision,
            train_passes=task.train_loader.passes,
        )
        require(counts == dict(global_step=completed, adam_count=completed, params_revision=completed,
                               roles_revision=0, train_passes=cfg.epochs_per_task),
                "boundary counts", task_id=tid, counts=counts)
        require(not state.recently_demoted, "fixed allocation must not record dropped columns")
        require({i: s.nonshared for i, s in state.task_support_snapshots.items()}
                == {i: supports[i] for i in range(tid + 1)}, "saved supports differ from the allocation")
        current = column_fingerprints(trainer.params, references)
        changed = sorted(c for c in references if current[c] != references[c])
        require(set(changed) <= set(own), "a column outside this task's support changed",
                task_id=tid, changed=changed)
        references = current  # the next task is checked against this boundary
        if guard:
            require(_frozen_digest(trainer.params, guard["leaves"]) == guard["digest"],
                    "a protected leaf changed", task_id=tid)
        log_progress(f"task={tid} checks passed counts={counts} changed_columns={changed} "
                     f"protected_leaves={len(guard.get('leaves', ()))}", component="runner")

        # Boundary evaluation: test from the completion pass, validation without moving the RNG.
        tests = trainer.completion_reports
        require(set(tests) == set(range(tid + 1)), "completion reports must cover every seen task")
        reports = {"test": {j: tests[j] for j in range(tid + 1)}, "validation": {}}
        with preserve_reporting_state(trainer):
            for seen in tasks[: tid + 1]:
                reports["validation"][seen.task_id] = trainer.evaluate_loader(
                    seen, seen.validation_loader, supports[seen.task_id], split_name="validation",
                    mode="saved", refresh_certificates=False,
                )
        for split, by_task in reports.items():
            for j, report in by_task.items():
                require(report["max_selected_columns"] <= cfg.composer.topk,
                        "composer selected more than top-k columns")
            matrices[split][tid] = {j: report["accuracy"] for j, report in by_task.items()}
        task_summaries.append(task_summary_row(summary))
        extra["boundaries"].append({
            "task_id": tid,
            "support": own,
            "counts": counts,
            "changed_columns": changed,
            "reports": {split: {j: {k: v for k, v in r.items() if k != "examples"} for j, r in by_task.items()}
                        for split, by_task in reports.items()},
        })
        results, rows = _records(trainer, matrices, task_summaries, extra)
        require_finite((trainer.params, trainer.opt_state), label=f"task {tid} model state")
        require_finite((results, state.epoch_evaluations, trainer.clipping_epoch_summaries),
                       label=f"task {tid} results")
        write_run_records(run_output_root, results, rows,
                          clipping_epoch_summaries=trainer.clipping_epoch_summaries)
        log_progress(f"task={tid} train done accuracy={summary.accuracy:.4f} "
                     f"support={summary.support_indices}", component="runner")

    extra["status"] = "complete" if len(tasks) == cfg.num_tasks else "prefix_complete"
    results, rows = _records(trainer, matrices, task_summaries, extra)
    write_run_records(run_output_root, results, rows, clipping_epoch_summaries=trainer.clipping_epoch_summaries)
    plots_dir = run_output_root / "plots"
    plot_sequential_run(results, plots_dir)
    log_progress(f"run complete id={run_id} dir={run_output_root} status={extra['status']}", component="runner")
    return {
        "cfg": cfg,
        "trainer": trainer,
        "tasks": tasks,
        "results": results,
        "run_id": run_id,
        "run_output_root": run_output_root,
        "plots_dir": plots_dir,
        "summary_path": run_output_root / "run_summary.json",
        "table_path": run_output_root / "task_summary.csv",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-limit", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    require_production_runtime(deterministic_xla=True)
    cfg = override(make_hibacaml_config(MODE), **{**OVERRIDES, "seed": args.seed})
    return run_experiment(cfg, tasks_limit=args.tasks_limit, run_id=args.run_id)


if __name__ == "__main__":
    main()
