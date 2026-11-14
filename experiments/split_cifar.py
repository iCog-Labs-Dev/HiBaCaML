"""Split-CIFAR-10 integrated reference A and its controls (Task-IL over five class pairs).

Each task's support comes from the restricted support teacher (`hibacaml.control.reuse`):
after T0 a task may adopt one earlier learned column, chosen from isolated 35-update
trials. `fixed_fresh` and `replay` are the controls. Physical role labels exist from
T0, post-T0 write protection is exact, training state is continuous, and Split-MNIST
style records are rewritten after every boundary. From T1 the structural controller
(`hibacaml.control.structure`) runs audited relabel/inhibition/pruning rounds in the
epoch hook; `structure="off"` keeps its observations without installing anything.

Production runs all five tasks on GPU/JIT; no deterministic-XLA flag is required.
Choose controls in the settings block below. Output names are generated automatically.

    python -m experiments.split_cifar [--seed S]
"""

from __future__ import annotations

import argparse
import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import hibacaml.control.structure as structural
from hibacaml import build_split_cifar_tasks, build_trainer, make_hibacaml_config, override, prepare_run_root
from hibacaml.control.reuse import (
    SUPPORT_POLICIES,
    BoundarySelector,
    shared_references,
    trial_update_bound,
    validate_replay,
)
from hibacaml.control.roles import initial_roles, validate_roles
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
from hibacaml.reporting.integrity import changed_leaves, column_fingerprints, preserve_reporting_state
from hibacaml.training.trainer import adam_count
from hibacaml.training.write_policy import PolicyLayout

MODE = "cifar"
VALIDATION_FRACTION = 0.10  # per task, carved from its fit data; decisions read it, never test
AUGMENT = True              # random crop + horizontal flip on the fit split
EVAL_BATCH_SIZE = 500       # one evaluation shape for the 1,000 validation and 2,000 test images
SHARED_K_CONNECTIVITY = "kernel-only"  # shared K kernel units read only kernel coordinates

# Production choices: A by default; controls use the same learning machinery.
SUPPORT_POLICY = "restricted_reuse"
STRUCTURE = "on"
PROMOTIONS = "on"
REPLAY_SOURCE = None  # JSON run summary; required only when SUPPORT_POLICY == "replay"
PROMOTION_REPRESENTATION = "role_relabel"  # fixed representation, not a selectable algorithm

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


def condition_label(support_policy: str, structure: str, promotions: str) -> str:
    """Name only established control combinations; other valid combinations are custom."""
    if support_policy == "replay":
        return "replay"
    effective = promotions if structure == "on" else "off"
    return {
        ("restricted_reuse", "on", "on"): "A",
        ("restricted_reuse", "on", "off"): "C",
        ("restricted_reuse", "off", "off"): "D",
        ("fixed_fresh", "off", "off"): "E",
    }.get((support_policy, structure, effective), "custom")


def _run_name(cfg, support_policy: str, structure: str, promotions: str) -> str:
    label = condition_label(support_policy, structure, promotions)
    if label in ("custom", "replay"):
        effective = promotions if structure == "on" else "off"
        label += (f"_{support_policy}" if label == "custom" else "")
        label += f"_structure-{structure}_promotions-{effective}"
    pool = cfg.column_pool
    return (f"SCIFAR10_backprop_integrated-v2_{label}_s{pool.shared_count}o{pool.topk_nonshared}"
            f"_seed{cfg.seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")


def load_replay(source, pool, task_count: int):
    """Read decisions from a current or historical v2 JSON summary, then validate allocation."""
    payload = json.loads(Path(source).read_text())
    require(isinstance(payload, dict), "replay source must be a JSON run summary")
    if "support_selection" in payload:
        selection = payload["support_selection"]
        require(isinstance(selection, dict) and isinstance(selection.get("boundaries"), list),
                "replay source lacks support-selection boundaries")
        records = []
        for boundary in selection["boundaries"]:
            require(isinstance(boundary, dict) and "task_id" in boundary and "selected" in boundary,
                    "replay boundary lacks a task id or selected condition")
            records.append((boundary["task_id"], boundary["selected"]))
    else:
        pilot = payload.get("pilot", {})
        selection = pilot.get("selection", {}) if isinstance(pilot, dict) else {}
        decisions = selection.get("decisions") if isinstance(selection, dict) else None
        require(isinstance(decisions, dict), "replay source has no supported decision records")
        records = list(decisions.items())
    replay = {}
    fields = ("kind", "column", "support", "writable")
    for raw_id, record in records:
        require(isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool),
                "replay task ids must be integers", task_id=raw_id)
        task_id = int(raw_id)
        require(str(task_id) == str(raw_id) and task_id not in replay,
                "replay task ids must be distinct integers", task_id=raw_id)
        require(isinstance(record, dict) and all(k in record for k in fields),
                "replay decision lacks kind, column, support or writable", task_id=task_id)
        replay[task_id] = {k: record[k] for k in fields}
    validate_replay(pool, replay, range(task_count))
    return replay


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


def run_experiment(cfg, *, tasks_limit: int | None = None, run_id: str | None = None, on_epoch_end=None,
                   support_policy: str = "restricted_reuse", replay_supports=None,
                   structure: str = "on", promotions: str = "on"):
    """Train the task sequence under one support policy and write the records after every boundary.

    `replay_supports` (replay only) maps task ids to recorded `{kind, column, support, writable}` decisions.
    `structure`/`promotions` ("on"/"off") switch structural installs; observations always run.
    """
    _preflight(cfg)
    require(support_policy in SUPPORT_POLICIES, f"support policy must be one of {SUPPORT_POLICIES}")
    require((support_policy == "replay") == (replay_supports is not None),
            "the replay policy needs, and only it accepts, a recorded schedule")
    require(structure in ("on", "off") and promotions in ("on", "off"),
            "structure and promotions are 'on' or 'off'", structure=structure, promotions=promotions)
    structural.validate_schedule(cfg)
    effective_promotions = promotions if structure == "on" else "off"  # structure off overrides promotions
    runtime = log_runtime()
    tasks = build_split_cifar_tasks(
        cfg, limit=tasks_limit, validation_fraction=VALIDATION_FRACTION,
        augment=AUGMENT, eval_batch_size=EVAL_BATCH_SIZE,
    )
    require(all(task.train_loader.passes == 0 for task in tasks), "task construction consumed a fit pass")
    log_progress(f"loaded {len(tasks)} Split-CIFAR-10 tasks", component="runner")
    if replay_supports is not None:
        replay_supports = {int(t): dict(r) for t, r in replay_supports.items()}
        validate_replay(cfg.column_pool, replay_supports, [task.task_id for task in tasks])
    label = condition_label(support_policy, structure, promotions)
    run_id = run_id or _run_name(cfg, support_policy, structure, promotions)
    cfg, run_output_root = prepare_run_root(cfg, run_id)
    graph, trainer = build_trainer(
        cfg, tasks, "backprop", run_id=run_id,
        experiment_metadata={
            "seed": int(cfg.seed),
            "condition_label": label,
            "promotion_representation": PROMOTION_REPRESENTATION,
            "support_policy": support_policy,
            "structure": structure,
            "promotions": effective_promotions,
            "promotions_requested": promotions,
            "validation_fraction": VALIDATION_FRACTION,
            "augment": AUGMENT,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "shared_k_connectivity": SHARED_K_CONNECTIVITY,
            "task_seeds": {task.task_id: task_seeds(cfg.seed, task.task_id) for task in tasks},
        },
        shared_k_connectivity=SHARED_K_CONNECTIVITY,
    )
    # Kernel-only changes no shape or count, so assert it rather than trust the default.
    require(graph.config["hibacaml"]["shared_k_connectivity"] == SHARED_K_CONNECTIVITY,
            "shared K connectivity does not match the profile")
    # Roles before T0: certificates use role masks from the first refresh.
    trainer.persistent_state.shell_roles = validate_roles(graph, cfg, initial_roles(graph, cfg))
    layout = PolicyLayout.from_structure(graph, cfg)
    selector = BoundarySelector(cfg, layout, support_policy, replay_supports)
    controller = structural.StructuralController(cfg, layout, task_count=len(tasks), structure=structure,
                                                 promotions=promotions)  # the request; it applies the override
    pool = cfg.column_pool
    references = column_fingerprints(trainer.params, pool.adaptive_indices + pool.reserve_indices)
    per_task = [len(task.train_loader) * cfg.epochs_per_task for task in tasks]
    trial_bound = (trial_update_bound(pool.topk_nonshared, [len(task.train_loader) for task in tasks])
                   if support_policy == "restricted_reuse" else 0)
    extra = {
        "status": "running",
        "seed": int(cfg.seed),
        "condition_label": label,
        "promotion_representation": PROMOTION_REPRESENTATION,
        "support_policy": support_policy,
        "structure": structure,
        "promotions": effective_promotions,
        "runtime": runtime,
        "architecture": architecture_metadata(cfg, graph, trainer.params),
        "schedule": {
            "epochs_per_task": cfg.epochs_per_task,
            "batch_size": cfg.batch_size,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "updates_per_task": per_task,
            "expected_total_updates": sum(per_task),
        },
        "boundaries": [],
    }
    guard: Dict[tuple, str] = {}  # protected leaf -> byte digest; never refreshed once captured

    def check_guard(params, **details):
        changed = changed_leaves(params, guard)
        require(not changed, "a protected leaf changed", leaves=changed, **details)

    active: Dict[str, object] = {}  # the committed condition of the task being trained

    def hook(t, task, epoch):
        """Guard, scheduled structural round, guard, external callback, guard; validation follows."""
        check_guard(t.params, task_id=task.task_id, epoch=epoch)
        kind = controller.due(task.task_id, epoch)
        if kind is not None:
            controller.round(t, task, epoch, condition=active["condition"], history=selector.allocator.history,
                             guard=guard, boundary=kind == "boundary")
            check_guard(t.params, task_id=task.task_id, epoch=epoch, stage="post_round")
        if on_epoch_end is not None:
            on_epoch_end(t, task, epoch)
            check_guard(t.params, task_id=task.task_id, epoch=epoch)

    matrices = {"test": {}, "validation": {}}
    task_summaries: List[Dict[str, object]] = []
    completed = 0
    try:
        for task in tasks:
            tid = task.task_id
            require(task.train_loader.passes == 0, "a future fit stream was consumed", task_id=tid)
            if tid >= 1 and not guard:  # T0 stem, composer maps and shared columns freeze before T1 selection
                guard.update(shared_references(trainer, layout))
                extra["protection_activation_step"] = trainer.persistent_state.global_step + 1
                log_progress(f"post-T0 protection active from step {extra['protection_activation_step']} "
                             f"protected_leaves={len(guard)}", component="runner")
            condition = selector.select(trainer, task, guard)
            selector.commit(trainer, tid, condition, guard)
            active["condition"], own = condition, condition.support
            log_progress(f"task={tid} train start support={own}", component="runner")
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
            # Optimizer updates, parameter edits and role changes are counted separately.
            require(counts == dict(global_step=completed, adam_count=completed,
                                   params_revision=completed + controller.functional_installs,
                                   roles_revision=controller.relabel_pairs, train_passes=cfg.epochs_per_task),
                    "boundary counts", task_id=tid, counts=counts)
            require(not state.recently_demoted, "support changes must not record dropped columns")
            history = selector.allocator.history
            require({i: s.nonshared for i, s in state.task_support_snapshots.items()} == history,
                    "saved supports differ from the committed allocation")
            current = column_fingerprints(trainer.params, references)
            changed = sorted(c for c in references if current[c] != references[c])
            require(set(changed) <= set(own), "a column outside this task's support changed",
                    task_id=tid, changed=changed)
            references = current  # the next task is checked against this boundary
            check_guard(trainer.params, task_id=tid)
            log_progress(f"task={tid} checks passed counts={counts} changed_columns={changed} "
                         f"protected_leaves={len(guard)}", component="runner")

            # Boundary evaluation: test from the completion pass, validation without moving the RNG.
            tests = trainer.completion_reports
            require(set(tests) == set(range(tid + 1)), "completion reports must cover every seen task")
            reports = {"test": {j: tests[j] for j in range(tid + 1)}, "validation": {}}
            with preserve_reporting_state(trainer):
                for seen in tasks[: tid + 1]:
                    reports["validation"][seen.task_id] = trainer.evaluate_loader(
                        seen, seen.validation_loader, history[seen.task_id], split_name="validation",
                        mode="saved", refresh_certificates=False,
                    )
            for split, by_task in reports.items():
                for j, report in by_task.items():
                    require(report["max_selected_columns"] <= cfg.composer.topk,
                            "composer selected more than top-k columns")
                matrices[split][tid] = {j: report["accuracy"] for j, report in by_task.items()}
            controller.observe_boundary(tid, {j: dict(r, support=list(history[j]))
                                              for j, r in reports["validation"].items()})
            task_summaries.append(task_summary_row(summary))
            extra["boundaries"].append({
                "task_id": tid,
                "support": own,
                "counts": counts,
                "changed_columns": changed,
                "reports": {split: {j: {k: v for k, v in r.items() if k != "examples"} for j, r in by_task.items()}
                            for split, by_task in reports.items()},
            })
            extra["support_selection"] = selector.summary(main_updates=state.global_step,
                                                          trial_update_upper_bound=trial_bound)
            extra["structural_control"] = controller.summary()
            results, rows = _records(trainer, matrices, task_summaries, extra)
            require_finite((trainer.params, trainer.opt_state), label=f"task {tid} model state")
            require_finite((results, state.epoch_evaluations, trainer.clipping_epoch_summaries),
                           label=f"task {tid} results")
            write_run_records(run_output_root, results, rows,
                              clipping_epoch_summaries=trainer.clipping_epoch_summaries)
            log_progress(f"task={tid} train done accuracy={summary.accuracy:.4f} "
                         f"support={summary.support_indices}", component="runner")
    except BaseException:
        # Executed work stays visible even when a check or a trial fails; best effort, so a logging
        # failure never replaces the original error, which always propagates.
        with contextlib.suppress(Exception):
            log_progress(f"run failed main_updates={trainer.persistent_state.global_step} "
                         f"accounting={selector.accounting} structural={controller.summary()['totals']}",
                         component="runner")
        raise

    expected_rounds = sum(controller.due(task.task_id, epoch) is not None
                          for task in tasks for epoch in range(1, cfg.epochs_per_task + 1))
    require(controller.counts["rounds"] == expected_rounds, "structural round count",
            expected=expected_rounds, actual=controller.counts["rounds"])
    require(selector.accounting["trial_updates"] <= trial_bound, "trial update budget exceeded")
    extra["status"] = "complete" if len(tasks) == cfg.num_tasks else "prefix_complete"
    extra["support_selection"] = selector.summary(main_updates=trainer.persistent_state.global_step,
                                                  trial_update_upper_bound=trial_bound)
    extra["structural_control"] = controller.summary()
    results, rows = _records(trainer, matrices, task_summaries, extra)
    write_run_records(run_output_root, results, rows, clipping_epoch_summaries=trainer.clipping_epoch_summaries)
    plots_dir = run_output_root / "plots"
    plot_sequential_run(results, plots_dir)
    log_progress(f"run complete id={run_id} dir={run_output_root} status={extra['status']} "
                 f"accounting={extra['support_selection']['accounting']}", component="runner")
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
    args = parser.parse_args(argv)
    require_production_runtime()
    cfg = override(make_hibacaml_config(MODE), **{**OVERRIDES, "seed": args.seed})
    require((SUPPORT_POLICY == "replay") == (REPLAY_SOURCE is not None),
            "the replay policy needs, and only it accepts, REPLAY_SOURCE")
    replay = load_replay(REPLAY_SOURCE, cfg.column_pool, cfg.num_tasks) if REPLAY_SOURCE is not None else None
    return run_experiment(cfg, support_policy=SUPPORT_POLICY, replay_supports=replay,
                          structure=STRUCTURE, promotions=PROMOTIONS)


if __name__ == "__main__":
    main()
