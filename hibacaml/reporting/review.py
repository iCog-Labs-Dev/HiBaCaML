"""Pre-run summary and confirmation for HiBaCaML experiments."""

from __future__ import annotations

from typing import Sequence

from hibacaml.config import HiBaCaMLConfig


def _cap_label(value: int | None) -> str:
    return "full" if value is None else str(value)


def print_pre_run_review(
    cfg: HiBaCaMLConfig,
    structure,
    tasks: Sequence,
    learning: str,
    *,
    confirm: bool = True,
) -> None:
    """Print a pre-run summary and optionally wait for confirmation."""
    sep = "=" * 72
    phi_candidates = 1 + 2 * 4
    learning_label = {
        "backprop": "predictive-coding inference + backprop",
        "pc": "predictive-coding inference + local PC gradients",
    }.get(learning, learning)

    print(f"\n{sep}")
    print("  HiBaCaML Pre-Run Summary")
    print(sep)

    print("\n[1] Config")
    print(f"    mode              : {cfg.mode}")
    print(f"    learning          : {learning_label}")
    print(f"    batch_size        : {cfg.batch_size}")
    print(
        "    search batch caps : "
        f"current={_cap_label(cfg.exact_search.boundary_current_data_batch_size)}, "
        f"rollout={_cap_label(cfg.exact_search.rollout_train_data_batch_size)}, "
        f"worst_old={_cap_label(cfg.exact_search.boundary_worst_old_data_batch_size)}, "
        f"mixed_old={_cap_label(cfg.exact_search.boundary_mixed_old_data_batch_size)}, "
        f"local_swap={_cap_label(cfg.exact_search.local_swap_audit_data_batch_size)}, "
        f"demotion={_cap_label(cfg.exact_search.demotion_audit_data_batch_size)}"
    )
    print(f"    epochs_per_task   : {cfg.epochs_per_task}")
    print(f"    infer_steps       : {cfg.infer_steps}")
    print(f"    eta_infer         : {cfg.eta_infer}")
    print(f"    shortlist         : {cfg.exact_search.boundary_shortlist}")
    print(f"    rollout steps     : {cfg.exact_search.boundary_rollout_steps}")
    print(f"    rollout gradients : {cfg.exact_search.rollout_gradient_mode}")
    print(
        "    rollout evals     : "
        f"{cfg.exact_search.boundary_shortlist * phi_candidates}  "
        f"({cfg.exact_search.boundary_shortlist} shortlist x {phi_candidates} phi candidates)"
    )
    print(
        "    boundary weights  : "
        f"lambda_worst={cfg.exact_search.exact_old_worst_weight}, "
        f"lambda_mix={cfg.exact_search.exact_old_mix_weight}, "
        f"lambda_switch={cfg.exact_search.switch_penalty}"
    )
    print(
        "    phi defaults      : "
        f"({cfg.phi.outer_quantile:.3f}, {cfg.phi.middle_quantile:.3f}, "
        f"{cfg.phi.replacement_margin_base:.3f}, {cfg.phi.demotion_min_role_gain:.3f})"
    )
    print(f"    optimizer_lr      : {cfg.optimizer_lr}")
    print(f"    weight_decay      : {cfg.weight_decay}")
    print(f"    seed              : {cfg.seed}")
    print(f"    experiment_root   : {cfg.reporting.experiment_root}")
    print(f"    selector_state    : {cfg.reporting.selector_state_root}")
    print(
        "    reselection       : "
        f"replay_topk={cfg.exact_search.replay_topk}, "
        f"overlap_floor={cfg.exact_search.replay_overlap_floor}, "
        f"alpha={cfg.exact_search.replay_overlap_penalty_alpha}, "
        f"beta={cfg.exact_search.replay_jump_penalty_beta}, "
        f"gamma={cfg.exact_search.replay_history_penalty_gamma}"
    )

    print("\n[2] Architecture")
    cp = cfg.column_pool
    print(f"    total nodes       : {len(structure.nodes)}")
    print(f"    columns           : {cp.total_columns} total")
    print(
        f"    partition         : {cp.shared_count} shared, "
        f"{cp.adaptive_count} adaptive, {cp.reserve_count} reserve"
    )
    print(f"    active nonshared  : {cp.topk_nonshared}")
    print(
        f"    shell sizes       : kernel={cp.memory_dim}, "
        f"kernel_depth={cp.kernel_depth}, tiers={cp.shell_sizes}"
    )
    print(f"    patch tokens      : {cfg.num_patches} x {cfg.patch_token_dim}")
    print(
        f"    hierarchy heads   : mid={cfg.hierarchy.mid_targets}, "
        f"global={cfg.output_dim}, query_dim={cfg.composer.query_dim}"
    )

    print("\n[3] Task Setting")
    print(f"    num tasks         : {len(tasks)}")
    for task in tasks:
        print(f"    task {task.task_id:<2}            : classes={task.classes}")

    print("\n[4] Algorithm Reference")
    print("    boundary objective : shared by shortlist, rollout, and one-swap (current + old-worst + old-mix + switch)")
    print("    shell controller   : 4-parameter phi (outer/middle quantile, replacement margin, demotion gain)")
    print("    support search     : exact enumeration -> shortlist -> controller rollout -> one-swap maintenance")
    print(
        "    runtime controls   : "
        "local_maintenance=on, pad_support_batches=on"
    )
    print(
        "    paper controls     : "
        f"cert_weight={cfg.exact_search.certificate_support_weight}, "
        f"precision_resistance={cfg.exact_search.enable_precision_update_resistance}, "
        f"demotion_audit={cfg.exact_search.enable_demotion_swap_audit}, "
        f"reserve_gate=({cfg.exact_search.reserve_saturation_threshold}, "
        f"{cfg.exact_search.reserve_entropy_threshold}, "
        f"{cfg.exact_search.reserve_top1_prob_threshold})"
    )
    print(f"    learning           : {learning_label}")
    print("    column graph       : 20-column K/L/B graph, shell edits, stage-2 composer, hierarchy heads")
    print("    occupancy          : semantic occupancy regularizer and conservative one-swap audits")
    print("    data               : Split-MNIST tasks (0,1)...(8,9) with 28x28 -> 16 patch tokens")
    print("    exports            : support tables, controller tables, swap tables, provenance artifacts")

    print("\n[5] Implementation Notes")
    print("    hierarchy targets : deterministic quadrant-sensitive task-local targets")
    print("    controller search : bounded local phi neighborhood around the current controller")
    print("    artifacts         : per-run tables and metrics written under experiment_root")

    print(f"\n{sep}")
    if confirm:
        try:
            input("  Press Enter to begin training, or Ctrl-C to abort... ")
        except KeyboardInterrupt:
            print("\n  Aborted by user.")
            raise SystemExit(1)
    else:
        print("  Auto-proceeding (--no-confirm).")
    print(sep + "\n")
