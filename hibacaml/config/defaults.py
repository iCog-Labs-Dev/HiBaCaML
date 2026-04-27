"""Default configs for the FabricPC-native HiBaCaML subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

from hibacaml.types import PhiLike


@dataclass(frozen=True)
class ColumnPoolConfig:
    """Column pool partition and dimensions."""

    total_columns: int = 20
    shared_count: int = 2
    adaptive_count: int = 15
    reserve_count: int = 3
    topk_nonshared: int = 3
    memory_dim: int = 8
    shell_sizes: Tuple[int, int, int] = (4, 6, 8)
    kernel_depth: int = 2

    @property
    def shared_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.shared_count))

    @property
    def adaptive_indices(self) -> Tuple[int, ...]:
        start = self.shared_count
        end = start + self.adaptive_count
        return tuple(range(start, end))

    @property
    def reserve_indices(self) -> Tuple[int, ...]:
        start = self.shared_count + self.adaptive_count
        end = start + self.reserve_count
        return tuple(range(start, end))

    @property
    def active_support_size(self) -> int:
        return self.shared_count + self.topk_nonshared

    @property
    def shell_output_dim(self) -> int:
        return self.memory_dim + sum(self.shell_sizes)


@dataclass(frozen=True)
class PhiConfig(PhiLike):
    """Low-dimensional shell controller values."""

    outer_quantile: float = 0.18
    middle_quantile: float = 0.08
    replacement_margin_base: float = 0.08
    demotion_min_role_gain: float = 0.02


@dataclass(frozen=True)
class HierarchyConfig:
    """Auxiliary hierarchy head settings."""

    mid_targets: int = 4
    mid_loss_weight: float = 0.06
    global_loss_weight: float = 0.03
    parent_child_loss_weight: float = 0.02


@dataclass(frozen=True)
class ComposerConfig:
    """Stage-2 composer settings."""

    hidden_dim: int = 64
    query_dim: int = 5
    query_score_scale: float = 1.0
    active_top_k: int = 5
    cert_prior_scale: float = 0.25
    gate_entropy_weight: float = 0.002
    gate_deviation_weight: float = 0.01
    residual_gate_scale: float = 0.42


@dataclass(frozen=True)
class ExactSearchConfig:
    """Boundary and local support search settings."""

    enable_exact_search: bool = True
    boundary_current_batches: int = 2
    boundary_old_batches: int = 1
    boundary_rollout_steps: int = 4
    boundary_shortlist: int = 3
    static_support_batch_size: int = 64
    exact_old_worst_weight: float = 1.0
    exact_old_mix_weight: float = 0.5
    switch_penalty: float = 0.015
    controller_l1_penalty: float = 0.01
    local_swap_margin: float = 0.005
    maintenance_interval: int = 64
    cache_evaluations: bool = True
    outer_quantile_bounds: Tuple[float, float] = (0.05, 0.30)
    middle_quantile_bounds: Tuple[float, float] = (0.02, 0.15)
    replacement_margin_bounds: Tuple[float, float] = (0.04, 0.18)
    demotion_gain_bounds: Tuple[float, float] = (0.005, 0.08)
    semantic_targets: Tuple[float, float, float] = (1.0, 0.90, 0.66)
    log_semantic_penalty: bool = True


@dataclass(frozen=True)
class ReportingConfig:
    """Export and checkpoint settings."""

    output_root: str = "runs/hibacaml"
    checkpoint_filename: str = "hibacaml_checkpoint.pkl"
    event_log_filename: str = "events.jsonl"
    heartbeat_filename: str = "heartbeat.json"
    progress_filename: str = "progress.json"
    memory_samples_filename: str = "memory_samples.csv"
    phase_timings_filename: str = "phase_timings.json"
    task_summaries_filename: str = "task_summaries.json"
    enable_deep_logging: bool = True


@dataclass(frozen=True)
class HiBaCaMLConfig:
    """Top-level subsystem configuration."""

    mode: str = "full"
    seed: int = 0
    input_shape: Tuple[int, int, int] = (28, 28, 1)
    patch_size: Tuple[int, int] = (7, 7)
    patch_embed_dim: int = 12
    patch_coord_dim: int = 2
    task_local_heads: bool = True
    num_tasks: int = 5
    batch_size: int = 128
    epochs_per_task: int = 5
    infer_steps: int = 12
    eta_infer: float = 0.05
    optimizer_lr: float = 0.001
    weight_decay: float = 0.05
    cert_refresh_interval: int = 32
    train_batches_limit: int | None = None
    test_batches_limit: int | None = None
    column_pool: ColumnPoolConfig = field(default_factory=ColumnPoolConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    composer: ComposerConfig = field(default_factory=ComposerConfig)
    exact_search: ExactSearchConfig = field(default_factory=ExactSearchConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @property
    def output_dim(self) -> int:
        return 2 if self.task_local_heads else 10

    @property
    def num_patches(self) -> int:
        h, w, _ = self.input_shape
        ph, pw = self.patch_size
        return (h // ph) * (w // pw)

    @property
    def patch_token_dim(self) -> int:
        return self.patch_embed_dim + self.patch_coord_dim

    @property
    def support_vector_dim(self) -> int:
        return self.column_pool.total_columns

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def output_root_path(self) -> Path:
        return Path(self.reporting.output_root)


def make_hibacaml_config(mode: str = "full") -> HiBaCaMLConfig:
    """Construct a HiBaCaML config for the given mode."""
    if mode in {"full", "reference"}:
        return HiBaCaMLConfig(mode=mode)

    if mode == "compute_debug":
        return HiBaCaMLConfig(
            mode=mode,
            batch_size=64,
            epochs_per_task=1,
            infer_steps=6,
            cert_refresh_interval=16,
            train_batches_limit=8,
            test_batches_limit=4,
            exact_search=ExactSearchConfig(
                enable_exact_search=True,
                boundary_current_batches=1,
                boundary_old_batches=1,
                boundary_rollout_steps=1,
                boundary_shortlist=3,
                static_support_batch_size=128,
                maintenance_interval=32,
            ),
        )

    if mode == "smoke":
        return HiBaCaMLConfig(
            mode=mode,
            batch_size=32,
            epochs_per_task=1,
            infer_steps=4,
            column_pool=ColumnPoolConfig(
                memory_dim=4,
                shell_sizes=(1, 1, 1),
                kernel_depth=2,
            ),
            patch_embed_dim=8,
            cert_refresh_interval=1,
            train_batches_limit=1,
            test_batches_limit=1,
            composer=ComposerConfig(
                hidden_dim=16,
                query_dim=5,
                residual_gate_scale=0.25,
            ),
            exact_search=ExactSearchConfig(
                boundary_current_batches=1,
                boundary_old_batches=1,
                boundary_rollout_steps=1,
                boundary_shortlist=3,
                static_support_batch_size=64,
                maintenance_interval=2,
            ),
        )

    raise ValueError(f"Unsupported HiBaCaML config mode: {mode}")
