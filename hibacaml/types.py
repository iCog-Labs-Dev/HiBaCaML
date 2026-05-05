"""Public data types for HiBaCaML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp

from fabricpc.core.types import GraphParams


@dataclass(frozen=True)
class SplitMnistTask:
    """Task-local Split-MNIST loaders and metadata."""

    task_id: int
    classes: Tuple[int, int]
    train_loader: Any
    test_loader: Any
    task_query: jnp.ndarray
    output_dim: int


@dataclass(frozen=True)
class BoundaryBundle:
    """Audit bundle used for exact-search boundary scoring."""

    current_eval: Tuple[Dict[str, jnp.ndarray], ...]
    train_batches: Tuple[Dict[str, jnp.ndarray], ...]
    worst_old_eval: Optional[Tuple[int, Dict[str, jnp.ndarray]]] = None
    mixed_old_eval: Tuple[Tuple[int, Dict[str, jnp.ndarray]], ...] = ()
    worst_old: Optional[int] = None


@dataclass(frozen=True)
class SupportSnapshot:
    """Frozen support assignment for a completed task."""

    task_id: int
    nonshared: Tuple[int, ...]
    full_support: Tuple[int, ...]
    phi: "PhiLike"
    global_step: int


@dataclass(frozen=True)
class SupportSearchRow:
    """Static exact-search row for a candidate support."""

    task_id: int
    nonshared: Tuple[int, ...]
    static_total: float
    current_first_loss: float
    current_remaining_loss: float
    old_worst_loss: float
    old_mix_loss: float
    switch_penalty: float
    shortlist_rank: int = -1
    posterior_energy: float = 0.0
    support_log_prob: float = 0.0
    support_prob: float = 0.0
    posterior_rank: int = -1
    posterior_entropy: float = 0.0
    top1_prob: float = 0.0
    certificate_reuse_score: float = 0.0
    reserve_recruitment_candidate: bool = False


@dataclass(frozen=True)
class ControllerSearchRow:
    """Rollout-scored support/controller row."""

    task_id: int
    nonshared: Tuple[int, ...]
    phi: "PhiLike"
    rollout_total: float
    boundary_total: float
    current_loss: float
    old_worst_loss: float
    old_mix_loss: float
    switch_penalty: float
    phi_l1_penalty: float = 0.0
    semantic_regularizer: float = 0.0
    gate_entropy: float = 0.0
    gate_deviation: float = 0.0
    cert_prior_mean: float = 0.0
    occ_tier1: float = 0.0
    occ_tier2: float = 0.0
    occ_tier3: float = 0.0
    q_tier1: float = 0.0
    q_tier2: float = 0.0
    q_tier3: float = 0.0


@dataclass(frozen=True)
class LocalSwapRow:
    """One-swap local maintenance audit row."""

    task_id: int
    round_index: int
    current_nonshared: Tuple[int, ...]
    candidate_nonshared: Tuple[int, ...]
    total: float
    boundary_total: float = 0.0
    current_loss: float = 0.0
    old_worst_loss: float = 0.0
    old_mix_loss: float = 0.0
    switch_penalty: float = 0.0
    semantic_regularizer: float = 0.0
    gain: float = 0.0
    accepted: bool = False


@dataclass(frozen=True)
class SupportPosteriorSummary:
    """Per-boundary posterior-style summary over audited supports."""

    task_id: int
    candidate_count: int
    posterior_entropy: float
    top1_prob: float
    map_nonshared: Tuple[int, ...]
    map_posterior_energy: float
    reserve_recruitment_triggered: bool = False


@dataclass(frozen=True)
class ReserveRecruitmentRow:
    """Why reserve candidates were or were not admitted into support audit."""

    task_id: int
    triggered: bool
    adaptive_candidate_count: int
    reserve_candidate_count: int
    saturation: float
    posterior_entropy: float
    top1_prob: float
    saturation_threshold: float
    entropy_threshold: float
    top1_prob_threshold: float
    reason: str


@dataclass(frozen=True)
class DemotionSwapAuditRow:
    """Audited internal demotion-swap candidate."""

    task_id: int
    round_index: int
    global_step: int
    column_index: int
    node_name: str
    inner_shell: str
    outer_shell: str
    inner_index: int
    outer_index: int
    baseline_total: float
    candidate_total: float
    gain: float
    accepted: bool = False


@dataclass(frozen=True)
class ReplayProposalRow:
    """V20.2b reselection audit row.

    Captures the three streams considered at every support-acceptance decision
    (boundary or local one-swap): the exact-search winner, the local one-hop
    refinement baseline, and the best replay-bank candidate. Plus the penalised
    scores, overlap diagnostics, and which stream was finally accepted.
    """

    task_id: int
    global_step: int
    provenance: str  # "boundary" | "local_swap"
    original_nonshared: Tuple[int, ...]
    local_baseline_nonshared: Tuple[int, ...]
    replay_candidate_nonshared: Tuple[int, ...]
    original_total: float
    local_total: float
    replay_total: float
    original_penalised: float
    local_penalised: float
    replay_penalised: float
    overlap_jaccard: float
    jump_size: int
    history_intersection: int
    accepted_source: str  # "original" | "local" | "replay"
    accepted_nonshared: Tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class TaskSummary:
    """Per-task summary emitted after training/evaluation."""

    task_id: int
    classes: Tuple[int, int]
    support_indices: Tuple[int, ...]
    accuracy: float
    mean_loss: float
    best_old_accuracy: float
    support_entropy: float


@dataclass
class ColumnCertificate:
    """Compact column-level certificate logged for structural analysis."""

    column_index: int
    q_mean: float
    prec_mean: float
    pred_mean: float
    live_frac: float
    tier_q: Tuple[float, float, float]
    tier_occ: Tuple[float, float, float]
    shared_abstraction_mass: float
    specificity_load: float
    demotion_pressure: float
    saturation: float
    similarity_signature: Tuple[float, ...] = field(default_factory=tuple)


@dataclass
class ShellStats:
    """Running shell-side statistics used by the shell controller."""

    activation_ema: Dict[str, float] = field(default_factory=dict)
    task_variance_ema: Dict[str, float] = field(default_factory=dict)
    reuse_ema: float = 0.0
    specificity_ema: float = 0.0


@dataclass
class PersistentHiBaCaMLState:
    """Persistent experiment state saved with checkpoints and exports."""

    current_support: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    boundary_support: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    task_support_snapshots: Dict[int, SupportSnapshot] = field(default_factory=dict)
    last_maintenance_step: Dict[int, int] = field(default_factory=dict)
    support_tables: Dict[int, List[SupportSearchRow]] = field(default_factory=dict)
    support_posterior_tables: Dict[int, SupportPosteriorSummary] = field(default_factory=dict)
    reserve_recruitment_tables: Dict[int, List[ReserveRecruitmentRow]] = field(default_factory=dict)
    controller_tables: Dict[int, List[ControllerSearchRow]] = field(default_factory=dict)
    local_swap_tables: Dict[int, List[LocalSwapRow]] = field(default_factory=dict)
    demotion_swap_tables: Dict[int, List[DemotionSwapAuditRow]] = field(default_factory=dict)
    replay_proposals: Dict[int, List[ReplayProposalRow]] = field(default_factory=dict)
    recently_demoted: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    latest_local_swap: Optional[LocalSwapRow] = None
    last_demotion_audit_step: Dict[int, int] = field(default_factory=dict)
    composer_diagnostics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    certificates: Dict[int, ColumnCertificate] = field(default_factory=dict)
    shell_stats: Dict[int, ShellStats] = field(default_factory=dict)
    global_step: int = 0
    params_revision: int = 0
    timing_summaries: Dict[int, Dict[str, float]] = field(default_factory=dict)
    params: Optional[GraphParams] = None
    opt_state: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Best-effort conversion for JSON export and tests."""
        return asdict(self)


@dataclass(frozen=True)
class PhiLike:
    """Public phi value object shared across configs and search rows."""

    outer_quantile: float
    middle_quantile: float
    replacement_margin_base: float
    demotion_min_role_gain: float
