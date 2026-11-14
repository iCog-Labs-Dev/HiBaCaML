"""Audited internal structural control for Split-CIFAR (guide §9).

From T1, scheduled rounds in the epoch hook compare an unchanged `no_edit`
evaluation of every seen validation task with three mechanisms:

- promotion/demotion by relabelling adjacent shell tiers (function preserving);
- same-tier inhibition of labelled shell gates;
- recoverable pruning of weak fresh-column units.

Functional edits are sparse parameter/moment patches, audited on isolated clones
with the support teacher's objective and bootstrap, and capped per edit and
cumulatively on every old task. Rounds run no optimizer update.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from hibacaml.control.certificates import shell_slices
from hibacaml.control.reuse import (
    CE_TOLERANCE,
    HARM_CAP_PERCENT,
    check_pairing,
    collect_validation,
    decide,
    harm_check,
    score_family,
    teacher_objective,
)
from hibacaml.control.roles import (
    REGIONS,
    evidence_pairs,
    inhibited_log_precision,
    node_arrays,
    prune_trigger,
    relabel,
    roles_digest,
    slot_region,
    unit_scale_table,
    unit_usage,
    validate_roles,
)
from hibacaml.reporting.integrity import (
    changed_leaves,
    fingerprint,
    learning_state_signature,
    preserve_reporting_state,
    require,
)
from hibacaml.reporting.logger import log_progress
from hibacaml.training.trainer import adam_count, adam_state, with_moments

# Schedule (read through this module's globals at call time, so tests may shorten it).
AUDIT_EPOCHS = (5, 10, 15, 20, 25)
PROMOTION_START_EPOCH = 10
PRUNE_START_EPOCH = 10
ROUND_OUTER_LABEL = 3      # tier 3 -> 2 promotions in audit rounds
BOUNDARY_OUTER_LABEL = 2   # tier 2 -> 1 promotions in the final-epoch handoff round
PROMOTION_MIN_CONTEXTS = 2
INHIBITION_STRENGTHS = (0.30, 0.18, 0.08)
INHIBITION_DOSE = 10.0
INHIBITION_FLOOR = 0.25
PRUNE_WEAK_FRACTION = 0.10
PRUNE_SCORE_QUANTILE = 0.125
PRUNE_LOG_PRECISION = -4.0
MAX_RAW_PROPOSALS = 16     # fail-fast assertion, never truncation (production reaches 15)
USAGE_TAG = 0x55534147
PRUNE_PROBE_TAG = 0x50524F42
BOOTSTRAP_TAG = 0x424F4F54


def round_seed(*entropy: int) -> int:
    return int(np.random.SeedSequence([int(e) for e in entropy]).generate_state(1, dtype=np.uint64)[0])


def validate_schedule(cfg) -> None:
    require(all(1 <= e < cfg.epochs_per_task for e in AUDIT_EPOCHS),
            "audit epochs must precede the final epoch", audit_epochs=AUDIT_EPOCHS)


def round_kind(task_id: int, epoch: int, *, epochs_per_task: int, task_count: int) -> Optional[str]:
    """`"round"` at audit epochs from T1; `"boundary"` at the final epoch of T1 up to the
    second-to-last task of this run (a prefix's terminal task gets no handoff round)."""
    if task_id < 1:
        return None
    if epoch in AUDIT_EPOCHS:
        return "round"
    if epoch == epochs_per_task and task_id < task_count - 1:
        return "boundary"
    return None


def promotion_outer_label(epoch: int, kind: str) -> Optional[int]:
    if kind == "boundary":
        return BOUNDARY_OUTER_LABEL
    return ROUND_OUTER_LABEL if epoch >= PROMOTION_START_EPOCH else None


def prune_labels(epoch: int) -> Tuple[int, ...]:
    """Label 3 from the prune start epoch; label 2 as well at the last audit epoch."""
    return ((3,) if epoch >= PRUNE_START_EPOCH else ()) + ((2,) if epoch == max(AUDIT_EPOCHS) else ())


# ---- sparse parameter/moment patches ---------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One functional edit: label inhibition of a column's B/K gates, or a slot prune."""

    family: str            # S-INH | R-INH | F-INH | F-PRUNE
    column: int
    operation: str         # inhibit | prune
    node: Optional[str] = None
    label: Optional[int] = None
    slots: Tuple[int, ...] = ()


@dataclass(frozen=True)
class LeafEdit:
    tree: str              # params | mu | nu
    node: str
    field: str
    key: str
    shape: Tuple[int, ...]
    indices: np.ndarray
    before: np.ndarray
    after: np.ndarray


@dataclass(frozen=True)
class Patch:
    """A proposal's exact sparse edit, bound to the parent it was built from."""

    proposal: Proposal
    params_fingerprint: str
    optimizer_fingerprint: str
    roles_digest: str
    edits: Tuple[LeafEdit, ...] = field(default_factory=tuple)

    @property
    def content_fingerprint(self) -> str:
        return fingerprint(self.edits)


def _replace_leaf(tree, node, group, key, value):
    params = tree.nodes[node]
    leaves = {**getattr(params, group), key: jnp.asarray(value)}
    return tree._replace(nodes={**tree.nodes, node: params._replace(**{group: leaves})})


def column_nodes(layout, column: int, *, include_l: bool) -> List[str]:
    """The column's shell nodes in B, K0.., L order (L only when asked)."""
    positions = sorted((pos, name) for name, (col, pos) in layout.shells.items() if col == column)
    last = max(pos for pos, _ in positions)
    return [name for pos, name in positions if include_l or pos != last]


def _parent_fingerprints(trainer) -> Tuple[str, str]:
    return fingerprint(trainer.params), fingerprint(trainer.opt_state)


def _inhibited(trainer, layout, proposal: Proposal, params):
    require(proposal.node is None and proposal.label is None and not proposal.slots,
            "inhibition covers whole columns", proposal=proposal)
    regions = shell_slices(trainer.cfg)
    roles = trainer.persistent_state.shell_roles
    for name in column_nodes(layout, proposal.column, include_l=False):  # B and every K depth
        weights, log_precision = node_arrays(params.nodes[name])
        after = inhibited_log_precision(weights, log_precision, roles[name], strengths=INHIBITION_STRENGTHS,
                                        dose=INHIBITION_DOSE, floor=INHIBITION_FLOOR)
        for region in REGIONS[1:]:  # kernel gates are never written
            key = "log_precision_" + region
            old = np.asarray(params.nodes[name].biases[key])
            new = after[regions[region]].reshape(old.shape)
            if not np.array_equal(new.view(np.uint32), old.view(np.uint32)):
                params = _replace_leaf(params, name, "biases", key, new)
    return params


def _pruned(trainer, layout, proposal: Proposal, tree, *, moments: bool):
    """Zero the slots' incoming columns, biases and recurrent inputs; gates to the prune value."""
    by_region: Dict[str, List[int]] = {}
    for slot in proposal.slots:
        region, index = slot_region(trainer.cfg, int(slot))
        require(region != "kernel", "kernel slots are never pruned", slot=slot)
        by_region.setdefault(region, []).append(index)
    node = tree.nodes[proposal.node]
    for region, indices in by_region.items():
        idx = np.asarray(indices, dtype=np.int64)
        weights = np.asarray(node.weights[region]).copy()
        weights[..., idx] = 0.0
        tree = _replace_leaf(tree, proposal.node, "weights", region, weights)
        bias = np.asarray(node.biases["b_" + region]).copy()
        bias[..., idx] = 0.0
        tree = _replace_leaf(tree, proposal.node, "biases", "b_" + region, bias)
        gate = np.asarray(node.biases["log_precision_" + region]).copy()
        gate[..., idx] = 0.0 if moments else np.float32(PRUNE_LOG_PRECISION)
        tree = _replace_leaf(tree, proposal.node, "biases", "log_precision_" + region, gate)
        if "recur_" + region in node.weights:  # incoming recurrent columns only
            recurrent = np.asarray(node.weights["recur_" + region]).copy()
            recurrent[:, idx] = 0.0
            tree = _replace_leaf(tree, proposal.node, "weights", "recur_" + region, recurrent)
    return tree


def _validate_prune(trainer, layout, proposal: Proposal) -> None:
    labels = np.asarray(trainer.persistent_state.shell_roles.get(proposal.node, ()))
    slots = proposal.slots
    require(
        proposal.node in column_nodes(layout, proposal.column, include_l=True)
        and proposal.label in (2, 3) and len(slots) > 0 and len(set(slots)) == len(slots)
        and all(isinstance(s, (int, np.integer)) and not isinstance(s, bool) and 0 <= s < labels.size
                for s in slots)
        and all(int(labels[s]) == proposal.label for s in slots),
        "invalid prune proposal", proposal=proposal,
    )


def _sparse_edits(before: Dict[str, object], after: Dict[str, object], names: Sequence[str]) -> List[LeafEdit]:
    edits = []
    for tree in ("params", "mu", "nu"):
        for name in names:
            for group in ("weights", "biases"):
                for key, old in sorted(getattr(before[tree].nodes[name], group).items()):
                    x = np.asarray(old)
                    y = np.asarray(getattr(after[tree].nodes[name], group)[key])
                    require(x.shape == y.shape and x.dtype == y.dtype == np.float32 and np.isfinite(y).all(),
                            "invalid patch leaf", leaf=(tree, name, group, key))
                    indices = np.flatnonzero(x.reshape(-1).view(np.uint32) != y.reshape(-1).view(np.uint32))
                    if indices.size:
                        edits.append(LeafEdit(tree, name, group, key, tuple(x.shape), indices,
                                              x.reshape(-1)[indices].copy(), y.reshape(-1)[indices].copy()))
    return edits


def build_patch(trainer, proposal: Proposal, layout, eligible: Sequence[int], *,
                parent_fingerprints: Optional[Tuple[str, str]] = None) -> Patch:
    """The proposal's deterministic sparse edit against the trainer's current state."""
    require(proposal.column in tuple(eligible), "the proposal column is not eligible", proposal=proposal)
    adam = adam_state(trainer.opt_state)
    before = {"params": trainer.params, "mu": adam.mu, "nu": adam.nu}
    if proposal.operation == "inhibit":
        after = dict(before, params=_inhibited(trainer, layout, proposal, trainer.params))
        names = column_nodes(layout, proposal.column, include_l=False)
    elif proposal.operation == "prune":
        _validate_prune(trainer, layout, proposal)
        after = {tree: _pruned(trainer, layout, proposal, value, moments=tree != "params")
                 for tree, value in before.items()}
        names = [proposal.node]
    else:
        raise ValueError(f"unknown structural operation {proposal.operation!r}")
    params_fp, optimizer_fp = parent_fingerprints or _parent_fingerprints(trainer)
    return Patch(proposal, params_fp, optimizer_fp, roles_digest(trainer.persistent_state.shell_roles),
                 tuple(_sparse_edits(before, after, names)))


def apply_patch(trainer, patch: Patch, layout, eligible: Sequence[int]) -> None:
    """Install a validated patch: one pointer swap, one params revision, no optimizer step."""
    parent = _parent_fingerprints(trainer)
    require(parent == (patch.params_fingerprint, patch.optimizer_fingerprint), "stale patch parent")
    require(roles_digest(trainer.persistent_state.shell_roles) == patch.roles_digest, "stale patch roles")
    expected = build_patch(trainer, patch.proposal, layout, eligible, parent_fingerprints=parent)
    require(fingerprint(expected) == fingerprint(patch), "the patch differs from its proposal's edit")
    adam = adam_state(trainer.opt_state)
    trees = {"params": trainer.params, "mu": adam.mu, "nu": adam.nu}
    for edit in patch.edits:
        leaf = np.asarray(getattr(trees[edit.tree].nodes[edit.node], edit.field)[edit.key]).copy()
        flat = leaf.reshape(-1)
        require(np.array_equal(flat[edit.indices].view(np.uint32), edit.before.view(np.uint32)),
                "stale patch leaf", leaf=(edit.tree, edit.node, edit.key))
        flat[edit.indices] = edit.after
        trees[edit.tree] = _replace_leaf(trees[edit.tree], edit.node, edit.field, edit.key, leaf)
    trainer.params = trees["params"]
    trainer.opt_state = with_moments(trainer.opt_state, trees["mu"], trees["nu"])
    trainer.persistent_state.params, trainer.persistent_state.opt_state = trainer.params, trainer.opt_state
    if patch.edits:
        trainer._bump_params_revision()
        if hasattr(trainer, "_last_graph_state"):
            del trainer._last_graph_state  # computed from the replaced parameters


def verify_patch_scope(before: Dict[str, object], trainer, patch: Patch) -> None:
    """Every params/mu/nu byte outside the patch is unchanged; inside, exactly its `after` values."""
    adam = adam_state(trainer.opt_state)
    after = {"params": trainer.params, "mu": adam.mu, "nu": adam.nu}
    allowed = {(e.tree, e.node, e.field, e.key): e for e in patch.edits}
    for tree, old_tree in before.items():
        for node, params in old_tree.nodes.items():
            for group in ("weights", "biases"):
                for key, old in getattr(params, group).items():
                    new = getattr(after[tree].nodes[node], group)[key]
                    edit = allowed.get((tree, node, group, key))
                    if new is old and edit is None:
                        continue
                    x = np.asarray(old).reshape(-1).view(np.uint32)
                    y = np.asarray(new).reshape(-1).view(np.uint32)
                    changed = np.flatnonzero(x != y)
                    if edit is None:
                        require(changed.size == 0, "a patch changed a coordinate outside its scope",
                                leaf=(tree, node, key))
                    else:
                        require(set(changed.tolist()) <= set(edit.indices.tolist())
                                and np.array_equal(y[edit.indices], edit.after.view(np.uint32)),
                                "a patch installed unexpected values", leaf=(tree, node, key))


# ---- cumulative old-task harm ------------------------------------------------------------------


class HarmLedger:
    """Signed net edit-instant losses per old task (never reset), plus training-interval drift.

    Only `record_install` feeds the cumulative acceptance budget; ordinary training
    drift is recorded separately and never enters it.
    """

    def __init__(self):
        self.net_lost: Dict[int, int] = defaultdict(int)
        self.installs: List[Dict[str, object]] = []
        self.intervals: List[Dict[str, object]] = []
        self._open: Optional[Tuple[str, Dict[int, Dict]]] = None

    def cumulative_passes(self, task_id: int, lost: int, examples: int) -> bool:
        return 100 * (self.net_lost[task_id] + lost) <= HARM_CAP_PERCENT * examples

    def record_install(self, label: str, proposal_id: str, current_task: int, comparator, post, predicted) -> None:
        rows = {}
        for j in range(current_task):
            lost = int(comparator[j]["correct_examples"]) - int(post[j]["correct_examples"])
            self.net_lost[j] += lost
            rows[j] = dict(realized_lost_correct=lost, predicted_lost_correct=int(predicted[j]),
                           cumulative_net_lost_correct=self.net_lost[j], examples=int(post[j]["num_examples"]))
        self.installs.append(dict(round_id=label, proposal_id=proposal_id, current_task=current_task, old_tasks=rows))

    def close_interval(self, label: str, reports: Dict[int, Dict]) -> None:
        """Record training drift since the interval opened, for tasks whose support is unchanged."""
        if self._open is not None:
            start, before = self._open
            for j, report in sorted(reports.items()):
                previous = before.get(j)
                if previous is None or list(previous["support"]) != list(report["support"]):
                    continue
                self.intervals.append(dict(
                    task_id=j, start=start, end=label,
                    correct_delta=int(report["correct_examples"]) - int(previous["correct_examples"]),
                    accuracy_delta_pp=100.0 * (report["accuracy"] - previous["accuracy"]),
                    cross_entropy_delta=report["cross_entropy"] - previous["cross_entropy"],
                ))
        self._open = None

    def open_interval(self, label: str, reports: Dict[int, Dict]) -> None:
        self._open = (label, reports)

    def summary(self) -> Dict[str, object]:
        return dict(cap_percent=HARM_CAP_PERCENT, net_lost_correct=dict(self.net_lost),
                    edit_instant=self.installs, training_intervals=self.intervals)


# ---- the controller ------------------------------------------------------------------------------

PROMOTION_FAMILIES = ("S-PROMOTE", "R-PROMOTE")
FUNCTIONAL_FAMILIES = ("S-INH", "R-INH", "F-INH", "F-PRUNE")
_MISSING = object()


class StructuralController:
    """Scheduled audited rounds over the current support; owns dwell, revisions and the harm ledger.

    `structure="off"` still evaluates the comparator, samples usage and records shadow
    promotion nominations; it never installs. `promotions="off"` keeps the functional
    step but records promotions as shadows. Structure off overrides promotions.
    """

    def __init__(self, cfg, layout, *, task_count: int, structure: str = "on", promotions: str = "on"):
        require(structure in ("on", "off") and promotions in ("on", "off"),
                "structure and promotions are 'on' or 'off'", structure=structure, promotions=promotions)
        self.cfg, self.layout, self.task_count = cfg, layout, int(task_count)
        self.structure = structure
        self.promotions_requested = promotions
        self.promotions = promotions if structure == "on" else "off"
        self.shared = tuple(cfg.column_pool.shared_indices)
        self.ledger = HarmLedger()
        self.dwell = set()                      # (task, node, slot) moved by an installed pair
        self.relabel_pairs = 0
        self.functional_installs = 0
        self.counts = dict(rounds=0, evaluation_passes=0, usage_batches=0, prune_batches=0)
        self.seconds = 0.0
        self.rounds: List[Dict[str, object]] = []
        self.proposals: List[Dict[str, object]] = []
        self.pairs: List[Dict[str, object]] = []
        self.activity = {family: defaultdict(int) for family in PROMOTION_FAMILIES + FUNCTIONAL_FAMILIES}
        self._done = set()
        self._in_round = False

    def due(self, task_id: int, epoch: int) -> Optional[str]:
        return round_kind(task_id, epoch, epochs_per_task=self.cfg.epochs_per_task, task_count=self.task_count)

    def round(self, trainer, task, epoch: int, *, condition, history, guard, boundary: bool = False):
        key = (task.task_id, int(epoch), bool(boundary))
        require(not self._in_round, "structural rounds do not nest", round=key)
        require(key not in self._done, "a structural round runs once per task, epoch and kind", round=key)
        self._done.add(key)
        self._in_round = True
        try:
            return self._round(trainer, task, int(epoch), condition, history, guard, bool(boundary))
        finally:
            self._in_round = False

    def observe_boundary(self, task_id: int, reports) -> None:
        """Close and reopen the training interval at a task boundary (reporting only)."""
        self.ledger.close_interval(f"boundary_{task_id}", reports)
        self.ledger.open_interval(f"boundary_{task_id}", reports)

    def summary(self) -> Dict[str, object]:
        return dict(
            switches=dict(structure=self.structure, promotions=self.promotions,
                          promotions_requested=self.promotions_requested),
            schedule=dict(audit_epochs=list(AUDIT_EPOCHS), promotion_start_epoch=PROMOTION_START_EPOCH,
                          prune_start_epoch=PRUNE_START_EPOCH, task_count=self.task_count,
                          boundary_tasks=[t for t in range(1, self.task_count - 1)]),
            totals=dict(relabel_pairs=self.relabel_pairs, functional_installs=self.functional_installs,
                        seconds=self.seconds, **self.counts),
            # `activity` counts rounds per family and status; `relabel_pairs` counts individual label exchanges.
            counting=dict(activity="rounds per family and status", relabel_pairs="individual label exchanges",
                          functional_installs="installed patches (at most one per round)"),
            activity={family: dict(statuses) for family, statuses in self.activity.items()},
            rounds=self.rounds, proposals=self.proposals, promotions=self.pairs, harm=self.ledger.summary(),
        )

    # -- one round ------------------------------------------------------------------------------

    def _round(self, trainer, task, epoch, condition, history, guard, boundary):
        started, tid = time.perf_counter(), task.task_id
        rid = f"task_{tid}_epoch_{epoch}" + ("_boundary" if boundary else "")
        require(tuple(history.get(tid, ())) == tuple(condition.support), "the round's support is not the committed one",
                round_id=rid)
        state = trainer.persistent_state
        start = self._invariants(trainer)
        revisions = (state.params_revision, state.roles_revision, self.functional_installs, self.relabel_pairs)
        signature = learning_state_signature(trainer)
        reused = condition.column if condition.kind == "reuse" else None
        fresh = tuple(condition.writable)
        supports = {j: tuple(history[j]) for j in range(tid + 1)}
        record = dict(round_id=rid, task_id=tid, epoch=epoch, boundary=boundary, global_step=state.global_step,
                      support=list(condition.support))
        statuses = {}

        comparator, arrays = self._audit(trainer, tid, supports)
        self.ledger.close_interval(f"{rid}:comparator", comparator)
        record["comparator_J"] = self._objective(comparator, tid)
        usage, nodes = self._usage(trainer, tid, epoch, reused, history, supports)
        outer = promotion_outer_label(epoch, "boundary" if boundary else "round")
        latest = comparator
        if outer is None:
            statuses.update({family: "not_due" for family in PROMOTION_FAMILIES})
        else:
            latest = self._promote(trainer, tid, rid, outer, usage, nodes, reused, supports, comparator, arrays,
                                   record, statuses)
        if self.structure != "on":
            statuses.update({family: "switched_off" for family in FUNCTIONAL_FAMILIES})
        elif boundary:
            statuses.update({family: "not_due" for family in FUNCTIONAL_FAMILIES})
        else:
            latest = self._functional(trainer, task, epoch, rid, reused, fresh, supports, comparator, arrays,
                                      guard, record, statuses) or latest
        self.ledger.open_interval(f"{rid}:end", latest)

        # A round runs no optimizer step, refreshes nothing and moves no stream or support.
        require(self._invariants(trainer) == start, "a structural round changed protected run state", round_id=rid)
        installs, relabels = self.functional_installs - revisions[2], self.relabel_pairs - revisions[3]
        require((state.params_revision, state.roles_revision) == (revisions[0] + installs, revisions[1] + relabels),
                "structural revision counts", round_id=rid)
        if not installs and not relabels:
            require(learning_state_signature(trainer) == signature, "a round without installs changed state",
                    round_id=rid)
        validate_roles(trainer.structure, self.cfg, state.shell_roles)
        for family, status in statuses.items():
            self.activity[family][status] += 1
        seconds = time.perf_counter() - started
        self.seconds += seconds
        self.counts["rounds"] += 1
        record.update(status="complete", relabels_installed=relabels, functional_installed=installs,
                      activity=statuses, seconds=seconds)
        self.rounds.append(record)
        log_progress(f"structural round {rid} comparator_J={record['comparator_J']:.6f} relabels={relabels} "
                     f"functional={installs} activity={statuses}", component="structure")
        return record

    def _invariants(self, trainer):
        state = trainer.persistent_state
        return (
            state.global_step, adam_count(trainer.opt_state), fingerprint(trainer.rng_key),
            state.certificates_revision, fingerprint(state.certificates), fingerprint(state.shell_stats),
            fingerprint((state.task_support_snapshots, state.current_support, state.boundary_support)),
            tuple(sorted((tid, split, getattr(task.loader(split), "passes", None))
                         for tid, task in trainer.tasks.items() for split in ("train", "validation", "test"))),
        )

    def _audit(self, trainer, tid, supports):
        """All seen validation tasks (certificates fixed, RNG restored), tagged with their support."""
        reports, examples = collect_validation(trainer, range(tid + 1), {tid: supports[tid]})
        for j, report in reports.items():
            report["support"] = list(supports[j])
        self.counts["evaluation_passes"] += tid + 1
        return reports, examples

    @staticmethod
    def _objective(reports, tid) -> float:
        return float(teacher_objective({j: r["cross_entropy_float64"] for j, r in reports.items()}, tid)[0])

    def _activations(self, trainer, task, loader, support, names, seed):
        """Shell `z_mu` on one seeded batch of a loader copy; the parent stream and RNG never move."""
        passes = loader.passes
        batch = next(iter(dataclasses.replace(loader, shuffle=True, seed=int(seed), max_batches=1)))
        require(loader.passes == passes, "an evidence batch moved a loader")
        with preserve_reporting_state(trainer):
            clamps = trainer._build_clamps(batch, task, tuple(sorted(support)), refresh_certificates=False,
                                           include_targets=False)
            state = trainer._run_evaluation_inference(trainer.params, clamps, jax.random.PRNGKey(int(seed) % 2 ** 31))
            return jax.device_get({name: state.nodes[name].z_mu for name in names})

    def _usage(self, trainer, tid, epoch, reused, history, supports):
        """Unit usage of shared columns in every seen task and of the reused column in its contexts."""
        contexts = {column: list(range(tid + 1)) for column in self.shared}
        if reused is not None:
            dependents = [j for j, support in sorted(history.items()) if j < tid and reused in support]
            contexts[reused] = sorted({*dependents, tid})
        by_task = defaultdict(list)
        for column, tasks in contexts.items():
            for j in tasks:
                by_task[j].append(column)
        usage = defaultdict(dict)
        for j in sorted(by_task):
            names = [n for column in by_task[j] for n in column_nodes(self.layout, column, include_l=True)]
            seed = round_seed(self.cfg.seed, j, tid, epoch, USAGE_TAG)
            task = trainer.tasks[j]
            values = self._activations(trainer, task, task.validation_loader, supports[j], names, seed)
            for name in names:
                usage[name][j] = unit_usage(values[name])
            self.counts["usage_batches"] += 1
        nodes = [n for column in sorted(contexts) for n in column_nodes(self.layout, column, include_l=True)]
        return usage, nodes

    # -- promotion by relabelling ---------------------------------------------------------------

    def _promote(self, trainer, tid, rid, outer, usage, nodes, reused, supports, comparator, arrays, record,
                 statuses):
        state = trainer.persistent_state
        frozen = {(n, s) for (t, n, s) in self.dwell if t == tid}
        pairs = evidence_pairs(state.shell_roles, usage, nodes, outer, min_contexts=PROMOTION_MIN_CONTEXTS,
                               frozen=frozen)
        for pair in pairs:
            pair["column"] = self.layout.shells[pair["node"]][0]
            pair["family"] = "S-PROMOTE" if pair["column"] in self.shared else "R-PROMOTE"
        record.update(promotion_outer_label=outer, promotion_pairs=len(pairs), promotions_installed=0)
        install = self.promotions == "on" and bool(pairs)
        for family in PROMOTION_FAMILIES:
            hit = any(pair["family"] == family for pair in pairs)
            statuses[family] = ("switched_off" if self.promotions != "on" else "installed" if hit
                                else "no_proposal")
        if not install:
            self.pairs.extend(dict(pair, round_id=rid, task_id=tid, status="shadow") for pair in pairs)
            return comparator

        before = (state.shell_roles, state.roles_revision, trainer.write_policy)
        roles = state.shell_roles
        for pair in pairs:
            roles = relabel(roles, pair["node"], pair["promoted_slot"], pair["demoted_slot"])
        validate_roles(trainer.structure, self.cfg, roles)
        scaled = self.shared + (() if reused is None else (reused,))
        table = unit_scale_table(trainer.structure, self.cfg, roles, scaled)
        try:
            state.shell_roles = roles
            state.roles_revision += len(pairs)
            trainer.set_write_policy(dataclasses.replace(trainer.write_policy, unit_scales=table))
            post, post_arrays = self._audit(trainer, tid, supports)
            for j in range(tid + 1):  # a relabel changes no parameter, so the function is bitwise the same
                for key in ("prediction", "cross_entropy"):
                    a, b = np.asarray(post_arrays[j][key]), np.asarray(arrays[j][key])
                    require(a.dtype == b.dtype and a.tobytes() == b.tobytes(), "a relabel changed the function",
                            round_id=rid, task_id=j, field=key)
        except BaseException:
            state.shell_roles, state.roles_revision = before[0], before[1]
            trainer.set_write_policy(before[2])
            raise
        self.relabel_pairs += len(pairs)
        for pair in pairs:  # dwell only for installed pairs
            self.dwell.update({(tid, pair["node"], pair["promoted_slot"]), (tid, pair["node"], pair["demoted_slot"])})
            self.pairs.append(dict(pair, round_id=rid, task_id=tid, status="installed"))
        self.ledger.record_install(rid, "relabel", tid, comparator, post, {j: 0 for j in range(tid)})
        record.update(promotions_installed=len(pairs), relabel_function_bitwise_equal=True)
        return post

    # -- audited functional edits ---------------------------------------------------------------

    def _proposals(self, trainer, task, epoch, reused, fresh, supports):
        rows = [Proposal("S-INH", c, "inhibit") for c in self.shared]
        if reused is not None:
            rows.append(Proposal("R-INH", reused, "inhibit"))
        rows += [Proposal("F-INH", c, "inhibit") for c in fresh]
        labels, silent = prune_labels(epoch), 0
        if labels and fresh:
            # One augmented current-task fit batch, shared by every fresh column; old fit data is never read.
            names = [n for c in fresh for n in column_nodes(self.layout, c, include_l=True)]
            seed = round_seed(self.cfg.seed, task.task_id, epoch, PRUNE_PROBE_TAG)
            values = self._activations(trainer, task, task.train_loader, supports[task.task_id], names, seed)
            self.counts["prune_batches"] += 1
            usage = {name: unit_usage(values[name]) for name in names}
            for c in fresh:
                named = [(n, trainer.params.nodes[n]) for n in column_nodes(self.layout, c, include_l=True)]
                for label in labels:
                    hit = prune_trigger(named, trainer.persistent_state.shell_roles, usage, label,
                                        weak_fraction=PRUNE_WEAK_FRACTION, score_quantile=PRUNE_SCORE_QUANTILE)
                    if hit is None:
                        silent += 1
                    else:
                        rows.append(Proposal("F-PRUNE", c, "prune", node=hit["node"], label=label,
                                             slots=hit["slots"]))
        require(len(rows) <= MAX_RAW_PROPOSALS, "too many raw structural proposals", count=len(rows))
        return [(f"{p.family}_{i:02d}", p) for i, p in enumerate(rows)], silent

    def _harm(self, reports, comparator, tid):
        counts = {j: comparator[j]["num_examples"] for j in range(tid)}
        harm = harm_check({j: reports[j]["correct_examples"] for j in range(tid)},
                          {j: comparator[j]["correct_examples"] for j in range(tid)}, counts)
        for j, row in harm["per_task"].items():
            row["per_edit_passes"] = row["passes"]
            row["cumulative_passes"] = self.ledger.cumulative_passes(j, row["lost_correct"], row["examples"])
            row["passes"] = row["per_edit_passes"] and row["cumulative_passes"]
        harm["passes"] = all(row["passes"] for row in harm["per_task"].values())
        return harm

    def _functional(self, trainer, task, epoch, rid, reused, fresh, supports, comparator, arrays, guard, record,
                    statuses):
        tid = task.task_id
        eligible = {"S-INH": self.shared, "R-INH": () if reused is None else (reused,), "F-INH": fresh,
                    "F-PRUNE": fresh}
        proposals, silent = self._proposals(trainer, task, epoch, reused, fresh, supports)
        parent = _parent_fingerprints(trainer)  # every proposal is built from this unchanged parent
        rows, reports, examples, patches, learned, seen = {}, {"no_edit": comparator}, {"no_edit": arrays}, {}, {}, {}
        for pid, proposal in proposals:
            patch = build_patch(trainer, proposal, self.layout, eligible[proposal.family], parent_fingerprints=parent)
            touched = sorted({(e.node, e.field, e.key) for e in patch.edits if e.tree == "params"} & set(guard))
            require(not touched, "a structural proposal edits a protected leaf", proposal_id=pid, leaves=touched)
            row = dict(round_id=rid, proposal_id=pid, family=proposal.family, column=proposal.column,
                       operation=proposal.operation, node=proposal.node, label=proposal.label,
                       slots=list(proposal.slots), coordinates=int(sum(e.indices.size for e in patch.edits)),
                       selected=False)
            rows[pid] = row
            fp = patch.content_fingerprint
            if not patch.edits:
                row["status"] = "noop"
                continue
            if fp in seen:
                row.update(status="alias", alias_of=seen[fp])
                continue
            seen[fp] = pid
            clone = trainer.isolated_clone()  # zero optimizer updates; never a previous candidate's state
            apply_patch(clone, patch, self.layout, eligible[proposal.family])
            reports[pid], examples[pid] = self._audit(clone, tid, supports)
            del clone
            patches[pid], learned[pid] = patch, proposal.column
            row.update(status="audited", J=self._objective(reports[pid], tid))
        record.update(functional_proposals=len(proposals), functional_audited=len(learned), prune_silent=silent)

        winner = None
        if learned:
            check_pairing(examples, "no_edit", range(tid + 1))
            objectives = {cid: self._objective(r, tid) for cid, r in reports.items()}
            bound = score_family(examples, objectives, tid, tuple(learned),
                                 round_seed(self.cfg.seed, tid, epoch, BOOTSTRAP_TAG), comparator_id="no_edit")
            harm = {pid: self._harm(reports[pid], comparator, tid) for pid in learned}
            winner, reasons = decide(learned, bound, harm)
            record.update(margin=bound["margin"], quantile=bound["quantile"])
            for pid in learned:
                rows[pid].update(gain=bound["gain"][pid], lower_bound=bound["lower_bound"][pid],
                                 harm={j: dict(lost=r["lost_correct"], per_edit=r["per_edit_passes"],
                                               cumulative=r["cumulative_passes"])
                                       for j, r in harm[pid]["per_task"].items()},
                                 qualifies=not reasons[pid], reasons=reasons[pid])
        for family in FUNCTIONAL_FAMILIES:
            mine = [row for row in rows.values() if row["family"] == family]
            statuses[family] = (
                "not_due" if family == "F-PRUNE" and not prune_labels(epoch)
                else "no_proposal" if not mine
                else "installed" if winner is not None and rows[winner]["family"] == family
                else "noop_or_alias" if all(row["status"] in ("noop", "alias") for row in mine)
                else "qualified_not_selected" if any(row.get("qualifies") for row in mine)  # outranked
                else "failed_acceptance"
            )
        post = None
        if winner is not None:
            post = self._install(trainer, tid, rid, winner, patches[winner], eligible[rows[winner]["family"]],
                                 reports[winner], examples[winner], supports, guard)
            rows[winner]["selected"] = True
            self.ledger.record_install(rid, winner, tid, comparator, post,
                                       {j: harm[winner]["per_task"][j]["lost_correct"] for j in range(tid)})
            record["functional_selected"] = winner
        self.proposals.extend(rows.values())
        return post

    def _install(self, trainer, tid, rid, pid, patch, eligible, audited, audited_examples, supports, guard):
        state = trainer.persistent_state
        adam = adam_state(trainer.opt_state)
        before = dict(params=trainer.params, mu=adam.mu, nu=adam.nu)
        saved = (trainer.params, trainer.opt_state, state.params_revision,
                 getattr(trainer, "_last_graph_state", _MISSING))
        try:
            apply_patch(trainer, patch, self.layout, eligible)
            verify_patch_scope(before, trainer, patch)
            require(not changed_leaves(trainer.params, guard), "an install changed a protected leaf", proposal_id=pid)
            validate_roles(trainer.structure, self.cfg, state.shell_roles)
            post, post_examples = self._audit(trainer, tid, supports)
            for j in range(tid + 1):  # the installed state must reproduce its isolated audit
                gap = float(np.max(np.abs(post_examples[j]["cross_entropy"].astype(np.float64)
                                          - audited_examples[j]["cross_entropy"].astype(np.float64))))
                require(post[j]["correct_examples"] == audited[j]["correct_examples"] and gap <= CE_TOLERANCE,
                        "the installed state differs from its audit", round_id=rid, task_id=j, gap=gap)
        except BaseException:
            trainer.params, trainer.opt_state, state.params_revision = saved[0], saved[1], saved[2]
            state.params, state.opt_state = trainer.params, trainer.opt_state
            if saved[3] is _MISSING:
                if hasattr(trainer, "_last_graph_state"):
                    del trainer._last_graph_state
            else:
                trainer._last_graph_state = saved[3]
            raise
        self.functional_installs += 1
        log_progress(f"structural install {rid} {pid} coordinates={sum(e.indices.size for e in patch.edits)}",
                     component="structure")
        return post
