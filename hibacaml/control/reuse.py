"""Restricted support selection: a later task may adopt one earlier learned column (guide §8).

Every boundary after T0 draws four fresh columns. Each learned candidate replaces
the highest of them; a reserve column in the same slot is a diagnostic null. All
conditions train the same materialized current-task fit pass on isolated clones of
the parent, then the predictive teacher picks the candidate whose validation gain
over fresh survives a simultaneous class-stratified bootstrap and the per-old-task
harm cap. Only the allocation and its write policy reach the main trainer; every
trial-trained state is discarded.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hibacaml.control.roles import unit_scale_table
from hibacaml.data.cifar import task_seeds
from hibacaml.reporting.integrity import (
    changed_leaves,
    column_fingerprints,
    fingerprint,
    leaf_digests,
    learning_state_signature,
    preserve_reporting_state,
    require,
)
from hibacaml.reporting.logger import log_progress
from hibacaml.training.trainer import adam_count
from hibacaml.training.write_policy import WritePolicy, frozen_leaves

SUPPORT_POLICIES = ("restricted_reuse", "fixed_fresh", "replay")
TEACHER_WEIGHTS = (1.0, 1.0, 0.5)  # current CE, worst old CE, mean old CE
BOOTSTRAP_RESAMPLES = 9999
BOOTSTRAP_CONFIDENCE = 0.95
QUANTILE_METHOD = "higher"
STRICT_TOLERANCE = 1e-12            # observed gain and its lower bound must exceed it
HARM_CAP_PERCENT = 1                # per old task: 100 * lost_correct <= 1 * examples
CE_TOLERANCE = 1e-6                 # reported task CE vs float64 mean of per-example CE


@dataclass(frozen=True)
class Condition:
    """One support for the task; `writable` holds the newly allocated columns only."""

    condition_id: str
    kind: str  # fresh | reuse | null
    column: Optional[int]
    support: Tuple[int, ...]
    writable: Tuple[int, ...]

    @property
    def extra(self) -> Tuple[int, ...]:
        """The learned or null column written at label rates with its kernel and pool frozen."""
        return () if self.kind == "fresh" else (self.column,)


class ColumnAllocator:
    """Unused adaptive capacity (lowest indices first) and the committed support history."""

    def __init__(self, pool):
        self.adaptive = tuple(pool.adaptive_indices)
        self.own_count = int(pool.topk_nonshared)
        self.null_column = int(pool.reserve_indices[0])
        self.unused: List[int] = list(self.adaptive)
        self.history: Dict[int, Tuple[int, ...]] = {}

    def draw(self) -> Tuple[int, ...]:
        require(len(self.unused) >= self.own_count, "the unused adaptive pool cannot supply a fresh support",
                unused=sorted(self.unused))
        return tuple(sorted(self.unused)[: self.own_count])

    def candidates(self) -> Tuple[int, ...]:
        return tuple(sorted({c for support in self.history.values() for c in support}))

    def conditions(self, task_id: int) -> List[Condition]:
        """Fresh, then each learned candidate in column order, then the null; consumes nothing."""
        fresh = self.draw()
        result = [Condition("fresh", "fresh", None, fresh, fresh)]
        if task_id == 0:
            return result
        retained = fresh[:-1]  # the highest fresh column is the only displaced slot
        for kind, columns in (("reuse", self.candidates()), ("null", (self.null_column,))):
            for column in columns:
                result.append(Condition(f"{kind}_col{column}", kind, column,
                                        tuple(sorted(retained + (column,))), retained))
        return result

    def commit(self, task_id: int, condition: Condition) -> None:
        require(task_id == len(self.history), "supports are committed once, in task order", task_id=task_id)
        require(condition.kind != "null", "the null condition is diagnostic only", task_id=task_id)
        require(condition in self.conditions(task_id), "not a permitted condition for this task",
                task_id=task_id, proposed=dataclasses.asdict(condition))
        for column in condition.writable:
            self.unused.remove(column)
        self.history[task_id] = tuple(condition.support)


def replayed_condition(conditions: Sequence[Condition], record: Mapping, task_id: int) -> Condition:
    """The permitted condition a recorded `{kind, column, support, writable}` decision names."""
    matches = [
        c for c in conditions
        if c.kind == record["kind"] and c.column == record["column"]
        and list(c.support) == list(record["support"]) and list(c.writable) == list(record["writable"])
    ]
    require(len(matches) == 1 and matches[0].kind != "null",
            "replayed decision is not a permitted condition", task_id=task_id, record=dict(record))
    return matches[0]


def validate_replay(pool, replay: Mapping[int, Mapping], task_ids: Iterable[int]) -> None:
    """Simulate the allocation so an impermissible schedule fails before any training."""
    allocator = ColumnAllocator(pool)
    for tid in task_ids:
        require(tid in replay, "the replayed schedule lacks a requested task", task_id=tid)
        allocator.commit(tid, replayed_condition(allocator.conditions(tid), replay[tid], tid))


def trial_update_bound(own_count: int, trial_lengths: Sequence[int]) -> int:
    """Upper bound on discarded trial updates: fresh + every earlier own column + the null."""
    return sum((own_count * t + 2) * trial_lengths[t] for t in range(1, len(trial_lengths)))


# ---- teacher ---------------------------------------------------------------------------------


def teacher_objective(ce: Mapping[int, object], current: int):
    """J = CE_current + max old CE + 0.5 mean old CE; values are scalars or resample arrays."""
    old = [task for task in sorted(ce) if task < current]
    require(bool(old), "the teacher objective needs at least one old task")
    stacked = np.stack([np.asarray(ce[task], dtype=np.float64) for task in old])
    worst, mean = stacked.max(axis=0), stacked.mean(axis=0)
    w_current, w_worst, w_mean = TEACHER_WEIGHTS
    return w_current * np.asarray(ce[current], dtype=np.float64) + w_worst * worst + w_mean * mean, worst, mean


def harm_check(candidate_correct, reference_correct, counts) -> Dict[str, object]:
    """Net lost correct predictions per old task; a task passes when 100*lost <= cap*n."""
    rows = {}
    for task in sorted(counts):
        lost, n = int(reference_correct[task]) - int(candidate_correct[task]), int(counts[task])
        rows[task] = dict(lost_correct=lost, examples=n, passes=100 * lost <= HARM_CAP_PERCENT * n)
    return dict(per_task=rows, passes=all(row["passes"] for row in rows.values()))


def bootstrap_guard(examples, current: int, learned_ids: Sequence[str], seed: int, *,
                    comparator_id: str = "fresh") -> Dict[str, object]:
    """Simultaneous lower bounds on every candidate's gain over the comparator.

    Within each task x true-class stratum, one draw per resample is shared by the
    comparator (fresh for support selection, no_edit for structural audits) and every
    candidate; the worst-old term is recomputed per resample.
    """
    ids = (comparator_id, *learned_ids)
    rng = np.random.default_rng(int(seed))
    observed, resampled = {cid: {} for cid in ids}, {cid: {} for cid in ids}
    for task in sorted(examples[comparator_id]):
        labels, rows_order = examples[comparator_id][task]["label"], examples[comparator_id][task]["row"]
        for cid in ids:
            other = examples[cid][task]
            require(np.array_equal(other["label"], labels) and np.array_equal(other["row"], rows_order),
                    "unpaired validation examples", task_id=task, condition_id=cid)
        ce = np.stack([np.asarray(examples[cid][task]["cross_entropy"], dtype=np.float64) for cid in ids])
        sums = np.zeros((len(ids), BOOTSTRAP_RESAMPLES), dtype=np.float64)
        for label in np.unique(labels):
            rows = np.flatnonzero(labels == label)
            k = rows.size
            draws = rng.integers(0, k, size=(BOOTSTRAP_RESAMPLES, k))
            counts = np.bincount(
                (np.arange(BOOTSTRAP_RESAMPLES)[:, None] * k + draws).ravel(), minlength=BOOTSTRAP_RESAMPLES * k
            ).reshape(BOOTSTRAP_RESAMPLES, k)
            sums += (counts.astype(np.float64) @ ce[:, rows].T).T
        for index, cid in enumerate(ids):
            observed[cid][task] = float(ce[index].mean())
            resampled[cid][task] = sums[index] / labels.size
    objective = {cid: float(teacher_objective(observed[cid], current)[0]) for cid in ids}
    star = {cid: teacher_objective(resampled[cid], current)[0] for cid in ids}
    gains = {cid: objective[comparator_id] - objective[cid] for cid in learned_ids}
    if learned_ids:
        z = np.max(np.stack([(star[comparator_id] - star[cid]) - gains[cid] for cid in learned_ids]), axis=0)
        quantile = float(np.quantile(z, BOOTSTRAP_CONFIDENCE, method=QUANTILE_METHOD))
    else:
        quantile = 0.0
    margin = max(0.0, quantile)
    return dict(objective=objective, gain=gains, quantile=quantile, margin=margin,
                lower_bound={cid: gains[cid] - margin for cid in learned_ids})


def check_pairing(examples, comparator_id: str, task_ids: Sequence[int]) -> None:
    """Every condition covers exactly the seen tasks, row for row and label for label."""
    expected = list(task_ids)
    for cid, by_task in examples.items():
        require(sorted(by_task) == expected, "validation must cover every seen task", condition_id=cid)
        for j in expected:
            require(np.array_equal(by_task[j]["row"], examples[comparator_id][j]["row"])
                    and np.array_equal(by_task[j]["label"], examples[comparator_id][j]["label"]),
                    "unpaired validation examples", task_id=j, condition_id=cid)


def score_family(examples, objectives: Mapping[str, float], current: int, learned_ids: Sequence[str], seed: int,
                 *, comparator_id: str) -> Dict[str, object]:
    """The bootstrap over comparator + candidates, consistent with their audited J and finite."""
    bound = bootstrap_guard({cid: examples[cid] for cid in (comparator_id, *learned_ids)}, current,
                            tuple(learned_ids), seed, comparator_id=comparator_id)
    for cid, value in bound["objective"].items():
        require(abs(value - objectives[cid]) <= STRICT_TOLERANCE, "bootstrap and audited objectives differ",
                condition_id=cid)
    scores = [bound["quantile"], bound["margin"], *bound["objective"].values(), *bound["gain"].values(),
              *bound["lower_bound"].values()]
    require(all(np.isfinite(scores)), "nonfinite teacher scores")
    return bound


def decide(learned: Mapping[str, int], guard, harm) -> Tuple[Optional[str], Dict[str, List[str]]]:
    """Lowest-J qualifying learned candidate (ties to the lower column), else None for fresh."""
    reasons, qualifying = {}, []
    for cid in learned:
        why = []
        if not guard["gain"][cid] > STRICT_TOLERANCE:
            why.append("no_positive_gain")
        if not guard["lower_bound"][cid] > STRICT_TOLERANCE:
            why.append("bootstrap_lower_bound_not_positive")
        failed = [task for task, row in harm[cid]["per_task"].items() if not row["passes"]]
        if failed:
            why.append(f"harm_cap_failed_tasks_{failed}")
        reasons[cid] = why
        if not why:
            qualifying.append(cid)
    winner = min(qualifying, key=lambda cid: (guard["objective"][cid], learned[cid]), default=None)
    return winner, reasons


# ---- trainer-facing helpers ------------------------------------------------------------------


def condition_policy(trainer, condition: Condition) -> WritePolicy:
    """Post-T0 write policy: shared (+ the extra column) at label rates with frozen pools."""
    shared = tuple(trainer.cfg.column_pool.shared_indices)
    scaled = shared + condition.extra
    table = unit_scale_table(trainer.structure, trainer.cfg, trainer.persistent_state.shell_roles, scaled)
    return WritePolicy(tuple(sorted(shared + tuple(condition.writable) + condition.extra)), scaled, table)


def install_condition(trainer, task_id: int, condition: Condition) -> None:
    """The one install path for trials and the main line; T0 keeps the unprotected program."""
    phi = trainer.cfg.phi if task_id == 0 else trainer.current_phi
    if task_id > 0:
        trainer.set_write_policy(condition_policy(trainer, condition))
    trainer.set_boundary_choice(task_id, condition.support, phi, record_dropped_columns=False)
    trainer.set_current_support(task_id, condition.support, phi)


def shared_references(trainer, layout) -> Dict[tuple, str]:
    """Byte references of every leaf the post-T0 protection freezes (stem, composer maps, shared)."""
    policy = condition_policy(trainer, Condition("shared", "fresh", None, (), ()))
    return leaf_digests(trainer.params, frozen_leaves(trainer.params, policy, layout))


def collect_validation(trainer, task_ids, supports: Mapping[int, Sequence[int]]):
    """Complete validation reports and per-example arrays; no certificate refresh, RNG restored.

    Old tasks use their saved supports unless `supports` overrides them.
    """
    reports, examples = {}, {}
    with preserve_reporting_state(trainer):
        for j in task_ids:
            support = supports.get(j)
            if support is None:
                support = trainer.persistent_state.task_support_snapshots[j].nonshared
            task = trainer.tasks[j]
            report = trainer.evaluate_loader(
                task, task.validation_loader, support, split_name="validation", mode="saved",
                refresh_certificates=False, collect_examples=True,
            )
            arrays = report.pop("examples")
            rows = arrays["row"].size
            mean = float(np.mean(arrays["cross_entropy"].astype(np.float64))) if rows else float("nan")
            require(
                rows > 0 and rows == task.validation_loader.examples == report["num_examples"]
                and all(arrays[k].size == rows for k in ("label", "prediction", "cross_entropy"))
                and report["accuracy"] == report["correct_examples"] / rows  # the evaluator's own division
                and np.isfinite(report["accuracy"]) and np.isfinite(report["cross_entropy"])
                and np.all(np.isfinite(arrays["cross_entropy"]))
                and abs(mean - report["cross_entropy"]) <= CE_TOLERANCE
                and int(np.sum(arrays["prediction"] == arrays["label"])) == report["correct_examples"],
                "validation evidence is incomplete or inconsistent", task_id=j, rows=rows,
            )
            reports[j] = dict(report, cross_entropy_float64=mean)
            examples[j] = arrays
    return reports, examples


# ---- boundary selection ----------------------------------------------------------------------


class BoundarySelector:
    """Chooses, commits and records each task's support under one support policy."""

    def __init__(self, cfg, layout, policy: str = "restricted_reuse", replay: Optional[Mapping] = None):
        require(policy in SUPPORT_POLICIES, f"support policy must be one of {SUPPORT_POLICIES}", policy=policy)
        require((policy == "replay") == (replay is not None),
                "the replay policy needs, and only it accepts, a recorded schedule")
        self.cfg, self.layout, self.policy = cfg, layout, policy
        self.replay = None if replay is None else {int(t): dict(r) for t, r in replay.items()}
        self.allocator = ColumnAllocator(cfg.column_pool)
        self.boundaries: List[Dict[str, object]] = []
        self.accounting = dict(trial_conditions=0, trial_updates=0,
                               selection_seconds=0.0, bootstrap_seconds=0.0)

    def select(self, trainer, task, guard: Mapping[tuple, str]) -> Condition:
        """The task's support; trials run only under restricted reuse after T0."""
        tid = task.task_id
        conditions = self.allocator.conditions(tid)
        record = dict(task_id=tid, policy=self.policy, fresh_draw=list(conditions[0].support),
                      candidates=list(self.allocator.candidates()), unused_before=sorted(self.allocator.unused),
                      trial_updates=0, conditions={})
        if self.policy == "replay":
            require(tid in self.replay, "the replayed schedule lacks a requested task", task_id=tid)
            chosen = replayed_condition(conditions, self.replay[tid], tid)
        elif tid == 0 or self.policy == "fixed_fresh":
            chosen = conditions[0]
        else:
            chosen = self._trials(trainer, task, guard, conditions, record)
        record["selected"] = dict(condition_id=chosen.condition_id, kind=chosen.kind, column=chosen.column,
                                  support=list(chosen.support), writable=list(chosen.writable))
        record["displaced"] = conditions[0].support[-1] if chosen.kind == "reuse" else None
        self.boundaries.append(record)
        log_progress(f"task={tid} support selected policy={self.policy} condition={chosen.condition_id} "
                     f"support={chosen.support}", component="selector")
        return chosen

    def commit(self, trainer, task_id: int, condition: Condition, guard: Dict[tuple, str]) -> None:
        """Allocate, add first-reuse references to `guard`, and install on the main trainer."""
        self.allocator.commit(task_id, condition)
        if condition.kind == "reuse":
            # Captured from the main parent before installation; an existing reference is never refreshed.
            policy = condition_policy(trainer, condition)
            leaves = [leaf for leaf in frozen_leaves(trainer.params, policy, self.layout)
                      if self.layout.column_of_node.get(leaf[0]) == condition.column]
            for leaf, digest in leaf_digests(trainer.params, leaves).items():
                guard.setdefault(leaf, digest)
        install_condition(trainer, task_id, condition)
        self.boundaries[-1]["unused_after"] = sorted(self.allocator.unused)

    def summary(self, *, main_updates: int, trial_update_upper_bound: int) -> Dict[str, object]:
        accounting = dict(self.accounting, main_updates=int(main_updates),
                          trial_update_upper_bound=int(trial_update_upper_bound))
        accounting["total_updates"] = int(main_updates) + accounting["trial_updates"]
        return dict(
            policy=self.policy,
            teacher=dict(weights=list(TEACHER_WEIGHTS), resamples=BOOTSTRAP_RESAMPLES,
                         confidence=BOOTSTRAP_CONFIDENCE, quantile_method=QUANTILE_METHOD,
                         strict_tolerance=STRICT_TOLERANCE, harm_cap_percent=HARM_CAP_PERCENT,
                         ce_tolerance=CE_TOLERANCE, null_column=self.allocator.null_column),
            boundaries=self.boundaries,
            history={t: list(s) for t, s in self.allocator.history.items()},
            accounting=accounting,
        )

    # -- trials -----------------------------------------------------------------------------

    def _trials(self, trainer, task, guard, conditions: List[Condition], record) -> Condition:
        tid, started = task.task_id, time.perf_counter()
        updates_before = self.accounting["trial_updates"]
        seeds = task_seeds(self.cfg.seed, tid)
        parent_signature = learning_state_signature(trainer)
        pool = self.cfg.column_pool
        parent_columns = column_fingerprints(trainer.params, pool.adaptive_indices + pool.reserve_indices)
        # One independently seeded current-task fit pass (35 batches in production), shared by every condition.
        batches = list(dataclasses.replace(task.train_loader, seed=seeds["trial"]))
        require(len(batches) == len(task.train_loader), "the trial pass is not one full fit pass", task_id=tid)
        record.update(trial_seed=seeds["trial"], bootstrap_seed=seeds["bootstrap"], trial_batches=len(batches))

        parent = trainer.isolated_clone()  # diagnostic comparison only: the parent's own old-task evidence
        parent_reports, _ = collect_validation(parent, range(tid), {})
        del parent

        trials, examples = {}, {}
        for condition in conditions:
            cid = condition.condition_id
            trials[cid], examples[cid] = self._trial(trainer, tid, condition, batches, guard, parent_columns)
            require(learning_state_signature(trainer) == parent_signature, "a trial changed the main trainer",
                    task_id=tid, condition_id=cid)
        check_pairing(examples, "fresh", range(tid + 1))

        counts = {j: trials["fresh"]["examples"][j] for j in range(tid)}
        fresh_correct = {j: trials["fresh"]["correct"][j] for j in range(tid)}
        parent_correct = {j: parent_reports[j]["correct_examples"] for j in range(tid)}
        harm = {}
        for cid, trial in trials.items():
            correct = {j: trial["correct"][j] for j in range(tid)}
            harm[cid] = harm_check(correct, fresh_correct, counts)
            trial["harm_vs_fresh"] = {j: row["lost_correct"] for j, row in harm[cid]["per_task"].items()}
            trial["harm_vs_parent"] = {j: row["lost_correct"]
                                       for j, row in harm_check(correct, parent_correct, counts)["per_task"].items()}
            trial["acquisition_delta_pp"] = 100.0 * (trial["accuracy"][tid] - trials["fresh"]["accuracy"][tid])

        learned = {c.condition_id: c.column for c in conditions if c.kind == "reuse"}
        clock = time.perf_counter()
        bound = score_family(examples, {cid: trial["J"] for cid, trial in trials.items()}, tid, tuple(learned),
                             seeds["bootstrap"], comparator_id="fresh")
        self.accounting["bootstrap_seconds"] += time.perf_counter() - clock
        winner, reasons = decide(learned, bound, harm)
        for cid in learned:
            trials[cid].update(gain=bound["gain"][cid], lower_bound=bound["lower_bound"][cid],
                               qualifies=not reasons[cid], reasons=reasons[cid])
        for trial in trials.values():
            for key in ("correct", "examples"):
                trial.pop(key)
        chosen = next(c for c in conditions if c.condition_id == (winner or "fresh"))
        seconds = time.perf_counter() - started
        self.accounting["selection_seconds"] += seconds
        record.update(quantile=bound["quantile"], margin=bound["margin"], conditions=trials,
                      trial_updates=self.accounting["trial_updates"] - updates_before, seconds=seconds)
        log_progress(f"task={tid} support trials conditions={len(trials)} margin={bound['margin']:.6f} "
                     f"selected={chosen.condition_id}", component="selector")
        return chosen

    def _trial(self, parent, tid, condition, batches, guard, parent_columns):
        """Train one condition on an isolated clone, check isolation, and audit seen-task validation."""
        cid = condition.condition_id
        clone = parent.isolated_clone()
        require(fingerprint(clone.rng_key) == fingerprint(parent.rng_key), "a trial clone has another RNG key",
                condition_id=cid)
        install_condition(clone, tid, condition)
        state = clone.persistent_state
        start = dict(step=state.global_step, adam=adam_count(clone.opt_state), revision=state.params_revision,
                     certificates=state.certificates_revision, roles=state.roles_revision)
        roles = fingerprint(state.shell_roles)
        references = dict(leaf_digests(clone.params, frozen_leaves(clone.params, clone.write_policy, self.layout)))
        references.update(guard)
        self.accounting["trial_conditions"] += 1
        try:
            clone.train_batches(clone.tasks[tid], batches, condition.support, check_finite=True)
        finally:
            self.accounting["trial_updates"] += adam_count(clone.opt_state) - start["adam"]
        n = len(batches)
        require((state.global_step, adam_count(clone.opt_state), state.params_revision)
                == (start["step"] + n, start["adam"] + n, start["revision"] + n),
                "trial update counters", condition_id=cid)
        require(state.roles_revision == start["roles"] and fingerprint(state.shell_roles) == roles,
                "a trial changed the role labels", condition_id=cid)
        changed = changed_leaves(clone.params, references)
        require(not changed, "a trial changed a protected leaf", condition_id=cid, leaves=changed)
        current = column_fingerprints(clone.params, parent_columns)
        moved = sorted(c for c in parent_columns if c not in condition.support and current[c] != parent_columns[c])
        require(not moved, "a trial changed a column outside its support", condition_id=cid, columns=moved)
        reports, examples = collect_validation(clone, range(tid + 1), {tid: condition.support})
        J, worst, mean = teacher_objective({j: r["cross_entropy_float64"] for j, r in reports.items()}, tid)
        trial = dict(
            kind=condition.kind, column=condition.column, support=list(condition.support),
            writable=list(condition.writable), certificate_refreshes=state.certificates_revision - start["certificates"],
            J=float(J), current_ce=reports[tid]["cross_entropy_float64"], max_old_ce=float(worst),
            mean_old_ce=float(mean), accuracy={j: r["accuracy"] for j, r in reports.items()},
            ce={j: r["cross_entropy_float64"] for j, r in reports.items()},
            correct={j: r["correct_examples"] for j, r in reports.items()},
            examples={j: r["num_examples"] for j, r in reports.items()},
        )
        del clone
        log_progress(f"task={tid} trial {cid} J={trial['J']:.6f} acc={trial['accuracy'][tid]:.4f} "
                     f"cert_refreshes={trial['certificate_refreshes']}", component="selector")
        return trial, examples
