# Restricted Column Reuse and Audited Shell Adaptation

This document specifies the two controllers that the Split-CIFAR-10 runner
(`experiments/split_cifar.py`) uses in place of the exhaustive support search of the
MNIST path, and the write protection they rely on:

- **Restricted column reuse** (`hibacaml/control/reuse.py`) chooses each task's
  support at its boundary. A task may adopt at most one column learned on an earlier
  task; otherwise it takes fresh, unused columns.
- **Audited shell adaptation** (`hibacaml/control/structure.py`, arithmetic in
  `hibacaml/control/roles.py`) edits the inside of columns during training: it
  promotes useful shell units inward, inhibits redundant ones and prunes weak ones.

Both controllers judge every change on validation data, test candidate changes on
isolated copies of the model, and install a change only when it improves a fixed
objective under a bootstrap test without costing any old task more than 1% of its
validation predictions. The exceptions are label promotions, which leave the
model's function unchanged and are checked for exactly that instead, and pruning
proposals, which are drawn from a training batch (section 3.6). The background (columns, shells, certificates) is in
[ALGORITHM.md](ALGORITHM.md); the runner's settings and outputs are in
[EXPERIMENTS.md](EXPERIMENTS.md).

Terms: tasks are `T0` to `T4` in order. A column's *support* membership makes it
active for a task. Each column has a feature pool and five shell nodes, `B`, `K0`,
`K1`, `K2` and `L`, each with 92 units: 32 kernel units and shells of 10, 20 and 30
units. Every unit has a *label* (0 kernel, 1-3 shell tier), separate from its
physical slot.

## 1. Write Protection

From `T1` onward, the runner installs a `WritePolicy`
(`hibacaml/training/write_policy.py`) on the trainer. It has three fields:

- `trainable_columns`: the shared columns plus the task's own columns; every other
  column's parameters are frozen;
- `frozen_feature_pools`: the shared columns, plus a reused column;
- `unit_scales`: a per-unit write rate. In shared columns and a reused column, each
  unit's rate depends on its label: `LABEL_WRITE_SCALES = (0.0, 0.0414, 0.1242,
  0.3727)` for labels 0-3, so kernel units never change and outer shells adapt
  fastest. The task's fresh columns learn at rate 1.

Installing any policy also freezes the convolutional stem and the composer's
feature and output maps. The compiled update masks protected gradients before the
gradient clip, masks and scales the AdamW update, and finally restores the original
bytes of every protected coordinate, so protected values stay exactly unchanged
(signed zeros included), not merely close.

Before `T1`'s support is chosen, the runner records a byte digest of every frozen
leaf: the stem, the composer maps, the shared columns' kernel and feature-pool
leaves. When a column is reused for the first time, its kernel and feature-pool
leaves are added. These references are never refreshed. The runner checks them
after every epoch, after every shell-adaptation round and at every task boundary,
and stops with an `IntegrityError` naming any leaf that changed. The summary's
`protection_activation_step` is the first optimizer update covered (1,051, the first
update of `T1`).

## 2. Restricted Column Reuse

### 2.1 Candidate conditions

`ColumnAllocator` keeps the unused adaptive columns (31 at the start) and the
committed support of every task. At a boundary it lists the permitted *conditions*
without consuming anything:

- **fresh**: the four lowest-index unused adaptive columns. At `T0` this is the only
  condition (columns 3-6).
- **reuse** (`reuse_col<c>`): for every column `c` in an earlier task's support, the
  first three fresh columns plus `c`. The highest fresh column is the only slot that
  can be displaced.
- **null** (`null_col<r>`): the first three fresh columns plus the first reserve
  column (34). This is a diagnostic comparison only and can never be adopted.

A condition's `writable` columns are the ones it newly allocates: all four for
fresh, the first three for reuse and null. Committing a condition removes only those
from the unused pool, so a displaced fresh column stays unused, keeps its initial
parameters and is drawn again at the next boundary. Supports are committed once per
task, in order, and must equal one of the listed conditions exactly. If fewer than
four unused columns remain, the run stops instead of falling back.

### 2.2 Isolated trials

At every boundary after `T0`, under the default `restricted_reuse` policy:

1. One pass over the current task's fit data is materialized from an independent
   seed (`task_seeds(seed, task)["trial"]`): 35 batches, shared by every condition.
2. Each condition runs on an `isolated_clone()` of the real model: same parameters,
   optimizer state, certificates, role labels and random key, and its own copy of
   every data loader. The condition's support and write policy are installed on the
   clone. A reused or null column trains like a shared column: its kernel and
   feature pool are frozen and its shells learn at their label rates, while its
   output logits learn normally.
3. The clone trains on the 35 batches through the ordinary per-update path,
   including scheduled certificate refreshes, but no shell adaptation. A non-finite
   loss stops the run.
4. After training the runner checks that the clone advanced exactly 35 optimizer
   steps, that its role labels did not change, that every protected leaf is
   unchanged, and that no adaptive or reserve column outside the condition's support moved. The real
   model's learning state must also be unchanged.
5. The clone is evaluated on the validation split of every task seen so far (old
   tasks under their saved supports, the current task under the condition's
   support), without refreshing certificates and with the random key restored. The
   evaluation keeps the per-example labels, predictions and cross-entropies and
   checks them against the report: counts, finiteness, and agreement of the
   reported cross-entropy with the float64 mean of the per-example values within
   `1e-6`.

Old tasks' fit and test data are never read during selection.

### 2.3 Objective, bootstrap and harm cap

For each condition, with task cross-entropies taken as float64 means of the
per-example values:

```text
J = CE(current task) + max over old tasks CE(old) + 0.5 * mean over old tasks CE(old)
```

Lower is better. For each learned candidate `c`, the observed gain over fresh is
`G_c = J(fresh) - J(c)`.

The gains are tested together by a paired bootstrap (`bootstrap_guard`): 9,999
resamples, drawn within each task and true class, with the same draws applied to
fresh and every learned candidate. The worst-old-task term is recomputed inside every
resample. With `G*_c` the gain in a resample,

```text
Z      = max over candidates of (G*_c - G_c)
margin = max(0, 95th percentile of Z, method "higher")
L_c    = G_c - margin
```

so `L_c` is a lower bound that holds for all candidates simultaneously. The null
condition is evaluated but never enters this family.

The harm cap compares integer correct counts with fresh on each old task:
`lost = correct(fresh) - correct(c)`, and the task passes when
`100 * lost <= 1 * examples` (at most 1% of its validation examples).

### 2.4 Decision and commit

A learned candidate qualifies only if `G_c > 1e-12`, `L_c > 1e-12` and every old task
passes the harm cap. The qualifying candidate with the lowest `J` wins, ties going to
the lower column index; if none qualifies, fresh wins. The decision rule
(`decide`) and the bootstrap are shared with shell adaptation.

Only the decision reaches the real model: the allocator commits the condition, the
reused column's references are added to the protection digest on its first reuse,
and the condition's support and write policy are installed. Every trial-trained
state is discarded. Training of the task then continues from the real model's own
state.

### 2.5 Controls and accounting

The runner's `SUPPORT_POLICY` setting selects how supports are chosen:

- `restricted_reuse` (default): the trials above;
- `fixed_fresh`: always the fresh condition, with no trials;
- `replay`: the supports recorded in an earlier run summary. Each recorded choice is
  validated against the permitted conditions before training starts.

Trial updates are counted from the optimizer's step counter on each clone (also
when a trial fails part-way) and reported separately from main training. With five
tasks the trials run at most `35 * (6 + 10 + 14 + 18) = 1,680` updates (fresh, every
earlier column and the null at each boundary); reuse shrinks later candidate lists.

`run_summary.json` records, under `support_selection`, each boundary's fresh draw,
candidates, unused columns before and after, seeds, bootstrap margin, selected
condition and displaced column, and one row per condition: `J` and its terms,
per-task accuracy and cross-entropy, gain, lower bound, harm against fresh and
against the model before the trial, the current-task accuracy difference to fresh in
percentage points, and rejection reasons. It also records the committed history and
the update accounting.

## 3. Audited Shell Adaptation

### 3.1 Schedule

`StructuralController` runs *rounds* from the runner's end-of-epoch hook, before
that epoch's validation:

- **audit rounds** at epochs 5, 10, 15, 20 and 25 of every task from `T1`;
- a **handoff round** at the final epoch (30) of every task from `T1` except the
  last one in the run.

A five-task run therefore has 23 rounds. Rounds perform no optimizer update and
never run inside support trials.

Every round first evaluates the unchanged model, the `no_edit` comparator, on the
validation split of every seen task, then samples unit usage. Promotion and
parameter edits follow as below.

| Round | Promotion | Parameter edits |
|---|---|---|
| Audit, epoch 5 | none | inhibition |
| Audit, epochs 10-20 | tier 3 to tier 2 | inhibition, pruning of label 3 |
| Audit, epoch 25 | tier 3 to tier 2 | inhibition, pruning of labels 3 and 2 |
| Handoff | tier 2 to tier 1 | none |

### 3.2 Eligibility

| Column | Promotion | Inhibition | Pruning |
|---|---|---|---|
| Shared | B, K0-K2, L | B, K0-K2 | none |
| Reused this task | B, K0-K2, L | B, K0-K2 | none |
| Fresh this task | none | B, K0-K2 | B, K0-K2, L |
| Inactive | none | none | none |

Kernel units are never promoted, inhibited or pruned. Every parameter edit is
audited on all seen tasks, including tasks that never used the edited column.

### 3.3 Usage evidence

Unit usage is the mean absolute activation of each shell unit over a batch and its
tokens, computed in float32. A *context* is a task whose data the column has seen:
every seen task for a shared column, and for a reused column the earlier tasks that
used it plus the current task. For each context task the round draws one shuffled,
unaugmented 500-image validation batch from a seed derived from the run seed, the
context task, the current task, the epoch and a fixed tag. It runs inference under
that task's support without refreshing certificates. The original loaders and the
random key are untouched.

### 3.4 Promotion by relabelling

For each eligible node, with outer label `o` (3, or 2 in a handoff round) and inner
label `o - 1`:

1. The node needs at least two contexts.
2. In each context, a unit is *used* when its usage reaches the median usage of the
   node's inner-label units in that context.
3. The promoted unit is the outer-label unit used in the most contexts (at least
   two), then with the highest mean usage, then the lowest slot.
4. Its partner is the inner-label unit used in the fewest contexts, then with the
   lowest mean usage, then the lowest slot. It must be used in strictly fewer
   contexts than the promoted unit.
5. Units that already moved during the current task are excluded (dwell).

At most one pair per node is installed. Installation exchanges the two units'
labels, increases the role revision by the number of pairs, and rebuilds the write
policy's unit-scale table from the new labels. No parameter, optimizer moment,
random key or certificate changes. The runner then re-evaluates every seen
validation task and requires byte-identical predictions and per-example
cross-entropies. If they differ, labels, revision and policy are restored and the
run stops. The new labels affect certificates at the next scheduled refresh and
write rates from the next update.

With promotions switched off, the same pairs are recorded as *shadow* nominations;
they are not installed and do not count toward dwell.

### 3.5 Same-tier inhibition

One inhibition proposal per eligible column covers its B and K nodes. For each tier
label `t` in 1-3, with `W` the incoming feed-forward weight columns of the units carrying that
label (K nodes' recurrent weights are not included):

```text
W_hat      = W / (||W|| + 1e-8)                        (per column)
r          = mean over columns of max(W_hat^T W_hat - I, 0)   (denominator n)
adjustment = r;  twice: adjustment = r + 0.5 * adjustment
adjustment = 10 * (s_t * adjustment),   s = (0.30, 0.18, 0.08) for tiers 1-3
z_new      = max(z - adjustment, min(z, log(0.25 / 0.75)))
```

`z` is the unit's log precision (its gate is `sigmoid(z)`). The update pushes
redundant units' gates down toward 0.25 but never raises a gate that is already
below that floor. Only log-precision coordinates change, computed in float32 on the
host from the pre-edit weights; optimizer moments are untouched.

### 3.6 Recoverable pruning

Only the current task's fresh columns can be pruned. The round draws one augmented
256-image batch of the current task's fit data from a seed derived from the run
seed, the task, the epoch and a fixed tag, shared by all fresh columns. For each
fresh column and each eligible label, label 3 first and then label 2, each node's
units carrying that label are scored:

- **live**: incoming feed-forward weight norm `> 1e-6` and gate `> 0.2`;
- **weak usage**: usage below `0.10 x` the median usage of that label's units;
- **low score**: `gate * incoming norm - same-label redundancy` at or below the
  label's 0.125 quantile (NumPy's default linear method).

Medians and quantiles are over all units of the label, live or not. A unit is a
weak live unit when it meets all three conditions. The node with the most weak live
units wins, and all of them are pruned; ties go to the node with higher mean label
redundancy, then to the earlier node; each column-label hit is a separate proposal.

Pruning a unit zeroes its incoming weight column and bias, zeroes the incoming
recurrent column for K nodes (each unit's connection into the pruned unit), sets
its log precision to `-4.0`, and zeroes the matching Adam first and second moments.
The pruned unit's own outgoing recurrent weights, downstream weights and kernel
units are untouched, and nothing is permanently masked: later
training can regrow the unit.

### 3.7 Proposals, audit and installation

Proposals are built from the same post-promotion model, in a fixed order:

1. shared-column inhibition (`S-INH`), ascending column;
2. reused-column inhibition (`R-INH`);
3. fresh-column inhibition (`F-INH`), ascending column;
4. fresh-column pruning (`F-PRUNE`), ascending column, then label 3 before 2.

They get IDs `<family>_<index>` with a two-digit index (`S-INH_00`), numbered
across all rows. More than 16 rows stops
the run as an error; the Split-CIFAR-10 layout reaches at most 15.

Each proposal becomes a sparse *patch*: the exact parameter and moment coordinates
it changes, with their old and new values, bound to digests of the model and role
labels it was built from. A patch that changes nothing is a *no-op*; a patch whose
edits equal an earlier one is an *alias*. Neither is audited. No patch may touch a
protected leaf.

Each remaining patch is applied to its own isolated copy of the model, with no
optimizer update, and evaluated on every seen task's validation split. The family is
scored against `no_edit` with the same objective `J`, the same simultaneous
bootstrap and the same decision rule as column reuse. All audits in a round share one bootstrap seed.
A patch qualifies only if, on every old task, it passes both:

```text
per-edit:    100 * lost <= examples
cumulative:  100 * (net_lost + lost) <= examples
```

`net_lost` is the cumulative net loss of correct old-task predictions realized by
earlier installed edits. It is signed (improvements earn credit back), carries
across tasks and is never reset; ordinary training drift does not count.

At most one patch is installed per round: the lowest `J`, then lowest column, then
earliest proposal. Installation:

1. revalidates the patch against the current model;
2. installs parameters and moments in one step;
3. increases the parameter revision by one;
4. discards the cached graph state;
5. verifies that only the patch's coordinates changed and no protected leaf moved;
6. re-evaluates every seen task.

The result must match the audit exactly in correct counts and within `1e-6` in
per-example cross-entropy. If any step fails, the previous model, optimizer state,
revision and cache are restored and the run stops. Only after success are the harm
ledger and counters updated.

### 3.8 Round invariants and records

A round must leave unchanged: the optimizer step and Adam count, the random key, the
loader positions, the saved supports, and the certificates and shell statistics.
Revisions must change by exactly the installed parameter edits and label exchanges,
and a round that installs nothing must leave the whole learning state unchanged.
Rounds cannot nest, and each (task, epoch, round kind) runs at most once.

`run_summary.json` records, under `structural_control`:

- the switches and schedule;
- every round, with its comparator `J`, installs and a status per mechanism family
  (`S-PROMOTE` and `R-PROMOTE` are label promotions in shared columns and in the
  reused column; the other four are the proposal families above);
- every proposal row: family, column, coordinates changed, `J`, gain, lower bound,
  per-task harm, whether it qualified and was selected, a status (`noop`, `alias`
  or `audited`) and rejection reasons;
- the installed and shadow label exchanges;
- the harm ledger: per install, predicted and realized old-task losses, the net
  cumulative loss, and separately the old-task changes during ordinary training
  between rounds;
- totals and activity counts.

The family statuses are:
- `switched_off`;
- `not_due`;
- `no_proposal`;
- `noop_or_alias`;
- `failed_acceptance`: no proposal passed the acceptance checks;
- `qualified_not_selected`: passed, but another family's proposal won;
- `installed`.

Activity counts count rounds; `relabel_pairs` counts individual label exchanges.

`STRUCTURE = "off"` keeps the comparator, the usage sampling and shadow nominations
but installs nothing. `PROMOTIONS = "off"` keeps inhibition and pruning and records
promotions as shadows.

## 4. Counting Updates, Edits and Label Changes

Three counters are kept separately:

- **optimizer updates** (`global_step`, equal to the Adam step count): 1,050 per
  task, 5,250 for five tasks; support trials run on clones and are counted
  separately;
- **parameter revision**: optimizer updates plus installed shell-adaptation patches;
- **role revision**: installed label exchanges.

After every task the runner requires
`global_step == adam_count ==` completed updates,
`params_revision == global_step + installed patches`,
`roles_revision == installed label exchanges` and one training pass per epoch. It
also requires that no dropped columns were recorded, that the saved supports equal
the committed allocation, that no adaptive or reserve column outside the task's
support changed, and that the protected leaves are unchanged.

## 5. Constants

| Constant | Value | Code |
|---|---|---|
| Objective weights (current, worst old, mean old) | 1.0, 1.0, 0.5 | `reuse.TEACHER_WEIGHTS` |
| Bootstrap resamples / confidence / quantile method | 9,999 / 0.95 / `higher` | `reuse.BOOTSTRAP_RESAMPLES`, `BOOTSTRAP_CONFIDENCE`, `QUANTILE_METHOD` |
| Strict gain tolerance | `1e-12` | `reuse.STRICT_TOLERANCE` |
| Old-task harm cap | 1% (per edit and cumulative) | `reuse.HARM_CAP_PERCENT` |
| Cross-entropy agreement | `1e-6` | `reuse.CE_TOLERANCE` |
| Audit epochs | 5, 10, 15, 20, 25 | `structure.AUDIT_EPOCHS` |
| Promotion / pruning start epoch | 10 / 10 | `structure.PROMOTION_START_EPOCH`, `PRUNE_START_EPOCH` |
| Minimum promotion contexts | 2 | `structure.PROMOTION_MIN_CONTEXTS` |
| Inhibition strengths / dose / floor | 0.30, 0.18, 0.08 / 10 / 0.25 | `structure.INHIBITION_*` |
| Pruning weak fraction / score quantile / log precision | 0.10 / 0.125 / -4.0 | `structure.PRUNE_WEAK_FRACTION`, `PRUNE_SCORE_QUANTILE`, `PRUNE_LOG_PRECISION` |
| Maximum raw proposals per round | 16 | `structure.MAX_RAW_PROPOSALS` |
| Label write scales | 0.0, 0.0414, 0.1242, 0.3727 | `roles.LABEL_WRITE_SCALES` |
