# HiBaCaML: Algorithm

This document explains how the ideas of the HiBaCaML/ColBa papers are realized in
this codebase: what each mechanism computes, where it lives, and where the code
deviates from the papers. The original HiBaCaML paper is
[*Hierarchical Bayesian Causal Modular Learning: A Two-Level Columnar Architecture for Continual Learning*](https://link.springer.com/chapter/10.1007/978-3-032-33010-9_21).
Companion documents:

- [CODE_STRUCTURE.md](CODE_STRUCTURE.md): every module and how a run flows through them.
- [EXPERIMENTS.md](EXPERIMENTS.md): the four experiment runners, their configurations,
  how to run them, and reference results.
- [COLUMN_REUSE_AND_SHELL_ADAPTATION.md](COLUMN_REUSE_AND_SHELL_ADAPTATION.md): the
  support and shell controllers used by the Split-CIFAR-10 runner.

Notation: `T0, T1, ...` are the tasks of a continual-learning sequence in order; a
*support* is the set of columns active for a task; *shared* columns are always active.

## 1. Overview

The codebase implements this as a FabricPC/JAX graph with:

- a fixed pool of column modules, configured by `ColumnPoolConfig` (20 columns in
  the MNIST configuration, 40 in the CIFAR one);
- a support mask that activates the shared columns plus a few selected nonshared
  columns (2 + 3 for Split-MNIST, 3 + 4 for Split-CIFAR-10);
- shell-bank column internals with kernel, tier1, tier2, and tier3 channels;
- a certificate and shell-controller subsystem that summarizes and edits column
  internals;
- for Split-MNIST, an exact teacher-first controller that searches support candidates,
  runs rollout audits over phi controller settings, and performs one-swap
  maintenance;
- for CIFAR-10, a convolutional stem in front of the columns; Split-CIFAR-10 adds
  two audited controllers: restricted column reuse at task boundaries and shell adaptation
  during training (section 3);
- four experiment runners: Split-MNIST, Full-MNIST, Full-CIFAR-10 and
  Split-CIFAR-10.

The implementation is a faithful structural instantiation of the paper's
HiBaCaML tuple, but it also contains pragmatic additions and simplifications:
the support search enumerates adaptive-only candidates first, reserves
enter through a separate recruitment path, the "Bayesian" internal state is
represented by precision-like parameters and certificates rather than full
posterior inference, and the runners train with backpropagation (the MNIST runners can switch to
predictive coding) while preserving the same controller and column semantics.


## 2. Paper Concepts to Implementation

### 2.1 HiBaCaML Tuple: `H = (N, C, delta, pi, T)`

**Paper purpose.** The paper defines HiBaCaML as a family of structurally
restricted component learners `N`, a combiner `C`, internal probabilistic states
`delta`, a top-level controller `pi`, and local counterfactual teachers `T`.
The theoretical point is to make the causal-continual-learning assumptions
constructive: sparse task supports reduce gradient leakage and cross-coupling,
while internal certificates let the controller reuse modules more intelligently.

**Implementation location.**

- `hibacaml/graph/builder.py:create_hibacaml_structure`
- `hibacaml/types.py:PersistentHiBaCaMLState`
- `hibacaml/training/trainer.py:HiBaCaMLTrainer` (abstract base)
- `hibacaml/training/pc.py:HiBaCaMLPCTrainer`
- `hibacaml/training/backprop.py:HiBaCaMLBackpropTrainer`
- `hibacaml/control/search.py:ExactSearchService`
- `hibacaml/control/certificates.py:CertificateController`
- `hibacaml/control/shells.py:ShellController`

**Implementation details.**

- `N`: each column is a FabricPC subgraph created inside
  `create_hibacaml_structure`. Column metadata records nodes such as
  `b_micro`, `k_micro`, `l_micro`, `feature_pool`, gates, logits, and
  certificate inputs.
- `C`: the combiner pools the active columns' logits and adds a correction from
  the attention-style column composer. `ScaledAddNode` combines `pooled_logits`
  with the `ColumnComposerNode` correction and applies softmax.
- `delta`: persistent internal state is represented by `ShellStats`,
  `ColumnCertificate`, precision-like shell biases, support snapshots, and
  controller audit tables in `PersistentHiBaCaMLState`.
- `pi`: `ExactSearchService` implements teacher-first support control. It
  performs boundary search, support posterior ranking, phi rollout search,
  replay-bank reselection, and local one-swap maintenance.
- `T`: local teachers are implemented as support one-swap audits and demotion
  swap audits in `ExactSearchService.local_one_swap` and
  `ExactSearchService.demotion_swap_audit`.

In the Split-CIFAR-10 runner, `pi` and `T` are instead realized by
`hibacaml/control/reuse.py:BoundarySelector` (restricted column reuse) and
`hibacaml/control/structure.py:StructuralController` (audited shell adaptation);
see section 3.

**Alignment status.** Faithful at the architectural level. The code realizes the
paper's tuple explicitly, but the controller is exact/audited rather
than a learned probabilistic selector.

### 2.2 Structurally Restricted Component Learners / Columns

**Paper purpose.** ColBa columns are not arbitrary dense experts. They are
restricted modules intended to align with recurring causal mechanisms. The
paper's Split-MNIST configuration describes 20 columns: 2 shared, 15 adaptive,
and 3 reserve columns, with 5 active columns per example.

**Implementation location.**

- `hibacaml/config/defaults.py:ColumnPoolConfig`
- `hibacaml/graph/builder.py:create_hibacaml_structure`
- `hibacaml/control/support.py`

**Implementation details.**

`ColumnPoolConfig` defines:

- `total_columns = 20`
- `shared_count = 2`
- `adaptive_count = 15`
- `reserve_count = 3`
- `topk_nonshared = 3`
- `active_support_size = shared_count + topk_nonshared`

The graph builder loops over `range(total_columns)` and creates a namespace
`col{index}` for each column. Shared columns are not structurally separate in
the graph; they are made always-active by support construction:

```text
nonshared support -> build_full_support(cfg, nonshared)
                  -> shared_indices + nonshared
                  -> support_mask_from_nonshared(...)
                  -> ElementwiseGateNode gates per column
```

**Alignment status.** Mostly exact. The 20-column partition and 5-active-column
runtime semantics match the paper. A notable deviation is candidate enumeration:
the paper describes searching all `18 choose 3 = 816` nonshared sets from
adaptive plus reserve columns, while the code's ordinary enumeration uses only
adaptive columns (`15 choose 3 = 455`) in `enumerate_nonshared_supports`.
Reserve candidates are introduced conditionally through
`enumerate_reserve_recruitment_supports` when saturation/posterior diagnostics
trigger recruitment.

The CIFAR configuration (`make_hibacaml_config("cifar")`) uses 40 columns. Each
CIFAR runner sets its own partition: Full-CIFAR-10 uses 4 shared + 5 adaptive
columns, all active; Split-CIFAR-10 uses 3 shared + 31 adaptive + 6 reserve with
4 nonshared columns active per task.

### 2.3 Micro-Columns and Shell Semantics

**Paper purpose.** The paper describes three typed micro-columns `K`, `L`, and
`B`: kernel center, lateral refinement, and bridge. Each carries a hard kernel
plus concentric shells. Inner structure is reusable, middle structure is
semi-general, and outer structure is task-local residue. Pruning is outside-in,
promotion consolidates useful motifs inward, and demotion/recycling moves stale
or too-specific structure outward.

**Implementation location.**

- `hibacaml/nodes/micro_columns.py`
- `hibacaml/graph/builder.py:create_hibacaml_structure`
- `hibacaml/control/shells.py:ShellController`

**Implementation details.**

The code maps the paper's `B/K/L` roles to three node families:

- `ShellBankInputNode` as `b_micro`, fed by gated patch tokens.
- `ShellBankRecurrentNode` as one or more `k_micro` depth nodes, with recurrent
  self-dynamics across settling steps.
- `ShellBankResidualNode` as `l_micro`, receiving skip inputs from its own
  column's `b_micro` and `k_micro` nodes, and lateral inputs from the last
  `k_micro` depth node of the other columns.

Each shell-bank node stores separate weights/biases for `kernel`, `tier1`,
`tier2`, and `tier3`. The dimensions come from:

- `memory_dim = 8` for `kernel`;
- `shell_sizes = (4, 6, 8)` for tiers 1, 2, and 3;
- `kernel_depth = 2` by default, so the graph has two kernel recurrent nodes
  per column even though the conceptual paper summary says three typed
  micro-columns.

The CIFAR configuration uses `memory_dim = 32`, `shell_sizes = (10, 20, 30)` and
`kernel_depth = 3`, i.e. 92 units per shell node. The kernel units of shared columns' `k_micro`
nodes can optionally read only the kernel coordinates of their inputs
(`shared_k_connectivity = "kernel-only"`, used by Split-CIFAR-10).

**Role labels (Split-CIFAR-10).** `hibacaml/control/roles.py` gives every shell unit a tier
*label* (0 kernel, 1-3 shells) separate from its physical storage slot. Labels
start equal to the physical layout; kernel units always keep label 0 and each node
keeps its physical tier sizes. Labels decide per-unit write rates and certificate
tier statistics, and relabelling two units never moves a parameter or optimizer
moment. Shell adaptation (section 3) promotes and demotes units by exchanging
labels.

`ShellController.apply_structural_edits` implements the radial mechanics:

- same-tier inhibition by reducing `log_precision_*` values for redundant
  units;
- outside-in pruning for tier3 and tier2 when occupancy exceeds semantic
  targets;
- promotion-like swaps from outer to inner shells when outer score exceeds
  inner score by `replacement_margin_base`;
- demotion is not applied directly in this routine; it goes through the audited
  `demotion_swap_audit` path.

Structural observations are reduced by an un-jitted JAX projection and copied
to the host in one grouped transfer before the ordered comparisons. Demotion candidates use a separate projection because their audit
observes the parameter tree after structural edits. Prune and swap mutations
use mask-wise selection while preserving the selected units and parameter
values exactly.

`ShellController` runs only when `enable_structural_edits` is on (the Split-MNIST
configuration). The CIFAR runners turn it off; Split-CIFAR-10 uses the audited shell
adaptation of section 3 instead.

**Alignment status.** Substantially faithful with implementation-specific node
names. The radial shell semantics are implemented, including pruning,
inhibition, promotion-like swaps, and audited demotion. The hard kernel is
non-prunable and non-inhibited: pruning touches only tier2/tier3 and inhibition
only tiers 1-3. It stays swappable through two paths, the audited
`kernel <-> tier1` demotion swap and the per-step `tier1 -> kernel` promotion
inside `apply_structural_edits`, which is not audited. Kernel units also take
part in precision-weighted gradient scaling.

### 2.4 Internal Probabilistic State and Certificates

**Paper purpose.** The paper argues that internal state should flow upward into
selection. Certificates summarize reusable shared mass, specificity load,
demotion pressure, saturation, and similarity signatures. The theorem on
certificates states that informative internal certificates improve reuse-utility
estimation compared with external fit scores alone.

**Implementation location.**

- `hibacaml/types.py:ColumnCertificate`
- `hibacaml/types.py:ShellStats`
- `hibacaml/control/certificates.py:CertificateController.refresh_certificates`
- `hibacaml/control/certificates.py:CertificateController.certificate_matrix`
- `hibacaml/control/scoring.py:SupportScorer._certificate_reuse_score`

**Implementation details.**

`CertificateController.refresh_certificates` owns observation and certificate
production. Its un-jitted JAX projection reduces graph parameters and `z_mu`
to six compact column/node/shell arrays. One grouped `jax.device_get` transfers
those arrays, after which shell EMAs and certificate fields are aggregated in
Python with float64 arithmetic. Certificate fields include:

- `q_mean`, derived from sigmoid shell precisions;
- `prec_mean`, derived from exponentiated shell precision parameters;
- `pred_mean`, from mean predicted activation magnitude;
- `live_frac`, the kernel shell's occupancy only — not the all-shell mean;
- `tier_q` and `tier_occ`, the per-tier precision and occupancy means;
- `shared_abstraction_mass`;
- `specificity_load`;
- `demotion_pressure`;
- `saturation`, the mean occupancy across all shells (kernel plus all tiers),
  which is the all-shell figure `live_frac` is easily mistaken for;
- `similarity_signature`, computed as cosine similarity between certificate
  vectors.

Similarity uses a vectorized float32 host calculation and is export-only; no
controller decision consumes it. Similarity uses all 14 certificate fields,
while the composer gathers a 10-field subset. That composer stack
is cached by persistent-state identity and `certificates_revision`; refreshing
certificates increments the revision once and invalidates the stack, while a
parameter revision alone does not.

`certificate_matrix` masks the cached stack by active support, and
`certificate_stack_for_masks` masks it for a stack of supports at once. Certificates
refresh every `cert_refresh_interval` training updates. Split-MNIST also refreshes
them after every update's structural edits and in its controller audits, and the
predictive-coding learner refreshes them on every training batch while building
its clamps. Reporting evaluations never refresh certificates.

When role labels are installed (Split-CIFAR-10), tier statistics are aggregated over the
units carrying each label rather than over physical slices. A relabel therefore
changes the certificates at the next scheduled refresh, not immediately.

The static support scoring path also uses `_certificate_reuse_score`, which
rewards `q_mean` and
`shared_abstraction_mass` while penalizing specificity, demotion pressure, and
saturation. In `default` mode, `certificate_support_weight` is set to
`0.05`.

**Alignment status.** Faithful as a compact certificate channel. The
implementation does not maintain a full Bayesian posterior over internal
motifs; it uses precision-like parameters, EMAs, occupancy metrics, and
certificate summaries as a practical proxy.

### 2.5 Top-Level Controller and Teacher-First Search

This controller is used by the Split-MNIST runner. The CIFAR runners turn it off
(`enable_exact_search = False`); Split-CIFAR-10 uses restricted column reuse
instead (section 3).

**Paper purpose.** The teacher-first regime uses exact combinatorial search at
task boundaries, a multi-objective criterion coupling current-task fit with
old-task retention, a support-switching penalty, local one-swap audits during a
task, and short rollouts to choose low-dimensional shell-controller settings.

**Implementation location.**

- `hibacaml/control/search.py:ExactSearchService`
- `hibacaml/control/support.py`
- `hibacaml/config/defaults.py:ExactSearchConfig`
- `hibacaml/types.py:BoundaryBundle`, `SupportSearchRow`,
  `ControllerSearchRow`, `LocalSwapRow`, `DemotionSwapAuditRow`

**Implementation details.**

Boundary search follows this control flow:

```text
train_task(task)
  -> boundary_search(task_id) if no support is already set
     -> make_bundle(task_id)
        -> current task eval batches       (validation split)
        -> rollout train batches           (fit split)
        -> worst-old eval batch            (validation split)
        -> mixed-old fragments             (validation split)
     -> enumerate_nonshared_supports(cfg)
     -> SupportScorer.static_rows(...)
     -> support posterior ranking
     -> optional reserve recruitment
     -> shortlist top support candidates
     -> phi_candidates()
     -> rollout_rows(...) per shortlist entry, one trajectory per phi class
     -> replay-bank reselection
     -> set_boundary_choice(...)
  -> set_current_support(...)
```

The boundary objective is:

```text
current_first_loss
+ current_remaining_loss
+ exact_old_worst_weight * old_worst_loss
+ exact_old_mix_weight * old_mix_loss
+ switch_penalty
```

`rollout_rows` scores every phi candidate at one support. A trajectory clones
the trainer, trains over the rollout batches, applies the same gradient masking
and shell edits as training, and re-evaluates the boundary objective. The phi
neighborhood is deterministic: the current four-value `PhiConfig` plus bounded
+/- perturbations of each coordinate.

Two exact reductions shrink the physical work while keeping the logical grid.
Phi reaches a trajectory only through `apply_structural_edits`, which reads
`outer_quantile`, `middle_quantile`, and `replacement_margin_base`;
`demotion_min_role_gain` is read only by `demotion_swap_candidates`, which no
rollout runs. Candidates sharing `rollout_trajectory_key` therefore drive one
trajectory, and `_controller_row` rebuilds each logical row from it with that
row's own penalty. Separately, `training_update` takes no phi at all, so a
support's first update runs once on a prefix clone and every trajectory resumes
from it. At the default settings this turns 72 logical rows into 56
trajectories and 288 update invocations into 176, leaving all 72 rows, their
encounter order, and strict-`<` winner selection unchanged. `boundary_search`
records both the logical and the physical counts in `timing_summaries`.

`resume_clone` exists because `clone()` deliberately drops `_last_graph_state`.
A trial resuming from its own preceding update must keep it: predictive coding
refreshes certificates while the next clamp set is built, and without that
observation the refresh would run against a zero placeholder and move
decision-bearing certificate fields.

The old-task audit terms (`old_worst_loss`, `old_mix_loss`) are memoized in
`SupportScorer._old_audit_cache`, keyed by trainer identity, bundle identity,
`params_revision`, and `certificates_revision`. Trainer identity must come from
`HiBaCaMLTrainer.instance_id`, a monotonic counter, and **not** from `id(trainer)`:
every rollout clone reaches the same revisions and is released as soon as its
trajectory ends, so CPython reuses the address and distinct candidates would
collide on one cache entry and silently share each other's old-task losses.
`make_bundle` clears the cache, so an entry never outlives the bundle it was
keyed on, and a rollout or demotion trial declines to store at all: its entry
keys on an instance id the parent never uses, so it could only ever be written
and never read.

This audit cache is intentionally distinct from ordinary evaluation. General
loader evaluation is not cached, because certificates, loader position, RNG and
other state can affect its result. Every evaluation request therefore executes,
advances its normal RNG transition, and performs a requested certificate refresh. Both revision
counters remain because the audit cache depends on parameters and
the current certificate inputs.

During training, when exact search is enabled, `HiBaCaMLTrainer.train_task` runs
`local_one_swap` every `maintenance_interval` steps for tasks after task 0. It also
runs `demotion_swap_audit` at a configured interval when that audit is enabled.

**Alignment status.** Strong match for teacher-first control and local
counterfactual teachers. The replay-bank reselection layer is an implementation
extension beyond the core paper narrative: it stores prior accepted supports and
can propose similar support candidates using a file-backed `SelectorBank`.

### 2.6 Local One-Swap Monotonicity

**Paper purpose.** The paper prove a deliberately local theorem: if an exact
one-swap teacher evaluates all neighbors and applies only an improving swap,
the audited local objective cannot worsen.

**Implementation location.**

- `hibacaml/control/search.py:ExactSearchService.local_one_swap`
- `hibacaml/control/support.py:one_swap_neighbors`
- `hibacaml/types.py:LocalSwapRow`

**Implementation details.**

`one_swap_neighbors` enumerates supports produced by replacing one currently
active nonshared adaptive column with one inactive adaptive column. For the
default configuration this is `3 * (15 - 3) = 36` neighbors, not the 45
neighbors implied by a 15-candidate inactive set in the paper text.

`local_one_swap` scores the current support, all local neighbors, and filtered
replay-bank candidates under the penalized objective. It accepts the best
candidate only if the penalized gain exceeds `local_swap_margin`. It records all
audited rows in `local_swap_tables`.

**Alignment status.** The monotone-teacher idea is implemented: swaps require a
strict positive margin. The exact neighborhood differs from the paper's simple
count because the implementation's ordinary pool excludes reserves and because
inactive count is computed from the adaptive pool.

### 2.7 Bayesian Resistance / Precision-Weighted Updates

**Paper purpose.** The paper states that higher posterior precision inside
modules should reduce damage from mistaken gate openings. In ColBa terms,
inner/reusable motifs should resist destructive perturbation more than outer
task-local material.

**Implementation location.**

- `hibacaml/nodes/micro_columns.py:ShellBankInputNode`
- `hibacaml/nodes/micro_columns.py:ShellBankRecurrentNode`
- `hibacaml/nodes/micro_columns.py:ShellBankResidualNode`
- `hibacaml/control/shells.py:precision_weight_gradients`

**Implementation details.**

Shell nodes include `log_precision_{shell}` biases. Forward passes multiply
activated shell outputs by `sigmoid(log_precision)`. During optimizer updates,
`precision_weight_gradients` preconditions shell gradients by an inverse factor
derived from the current precision:

```text
factor = max(floor, 1 / (1 + strength * sigmoid(log_precision)))
```

This reduces updates to higher-precision shell weights and biases when
`enable_precision_update_resistance` is true (the Split-MNIST configuration; the
CIFAR runners turn it off, and Split-CIFAR-10 protects shared structure with exact
write policies instead, section 3).

**Alignment status.** Partially implemented. The code has a precision-resistance
mechanism consistent with the theorem's intent, but it is not full Bayesian
posterior inference with covariance matrices. It is a practical scalar/vector
precision proxy integrated into the gradient path.

### 2.8 Hierarchical Bias and Composer

**Paper purpose.** The ColBa Split-MNIST setup uses patch-level processing plus
coarser quadrant/global auxiliary targets, with consistency losses, to bias the
system toward local-to-global compositional structure. A small attention-style
composer combines active columns.

**Implementation location.**

- `hibacaml/data/common.py:_hierarchy_targets`
- `hibacaml/graph/builder.py:create_hibacaml_structure`
- `hibacaml/nodes/composer.py:ColumnComposerNode`
- `hibacaml/nodes/composer.py:ScaledAddNode`
- `hibacaml/training/shared.py:hierarchy_parent_child_penalty`

**Implementation details.**

`_hierarchy_targets` builds quadrant-aware soft labels by measuring normalized
image mass in each quadrant. `hier_mid` receives shape
`(mid_targets, output_dim)`, while `hier_global` receives the global task target.

#### Hierarchy target construction and cross-entropy losses

The hierarchy losses are auxiliary objectives for the current example. They are
not losses from previous tasks. Previous-task quantities such as
`old_worst_loss` and `old_mix_loss` belong to support search and
continual-learning evaluation instead.

The **global target** is the same task-local class target used by the main
classifier. With the default task-local binary heads, a Split-MNIST task such as
`(2, 3)` maps digit `2` to `[1, 0]` and digit `3` to `[0, 1]`. The
`hier_global` head reads `active_feature_summary` directly and predicts this
target. Here, "global" means global with respect to the aggregate active-column
representation; it does not mean a ten-digit target or a target spanning
previous tasks.

The **mid-level targets** are quadrant-aware soft class targets. With the
default `mid_targets = 4`, `_hierarchy_targets` splits each normalized image
into four quadrants. For quadrant `q`, it computes

```text
m_q     = mean(abs(image_quadrant_q))
alpha_q = m_q / max(max_r(m_r), 1e-6)
```

Let `y` be the example's one-hot task target and let `u` be the uniform class
distribution. The target for quadrant `q` is

```text
y_mid_q = u + alpha_q * (y - u)
```

Thus a quadrant with greater mean absolute normalized activation receives a
target closer to the hard class label, while a lower-activation quadrant
receives a softer, less certain target. For a binary example with
`y = [1, 0]`:

| `alpha_q` | Mid-level target |
|---:|:---|
| `1.0` | `[1.00, 0.00]` |
| `0.8` | `[0.90, 0.10]` |
| `0.3` | `[0.65, 0.35]` |
| `0.0` | `[0.50, 0.50]` |

The `hier_mid` head produces one class distribution for each mid-level target,
so its default output shape is `(4, output_dim)`. Its input is still the shared
`active_feature_summary`; the quadrant information affects target construction,
not the tensor fed directly into the head. If `mid_targets` is not `4`, the
implementation falls back to repeating the hard class target that many times
instead of constructing quadrant-aware targets.

For one example, the supervised cross-entropy portion of the objective is

```text
L_CE = CE(y, p_task)
     + lambda_mid * sum_q CE(y_mid_q, p_mid_q)
     + lambda_global * CE(y, p_global)
```

The default weights are `lambda_mid = 0.06` and `lambda_global = 0.03`; the
main task cross-entropy has weight `1.0`. The mid-level implementation sums over
the four target distributions before the batch mean.

The parent-child consistency term additionally encourages the average
mid-level prediction to agree with the global prediction:

```text
p_mid_parent = mean_q(p_mid_q)
L_parent_child = lambda_parent * mean_class(
    (p_mid_parent - p_global) ** 2
)
```

Its default weight is `lambda_parent = 0.04`. The reported composite also
includes the composer auxiliary penalty. Both learners differentiate all five
named terms. In backprop, targets remain outside the graph and the five terms
are differentiated end to end. In predictive coding, the three weighted
cross-entropies are graph energies on clamped target nodes, while the
parent-child and composer terms are explicit factors: `HiBaCaMLPCInference` adds
them to the settling latent gradients, and the PC update adds their local
parameter gradients for the nodes producing their operands (EXPERIMENTS.md,
section 2.2).

The graph adds:

- `active_feature_summary` as the mean active column feature summary;
- `hier_mid`, a weighted softmax head over midpoint/quadrant targets;
- `hier_global`, a weighted softmax head over the global target;
- a parent-child penalty that encourages the mean midpoint prediction to match
  the global prediction.

`ColumnComposerNode` stacks per-column gated features and certificate vectors,
builds a certificate-derived prior, applies query-conditioned residual attention
over active columns, and produces a correction vector. Its gate keeps at most
`composer.topk` active columns (default 3): it keeps exactly the indices returned
by `jax.lax.top_k`, so tied scores can never enlarge the selected set. `ScaledAddNode` combines
the pooled column logits with this correction.

The composer has two entry points over a shared core. The node's `forward` calls
`composer_correction`, which stops at the correction vector, because that is all
the graph needs. `composer_details` continues on to the gate
auxiliary penalty (`aux_penalty`, part of the objective) and the reported
diagnostics (`gate_entropy`, `prior_kl`, `gate_dev`, `top1_mass`,
`effective_k`); it is called separately from the settled state by
`composer_details_from_runtime`. Keep new diagnostics out of the forward path:
anything added there is computed on every graph execution, including compiled
backprop updates, whether or not it is read.

**Alignment status.** Faithful in purpose. The implementation's composer works
over per-column feature vectors and certificates rather than an explicitly
per-token representation at the final combiner. Patch tokens are processed
inside the column graph before feature pooling.


## 3. Split-CIFAR-10 Mechanisms

CIFAR-10 images need more than raw patches, and the MNIST controller's exhaustive
search does not scale to a 40-column pool. The CIFAR path therefore adds four
pieces around the same column graph. The controllers are specified in full in
[COLUMN_REUSE_AND_SHELL_ADAPTATION.md](COLUMN_REUSE_AND_SHELL_ADAPTATION.md).

**Convolutional stem.** Five 3x3 convolutions (channels 32, 64, 64, 128, 128;
strides 1, 2, 1, 2, 1) and a 2x2 average pool turn a 32x32x3 image into a 4x4
grid. Each grid cell plus its two normalized coordinates is one token, giving 16
tokens of 130 dimensions (`HiBaCaMLConfig.token_grid`, `patch_token_dim`). The
stem is shared by all columns.

**Exact write protection.** In Split-CIFAR-10, from the second task on, a
`WritePolicy` (`hibacaml/training/write_policy.py`) decides which parameter
coordinates an update may change and by how much. It freezes the stem and the
composer's feature/output maps, lets only the current support's columns learn,
freezes the feature pools of shared and reused columns, and scales each shell
unit's update in shared and reused columns by its label (kernel units frozen;
tiers 1-3 at increasing rates). The task's fresh columns learn at the full rate.
Full-CIFAR-10 installs no policy; its active columns train throughout. The policy masks gradients before
clipping, masks and scales the AdamW updates, then restores the exact original
bytes of every protected coordinate.

**Restricted column reuse.** At each later task boundary, the runner compares
"fresh" (four unused columns) against variants that replace one fresh column by a
column learned on an earlier task. Each variant trains briefly on an isolated copy
of the model; it is adopted only when a validation objective that weighs the new
task and the worst and mean old task improves under a simultaneous bootstrap lower
bound, and no old task loses more than 1% of its validation predictions relative
to the fresh condition. Only the
chosen support and write policy reach the real model.

**Audited shell adaptation.** At scheduled epochs during each later task, the
controller compares the current model against three kinds of edit: promotion of
frequently used outer-shell units inward by relabelling (which leaves the function
unchanged), same-tier inhibition of redundant shell gates, and recoverable pruning
of weak units in the task's fresh columns. Every parameter edit is evaluated on an
isolated copy against all seen validation tasks and installed only when it passes
the same objective and bootstrap test plus per-edit and cumulative old-task harm
caps.

## 4. Exact Matches, Deviations, and Missing Ideas

### Exact or Close Matches

- Two-level architecture: implemented through support control plus internal
  shell/certificate control.
- Split-MNIST five-task setup: implemented exactly as binary digit pairs.
- Column pool size and partition: defaults match 20 total, 2 shared, 15
  adaptive, 3 reserve.
- Sparse active support: 2 shared plus 3 nonshared columns.
- Hierarchical bias: implemented via quadrant/global targets and consistency
  penalty.
- Internal certificates: implemented as `ColumnCertificate` summaries and used
  in composer priors and support posterior scoring.
- Teacher-first boundary control: implemented with exact enumeration,
  shortlisting, phi rollout scoring, and audited objective terms.
- Local one-swap teacher: implemented with strict improvement margin.
- Demotion-swap teacher: implemented as conservative audited internal swaps.
- Precision resistance: implemented as precision-conditioned gradient scaling.

### Deviations and Simplifications

- Candidate count differs: paper text cites `18 choose 3 = 816`; code ordinary
  search is adaptive-only `15 choose 3 = 455`, with reserve candidates added
  only through reserve recruitment.
- One-swap neighborhood count differs: code adaptive-only neighbors are
  `3 * 12 = 36` in the default state, while the paper's simplified count uses
  a larger inactive pool.
- Task-local heads are represented by target remapping plus task query and a
  shared two-output graph, not five physically distinct heads.
- Patch tokenization keeps the paper's raw patches and coordinates but has
  no shared learned embedding, which every task would train outside the support
  mask; the Split-MNIST runner uses one flattened `28 x 28` token instead of
  sixteen `7 x 7` patches.
- The three paper micro-column roles map to `b_micro`, recurrent `k_micro`
  depth nodes, and `l_micro`, with implementation-specific lateral wiring.
- Internal Bayesian state is a precision/certificate approximation rather than
  full posterior inference.
- All four runners train with backprop; the MNIST runners can also use the
  predictive-coding learner.
- The composer operates over pooled column features rather than directly
  exposing a final per-token attention output.
- Paper-reported offline audit result tables are not part of the implementation
  documentation and are not used here.
- CIFAR-10 adds a shared convolutional stem in front of the columns, which the
  papers do not specify.
- For Split-CIFAR-10, exhaustive support search is replaced by restricted column
  reuse (one learned column may replace one fresh column), and the per-step shell
  edits are replaced by audited shell adaptation rounds at fixed epochs.
- Shell promotion/demotion on Split-CIFAR-10 exchanges unit labels rather than moving
  parameters between physical slots.

### Missing or Future-Facing Ideas

- Learned top-level selector policy: not implemented as the primary controller.
  Exact search and replay-bank proposal are the implemented mechanisms.
- Class-incremental CIFAR-10 (a shared ten-way head) is not implemented; the
  CIFAR continual-learning runner is task-incremental.
- Reinforcement-learning extension: conceptual only.
- Transformer/wavelet integration: conceptual only.
- Quantitative verification of adequacy parameters `epsilon_app`, `g`, and `h`:
  not implemented as a measurement suite.
- Explicit confusion-graph analysis over tasks: only indirect overlap/support
  diagnostics exist.


## 5. Pseudocode Summary

The Split-MNIST path can be summarized as:

```text
cfg = make_hibacaml_config("default")
tasks = build_split_mnist_tasks(cfg, limit=None)
structure = create_hibacaml_structure(cfg, inference, FeedforwardStateInit())
params = initialize_params(structure, seed)
trainer = HiBaCaMLBackpropTrainer or HiBaCaMLPCTrainer(cfg, structure, params, tasks)

for task in tasks:
    if task has no current support:
        bundle = boundary audit data
        support_rows = score all adaptive supports
        maybe add reserve-recruitment supports
        shortlist = best support posterior rows
        controller_rows = rollout(shortlist x phi_candidates)
        accepted_support = replay/local/original reselection(best rollout)
        trainer.set_current_support(task, accepted_support, phi)

    for epoch, batch in task.train_loader:
        params, optimizer_state, losses, state = compiled_update(
            params, optimizer_state, batch, support, rng
        )
        params = shell_controller.apply_structural_edits(params, state, support, phi)
        audit demotion swaps periodically
        refresh certificates periodically
        audit local support one-swaps periodically

    freeze support snapshot
    evaluate all frozen supports
    export_task_artifacts(trainer, task_id)
    save_checkpoint(trainer, task_id)
```

The Split-CIFAR-10 path replaces boundary search and per-step shell edits:

```text
cfg = make_hibacaml_config("cifar") with the Split-CIFAR-10 overrides
tasks = build_split_cifar_tasks(cfg)
trainer = HiBaCaMLBackpropTrainer(cfg, structure, params, tasks)

for task in tasks:
    if task is not the first:
        protect the stem, composer maps and shared columns (write policy)
    condition = restricted column reuse:
        fresh draw vs. one-learned-column variants, each trained briefly
        on an isolated copy and scored on all seen validation tasks
    install condition's support and write policy on the real model

    for epoch in range(epochs_per_task):
        for batch in task.train_loader:
            params, optimizer_state = compiled_update(params, optimizer_state,
                                                      batch, support, write_policy, rng)
            refresh certificates periodically
        if a shell-adaptation round is scheduled:
            audited relabel / inhibition / pruning (at most one parameter edit)
        validate

    freeze support snapshot
    evaluate every seen task under its saved support
    write run records
```

