# HiBaCaML Implementation Details

This document maps the HiBaCaML/ColBa ideas described in the papers to the implementation under `hibacaml/` and the Split-MNIST experiment runner in `experiments/split_mnist.py`.

## 1. Executive Mapping

The codebase implements this as a FabricPC/JAX graph with:

- a fixed pool of 20 column modules, configured by `ColumnPoolConfig`;
- a support mask that activates 2 shared columns plus 3 selected nonshared
  columns;
- shell-bank column internals with kernel, tier1, tier2, and tier3 channels;
- a certificate and shell-controller subsystem that summarizes and edits column
  internals;
- an exact teacher-first controller that searches support candidates, runs
  rollout audits over phi controller settings, and performs one-swap
  maintenance;
- a Split-MNIST runner that builds five binary tasks and trains sequentially.

The implementation is a faithful structural instantiation of the paper's
HiBaCaML tuple, but it also contains pragmatic additions and simplifications:
the current support search enumerates adaptive-only candidates first, reserves
enter through a separate recruitment path, the "Bayesian" internal state is
represented by precision-like parameters and certificates rather than full
posterior inference, and the default experiment script uses a backprop runner
while preserving the same controller and column semantics.

## 2. Paper Concepts to Implementation

### 2.1 HiBaCaML Tuple: `H = (N, C, delta, pi, T)`

**Paper purpose.** The paper define HiBaCaML as a family of structurally
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

**Alignment status.** Faithful at the architectural level. The code realizes the
paper's tuple explicitly, but the controller is currently exact/audited rather
than a learned probabilistic selector.

### 2.2 Structurally Restricted Component Learners / Columns

**Paper purpose.** ColBa columns are not arbitrary dense experts. They are
restricted modules intended to align with recurring causal mechanisms. The AGI
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
the AGI draft describes searching all `18 choose 3 = 816` nonshared sets from
adaptive plus reserve columns, while the code's ordinary enumeration uses only
adaptive columns (`15 choose 3 = 455`) in `enumerate_nonshared_supports`.
Reserve candidates are introduced conditionally through
`enumerate_reserve_recruitment_supports` when saturation/posterior diagnostics
trigger recruitment.

### 2.3 Micro-Columns and Shell Semantics

**Paper purpose.** The paper describe three typed micro-columns `K`, `L`, and
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
- `ShellBankResidualNode` as `l_micro`, receiving skip inputs from `b_micro` and
  lateral inputs from other columns' `k_micro` nodes.

Each shell-bank node stores separate weights/biases for `kernel`, `tier1`,
`tier2`, and `tier3`. The dimensions come from:

- `memory_dim = 8` for `kernel`;
- `shell_sizes = (4, 6, 8)` for tiers 1, 2, and 3;
- `kernel_depth = 2` by default, so the graph has two kernel recurrent nodes
  per column even though the conceptual paper summary says three typed
  micro-columns.

`ShellController.apply_structural_edits` implements the radial mechanics:

- same-tier inhibition by reducing `log_precision_*` values for redundant
  units;
- outside-in pruning for tier3 and tier2 when occupancy exceeds semantic
  targets;
- promotion-like swaps from outer to inner shells when outer score exceeds
  inner score by `replacement_margin_base`;
- demotion is not applied directly in this routine in V20.2b; it goes through
  the audited `demotion_swap_audit` path.

**Alignment status.** Substantially faithful with implementation-specific node
names. The radial shell semantics are implemented, including pruning,
inhibition, promotion-like swaps, and audited demotion. The "hard kernel is
non-prunable" idea is approximated by only pruning tier2/tier3 in
`apply_structural_edits`; kernel units are still involved in audited swaps and
precision-weighted gradient scaling.

### 2.4 Internal Probabilistic State and Certificates

**Paper purpose.** The paper argue that internal state should flow upward into
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

`ShellController.refresh_certificates` updates EMAs and computes a
`ColumnCertificate` per column. Certificate fields include:

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

The certificate vector used by the composer is constructed by
`certificate_matrix`, masked by the active support. `certificate_vectors` builds
the unmasked vectors and `mask_certificate_vectors` applies one mask, so paths
that score many supports against one certificate state build the vectors once.
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
        -> current task eval batches
        -> rollout train batches
        -> worst-old eval batch
        -> mixed-old fragments
     -> enumerate_nonshared_supports(cfg)
     -> static_support_scores_batched(...)
     -> support posterior ranking
     -> optional reserve recruitment
     -> shortlist top support candidates
     -> phi_candidates()
     -> rollout_score(...) for each shortlist x phi
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

`rollout_score` clones the trainer, trains over the rollout batches, applies the
same gradient masking and shell edits as training, and re-evaluates the boundary
objective. The phi neighborhood is deterministic: the current four-value
`PhiConfig` plus bounded +/- perturbations of each coordinate.

The old-task audit terms (`old_worst_loss`, `old_mix_loss`) are memoized in
`ExactSearchService._old_audit_cache`, keyed by trainer identity, bundle
identity, and `params_revision`. Trainer identity must come from
`HiBaCaMLTrainer.instance_id`, a monotonic counter, and **not** from `id(trainer)`:
every rollout clone reaches the same `params_revision` and is freed with an
explicit `gc.collect()` before the next clone is allocated, so CPython reuses the
address and distinct candidates would collide on one cache entry and silently
share each other's old-task losses.

This audit cache is intentionally distinct from ordinary evaluation. General
loader evaluation is not cached: its former key omitted certificate, loader,
RNG, and other state that can affect the result, while observed savings were
negligible. Every evaluation request therefore executes, advances its normal
RNG transition, and performs a requested certificate refresh. The
`params_revision` counter remains because the audit cache still uses it.

During training, `HiBaCaMLTrainer.train_task` runs `local_one_swap` every
`maintenance_interval` steps for tasks after task 0. It also runs
`demotion_swap_audit` at a configured interval when enabled.

**Alignment status.** Strong match for teacher-first control and local
counterfactual teachers. The replay-bank reselection layer is an implementation
extension beyond the core paper narrative: it stores prior accepted supports and
can propose similar support candidates using a file-backed `SelectorBank`.

### 2.6 Local One-Swap Monotonicity

**Paper purpose.** The paper prove a deliberately local theorem: if an exact
one-swap teacher evaluates all neighbors and applies only an improving swap,
the audited local objective cannot worsen.

**Implementation location.**

- `hibacaml/control/search.py:local_one_swap`
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

**Paper purpose.** The v2 paper states that higher posterior precision inside
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
`enable_precision_update_resistance` is true.

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

- `hibacaml/data/mnist.py:_hierarchy_targets`
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
parent-child and composer terms are contributed by `HiBaCaMLPCInference` as
explicit factors that reach both the settling latent gradients and the local
parameter gradients of the nodes producing their operands (section 4.5).

Historical note: until 2026-09-19 the predictive-coding runner computed these
two terms only after `compute_local_weight_gradients`, so they appeared in the
reported loss while influencing neither settling nor the parameter update. Any
PC result recorded before that date carries the narrower three-term update, and
the PC-versus-backprop comparisons in the older reports therefore differ in both
the learning rule and the differentiated objective.

The graph adds:

- `active_feature_summary` as the mean active column feature summary;
- `hier_mid`, a weighted softmax head over midpoint/quadrant targets;
- `hier_global`, a weighted softmax head over the global target;
- a parent-child penalty that encourages the mean midpoint prediction to match
  the global prediction.

`ColumnComposerNode` stacks per-column gated features and certificate vectors,
builds a certificate-derived prior, applies query-conditioned residual attention
over active columns, and produces a correction vector. `ScaledAddNode` combines
stage-1 logits with this correction.

The composer has two entry points over a shared core. The node's `forward` calls
`composer_correction`, which stops at the correction vector, because that is all
the graph needs. `composer_details` continues on to the gate
auxiliary penalty (`aux_penalty`, part of the objective) and the reported
diagnostics (`gate_entropy`, `prior_kl`, `gate_dev`, `top1_mass`,
`effective_k`); it is called separately from the settled state by
`_composer_details_from_runtime`. Keep new diagnostics out of the forward path:
the backprop runner does not execute the graph under `jax.jit`, so anything
added there is computed on every forward whether or not it is read.

**Alignment status.** Faithful in purpose. The implementation's composer works
over per-column feature vectors and certificates rather than an explicitly
per-token representation at the final combiner. Patch tokens are processed
inside the column graph before feature pooling.

## 3. System Architecture

### 3.1 Main Modules

`hibacaml/config/defaults.py`
: Defines all public hyperparameters and mode presets. `make_hibacaml_config`
  supports `default` and `smoke`.

`hibacaml/data/mnist.py`
: Builds deterministic loaders for task-incremental Split-MNIST and the
  single-task ten-class Full-MNIST protocol, including task queries,
  hierarchy targets, and the stratified fit/validation split. This is the
  single canonical MNIST data module; the former `data/split_mnist.py` shim
  and `SplitMnistTask` alias have been removed.

`hibacaml/graph/builder.py`
: Builds the static FabricPC graph and initializes persistent HiBaCaML state.

`hibacaml/nodes/pathways.py`
: Defines patch-token preparation and support-gated graph pathways.

`hibacaml/nodes/micro_columns.py`
: Defines the input, recurrent, and residual shell-bank micro-column nodes.

`hibacaml/nodes/composer.py`
: Defines stage-2 composer mathematics and final output integration. The
  numerical detail helpers stay beside `ColumnComposerNode` so training,
  evaluation, and graph execution share one implementation.

This node layout is an ownership and readability boundary; it does not change
the graph or claim a runtime improvement.

`hibacaml/control/support.py`
: Encodes support-set arithmetic: shared-column inclusion, support masks,
  adaptive support enumeration, reserve recruitment candidates, and one-swap
  neighbors.

`hibacaml/control/certificates.py`
: `CertificateController` — everything that *reads* the graph's shells: shell-EMA
  statistics, per-column `ColumnCertificate` construction and similarity
  signatures, the certificate vectors the composer and support scorer consume,
  and the semantic penalty. It also owns the shell geometry helpers
  (`shell_slices`, `shell_node_names`, `shell_occupancy`, `effective_precision`,
  `mean_or_zero`) that `shells.py` reuses, so both halves agree on what a shell
  is.

`hibacaml/control/shells.py`
: `ShellController` — everything that *mutates* shells: inhibition, outside-in
  pruning, promotion swaps, and audited demotion swaps. It is constructed with
  the certificate controller because `apply_structural_edits` returns with
  certificates already refreshed for the edited parameters; the disabled path
  returns first and deliberately does not refresh, and that asymmetry is refresh
  cadence rather than an optimization. `precision_weight_gradients` stays a
  module-level function so the training update can close over plain configuration
  values instead of a controller object.

`hibacaml/control/ranking.py`
: Stateless calculations, defined by a property rather than a topic: a function
  belongs here if it is pure over its arguments and imports no controller,
  trainer, bank, or reporting module. Posterior ranking and summaries, reserve
  recruitment diagnosis, the phi neighbourhood, candidate chunking, and support
  set geometry.

`hibacaml/control/scoring.py`
: `SupportScorer` — audit-bundle construction, the Eq. (1) boundary objective,
  batched static support rows, and the old-task audit cache. The trainer is a
  parameter on every call rather than an attribute, because rollout scoring
  evaluates candidates on cloned trainers and the clone should be visible at the
  call site. The audit cache is the class's only state; its key covers trainer
  identity, bundle identity, and parameter revision.

`hibacaml/control/search.py`
: `ExactSearchService` — the decisions and their consequences: rollout
  coordination, replay-bank proposal and acceptance policy, boundary search,
  local one-swap and demotion-swap orchestration, installing accepted supports
  and parameters, controller tables, timing, and events. Everything it records is
  written here; the modules above compute but never persist.

`hibacaml/training/trainer.py`
: The abstract `HiBaCaMLTrainer` base: support selection, certificate refresh,
  clamp orchestration, gradient masking, structural edits, the
  `training_update` template, the task loop, and evaluation orchestration. Five
  operations are abstract —
  `build_runtime`, `_training_inputs`, `_gradients`,
  `_run_evaluation_inference`, and `_evaluate_prepared_batch` — and everything
  else is shared by both learners. The trainer prepares target-free clamps and
  external targets, owns support ordering and RNG transitions, refreshes
  certificates on the first requested loader batch, and installs graph state
  only for explicit raw inference.
  Also holds `feedforward_state`, the one FabricPC entry point both learners
  use, and `LearnerRuntime`, the compiled programs a trainer shares with its
  rollout clones.

  Two class constants declare where the learner contracts genuinely differ:
  `COMPOSER_BEFORE_PARENT` (objective summation order),
  `CLAMPS_MAY_INCLUDE_TARGETS` (whether supervised targets may be clamped into
  the graph at all). Every update returns its graph state; the former unused
  lean-update declaration and ignored `need_state` argument have been removed.

`hibacaml/training/shared.py`
: Trainer-independent computation, in three sections following one batch's
  journey. **Clamp assembly** builds the clamps for one support or a stack of
  candidate supports; the stateful half — the certificate refresh, whose cadence
  is scientific state — stays in the trainer, which derives the support mask and
  certificate vectors and hands them here. **Objective mathematics** holds
  cross-entropy, the hierarchy penalties, the parent-child term, composer
  details, and the PC training loss vector. **Evaluation** holds stateless
  scoring, target preparation, ordered multi-support loss reduction, per-class
  metrics, and the single stateful `EvaluationAccumulator`, whose state spans
  exactly one loader pass. One prepared batch is represented directly as
  `(logits, per_sample_total, loss_vector, composer)`; no request/result
  dataclass or named tuple is introduced. `composer_before_parent` preserves the
  learners' established difference in objective summation order.

  The module's identity is what it refuses to import: no trainer, controller,
  selector bank, persistent state, or reporting. Everything stateful is passed
  in as a plain argument. A test enforces that boundary rather than leaving it
  to convention. This module replaced the former `clamps.py`, `objectives.py`,
  and `evaluation.py`, which shared one dependency profile and were merged
  without changing a single function body.

`hibacaml/training/pc.py`
: `HiBaCaMLPCTrainer`, the compiled PC programs, and the full-native factor
  machinery. Feedforward initialization, the established latent-state relaxation
  with supervised targets clamped, and local graph-energy weight gradients. For
  evaluation it supplies only target-free inference and prepared-batch execution
  hooks.
  Also holds `HiBaCaMLPCInference`, an `InferenceSGD` subclass overriding only
  the latent-gradient phase. After the ordinary FabricPC sweep it adds the
  parent-child and composer auxiliary contributions, so both take part in
  settling. `_add_full_native_weight_gradients` then adds their local parameter
  gradients at the settled state. Each factor reaches only the parameters
  producing its operands -- the two hierarchy heads and `composer` -- because
  everything upstream learns through the settling state the factors altered and
  its ordinary local prediction errors. That is what keeps this predictive
  coding rather than global backpropagation.
  The factors act only when every supervised target is clamped, which is exactly
  training; a partial target set is rejected. `build_runtime` refuses a graph
  built with stock `InferenceSGD`, since that would silently restore the former
  report-only behavior. The auxiliary weights are the whole control surface:
  with `parent_child_loss_weight` and the three composer penalty weights at
  zero, both factors take an early return and the update reduces exactly to the
  pre-2026-09-19 algorithm. That is a regression property, not a supported mode.

`hibacaml/training/backprop.py`
: `HiBaCaMLBackpropTrainer`, an end-to-end autodiff learner preserving the same
  support, certificate, shell-edit, and audit semantics. Its forward pass is
  exactly the PC learner's initialization without the relaxation that follows
  it there. For evaluation it supplies only feedforward inference and
  prepared-batch scoring hooks.

Both learners are siblings under the abstract base rather than one inheriting
the other. The stage order of an update — gradient production, support masking,
the configured precision treatment, AdamW, then parameter application — is
fixed once in `training_update`, and rollout trials run that same template, so
no trial can substitute a different update algorithm.

`hibacaml/experiment.py`
: Holds the wiring both runners share: `build_trainer` (inference, graph,
  parameter initialization, runner selection) and `prepare_run_root`
  (exclusive run directory plus config re-rooting).

`hibacaml/reporting/`
: Three modules, one job each. `logger.py` is the progress logger, a
  stdlib-only leaf so any module can log without pulling JAX or matplotlib;
  `rollout_logging()` demotes clone progress lines to DEBUG. `export.py` owns
  snapshot assembly, task artifact export, checkpoint serialization, the final
  bundle, and the two files written while a run is in flight (`events.jsonl`,
  `heartbeat.json`). `build_run_snapshot` is deliberately stateful: it evaluates
  every saved support, advances evaluation RNG, and may refresh certificates.
  `support_diagnostics.json` is no longer produced; task-level support entropy
  and controller-relevant diagnostics remain. `plots.py` holds the figures for
  both protocols and selects the headless Agg backend on import.

`experiments/split_mnist.py`
: Wires configuration overrides, task construction, training, evaluation
  aggregation, summaries, and plots.

`experiments/mnist.py`
: Runs the static Full-MNIST full-bank protocol with fixed support, epoch-level
  fit/validation evaluation, a configurable bank size, and isolated
  PC/backprop artifact roots.

### 3.2 Data and Control Flow

The end-to-end flow is:

```text
make_hibacaml_config(mode)
override(cfg, ...)
run_experiment(cfg, ...)
  -> build_split_mnist_tasks(cfg, limit=tasks_limit)
  -> prepare_run_root(cfg, run_id)            # hibacaml/experiment.py
  -> build_trainer(cfg, tasks, learning)      # hibacaml/experiment.py
       -> create_hibacaml_structure(cfg, inference, graph_state_initializer)
       -> initialize_params(...)
       -> HiBaCaMLBackpropTrainer or HiBaCaMLPCTrainer
       -> log protocol / graph / schedule summary
  -> _run_tasks(trainer, ...)
  -> for each task:
       train_task(task)
         -> boundary support search if needed
         -> per-batch gradient computation
         -> mask gradients outside active support
         -> optimizer update
         -> shell structural edits
         -> certificate refresh
         -> optional demotion and one-swap audits
       evaluate_all_saved_supports()
       export_task_artifacts(trainer, task_id)
       save_checkpoint(trainer, task_id)
  -> build_run_snapshot(trainer)
  -> aggregate accuracy/forgetting/support trajectories
  -> write run summary/table and plots
```

The support mask is the central runtime control variable. It is clamped into the
graph as `support_mask`, and each `ElementwiseGateNode` reads its column's
scalar mask entry to zero or retain token, feature, and logit pathways.

### 3.3 Training Modes

The experiment script accepts `learning = "pc"` or `"backprop"`.

- `HiBaCaMLPCTrainer` uses FabricPC inference and local predictive-coding weight
  gradients through `compute_local_weight_gradients`, plus the two auxiliary
  factors described in section 4.5. `learning = "pc"` means full-native PC;
  there is no selectable report-only variant.
- `HiBaCaMLBackpropTrainer` uses feedforward state initialization and JAX
  autodiff for supervised losses, while preserving support masks, certificate
  refresh, gradient masking, shell edits, boundary search, local swaps, and
  evaluation semantics. Its target-free evaluation path is also feedforward-only
  and never falls through to iterative predictive-coding inference.

The script default is `"backprop"`. This is a practical deviation from a purely
predictive-coding training story, but it keeps the HiBaCaML controller and
architecture under test.

## 4. Split-MNIST Experiment Mapping

### 4.1 Paper Experiment

The AGI draft describes task-incremental Split-MNIST with five binary tasks:

```text
(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
```

Images are converted into sixteen non-overlapping `7 x 7` patches. Each patch
is embedded, coordinates are appended, and the columnar recurrent/predictive
coding stack processes the resulting token sequence. The paper uses task-local
two-way readouts to isolate representation forgetting from class-head
interference.

### 4.2 Dataset Handling in Code

`hibacaml/data/mnist.py` is the canonical data module for both protocols. It
sets `KERAS_BACKEND=jax`, loads MNIST through
`keras.datasets.mnist.load_data()`, and exposes `build_split_mnist_tasks` plus
`build_full_mnist_task`. There is no project-owned `mnist.npz` fallback.

`build_split_mnist_tasks` implements the five tasks from `_TASK_CLASS_PAIRS`.
Images are filtered to each task's two classes first and then normalized by
MNIST mean/std and reshaped to `28 x 28 x 1`, so no float32 copy of the full
dataset is held alongside the per-task copies.

The optional `limit` argument caps how many tasks are built and is what
`run_experiment`'s `tasks_limit` forwards. It deliberately does not go through
`cfg.num_tasks`, because that value also sizes the task one-hot in the
replay-bank context vector (`hibacaml/control/replay_bank.py`).

For each task:

- labels are filtered to the task's two classes;
- task-local targets map the first class to index 0 and the second to index 1
  when `task_local_heads` is true;
- hierarchy targets are computed;
- `_ArrayTaskLoader` provides deterministic iteration with optional shuffling
  and batch limits;
- a one-hot `task_query` vector is attached to the task.

`build_full_mnist_task` instead creates one ten-class task with
`task_local_heads=False`. A seed-controlled stratified split partitions the
official 60,000 training examples into 54,000 fit and 6,000 validation
examples; the official 10,000-example test set remains separate. Batch limits
apply independently to the three loaders and run metadata records their
effective iterable counts. See Section 4.7 for the fixed full-bank training and
evaluation protocol.

### 4.3 Patch Processing

`PatchTokenizerNode` implements the paper's patch-token step:

```text
image: (28, 28, 1)
  -> reshape into 4 x 4 grid of 7 x 7 patches
  -> flatten each patch
  -> learned linear projection to patch_embed_dim
  -> append normalized (y, x) coordinates
  -> output: (16, patch_embed_dim + coord_dim)
```

The default config sets `patch_embed_dim = 12` and `patch_coord_dim = 2`, so
each token has 14 dimensions.

### 4.4 Task Splitting and Heads

The code implements task-local binary classification by target remapping and
task queries rather than by creating five separate physical classifier modules.
`HiBaCaMLConfig.output_dim` is `2` when `task_local_heads = True`, and
`_task_targets` maps labels into a two-way target for each task.

This matches the paper's experimental intent: avoid conflating shared
representation forgetting with ten-way class-head interference. It is a
mechanical simplification relative to the phrase "each task receives its own
local readout head."

### 4.5 Training Procedure

Both runners construct their trainer through `hibacaml.experiment.build_trainer`,
which `run_experiment` calls once and then passes to the task loop. The pre-run
review is printed from that same structure rather than from a separate throwaway
graph:

- `InferenceSGD` is always configured for graph inference;
- `FeedforwardStateInit` is used explicitly for both runners;
- `create_hibacaml_structure` builds the graph;
- FabricPC `initialize_params` initializes graph parameters;
- trainer class is selected from `learning`.

For predictive coding, feedforward initialization only supplies a coherent
starting state. Supervised targets are then clamped, FabricPC performs the
configured iterative settling, and `compute_local_weight_gradients` computes
local graph-energy gradients. Backprop uses the same initializer but skips PC
settling and differentiates its external objective end to end.

The graph builder requires an explicit initializer and the HiBaCaML trainer
rejects non-feedforward initialization. This prevents the former silent
fallback to independently sampled hidden states.

For each task, `_run_tasks` calls `trainer.train_task(task)`. Training:

1. selects a support at the task boundary if no support is set;
2. computes gradients on each training batch;
3. masks gradients for inactive columns;
4. applies optimizer updates with precision resistance;
5. applies shell structural edits to active columns;
6. periodically refreshes certificates;
7. periodically performs demotion and local one-swap audits;
8. freezes the task support as a `SupportSnapshot`;
9. evaluates the task using its saved support.

`train_task` owns task-level setup and completion. `_train_epoch` owns epoch
aggregation and held-out evaluation, while `_train_batch` owns the established
per-update hook order. The split is organizational only: gradient production,
structural edits, step advancement, demotion audit, composer reporting,
certificate refresh, and local maintenance retain their previous order.

#### Predictive-coding clamping contract

The runner implements standard discriminative predictive coding. Clamped nodes are
held fixed for the whole settling process; every non-clamped node is updated by
`InferenceBase.update_latents`, which applies `z_latent -= eta_infer * latent_grad`
once per iteration for `infer_steps` iterations under `jax.lax.fori_loop`.

During **training**, `HiBaCaMLPCTrainer._training_inputs` clamps:

- the input image `x`;
- the HiBaCaML control inputs: `support_mask`, `task_query`, and the per-column
  certificate inputs;
- the supervised outputs `y`, `hier_mid`, and `hier_global`.

Everything between them settles. `FeedforwardStateInit` sets each clamped node's
`z_latent` to its clamp value while still computing `z_mu`, `error`, and `energy`
from the forward pass, so the output-layer prediction error exists at step 0 and is
what drives the settling. `compute_local_weight_gradients` then reads the settled
state.

During **evaluation**, the target clamps are dropped. Only the image and the control
inputs stay clamped, so `y` settles freely and the prediction is read from its `z_mu`.
This is the `target_free_inference_external_supervision_v1` protocol described in
section 4.6; the class target is applied afterward as an external cross-entropy and
never enters the hidden-state trajectory.

The clamp policy is expressed by the `include_targets` flag on
`HiBaCaMLTrainer._build_clamps`. It defaults to `False`, so target-free is the
default behavior and supervised PC training opts in explicitly. The base
trainer rejects target clamps for a learner whose `CLAMPS_MAY_INCLUDE_TARGETS`
contract is false, so backprop targets remain external in every path.

#### Full-native auxiliary factors

The parent-child and composer auxiliary terms enter PC learning as explicit
factors rather than as node energies, because neither fits FabricPC's node-local
`energy(z_latent, z_mu)` contract: parent-child couples two hierarchy
predictions, and the composer penalty depends on gate probabilities derived from
every active column's features, certificates, and the task query.

Each settling step therefore runs the ordinary FabricPC sweep, then pushes each
factor's `dE/dz_mu` back through the producing node's own `forward` with
`jax.vjp`, accumulating into the source nodes' `latent_grad` before latents are
updated. Parent-child reaches `active_feature_summary` through both hierarchy
heads; the composer reaches each column's feature branch through its
`feature_gate`.

Two numerical boundaries are deliberate. Factor gradients differentiate the
**batch sum**, matching FabricPC, which differentiates `jnp.sum(energy)` in both
`forward_and_latent_grads` and `forward_and_weight_grads`; the reported
components remain batch means. And the factors inherit FabricPC's existing
one-step offset: settling factors act on fresh in-step `z_mu`, the weight phase
recomputes predictions from the settled `z_latent`, and reporting keeps reading
the stored `z_mu`, exactly as the three cross-entropies already did. No final
refresh was added.

Both factors read `z_mu`, never `z_latent`. During supervised training the
hierarchy and output nodes are clamped, so their `z_latent` holds the labels;
a factor reading those would measure agreement between targets rather than
between predictions. Certificates, the task query, and the support mask are
clamped, so cotangents reaching them are inert by construction.

### 4.6 Evaluation Setup

After each task, `_run_tasks` calls `evaluate_all_saved_supports`, producing
an accuracy matrix over all completed tasks under their frozen saved supports.
Evaluation uses the `target_free_inference_external_supervision_v1` protocol:
the graph is clamped only to the image, support mask, task query, and certificate
inputs. The class target and the two class-derived hierarchy targets are kept
outside inference and are used only afterward to compute external
cross-entropies. This separation is essential for the predictive-coding runner,
where clamping test targets during settling would leak the answer into the
hidden-state trajectory. Per-task exports include evaluated/correct example
counts and a task-local confusion matrix so aggregate accuracy can be audited.

`HiBaCaMLTrainer` owns the evaluation request and its stateful effects. It
normalizes ordered supports, constructs single- or multi-support clamps, splits
the evaluation RNG once for each executed batch, and refreshes certificates on
the first loader batch only when requested. `pc.py` and `backprop.py` turn that
prepared batch into the same four-value detail tuple. `shared.py` scores an
already-produced graph state and accumulates example-weighted metrics. Ordinary
metric evaluation does not replace `_last_graph_state`; the explicit
`run_batch_evaluation_inference` path does because controller rollouts consume
that state.

The script then derives:

- mean seen accuracy;
- mean forgetting curve;
- support sequence;
- phi trajectory;
- support entropy trajectory;
- best one-swap gains.

These derived metrics operationalize the paper's continual-learning concerns:
retention across prior tasks, support reuse/churn, and the local value of
counterfactual support edits. This document intentionally does not interpret any
actual run outputs.

### 4.7 Static Full-MNIST Phase 1A

`build_full_mnist_task` constructs one ten-way task and requires
`task_local_heads=False`. It deterministically stratifies the official 60,000
training examples into 54,000 fit examples and 6,000 validation examples. The
official 10,000-example test set remains separate and is evaluated only after
training. The split seed, counts, update budget, and bank size are stored in the run
metadata.

The static full-bank configuration uses all adaptive columns:

```text
shared columns       = 2
active adaptive      = 15
active_support_size  = 17
inactive reserves    = (17, 18, 19)
composer top-k       = 3
```

`pooled_logits` and `active_feature_summary` retain their 20 gated input
edges, but their aggregation scale changes from the Split-MNIST default
`1 / 5 = 0.200` to the true full-bank mean `1 / 17 = 0.0588235`. This is an
architectural normalization change. It does not alter parameter shapes or
initial values, and the effective support size and scale are recorded in every
Full-MNIST run. Slow learning in both static arms should therefore prompt a
scale/path inspection before a learner-specific conclusion.

Static mode disables boundary search, structural shell edits, precision
resistance, demotion audits, and selector-state writes. The default support
path inside `train_task` selects adaptive columns 2 through 16; the experiment
runner asserts that support and never installs a duplicate override. The
assertion is derived from `topk_nonshared` and `active_support_size` rather
than from the literal full-bank numbers, so a smaller declared support is
checked the same way (Section 5.2).
`ExactSearchConfig.enable_structural_edits` is the authoritative controller
guard. When false, `apply_structural_edits` returns the same parameter object,
does not refresh controller state, and causes no extra parameter-revision bump.

Evaluation metrics come from the same forward/inference operation for both
learners. The PC JIT exposes its already-computed loss vector and composer
details; backprop uses the same state-scoring helper
(`shared.py:evaluation_details_from_state`).
`evaluate_batch_outputs` remains the compatible
`(logits, per_sample_total)` interface, while `evaluate_loader` delegates
aggregation to `EvaluationAccumulator` and additionally reports pure class
cross-entropy, the full composite, per-class precision/recall, and per-column
composer use. It never runs `composer_diagnostics_from_state` as a second
forward and never reuses a cached general-evaluation result.

For compatibility, Split-MNIST `evaluate_task` still refreshes certificates on
its first test batch by default. Full-MNIST passes
`refresh_certificates=False` explicitly for fit, validation, official test,
snapshot, and export calls. `build_run_snapshot` and `export_task_artifacts`
remain separate evaluation requests; neither reuses the other's result.
Consequently held-out data cannot advance shell
EMAs or recompute certificates. Epoch loss means are weighted by actual batch
size, including partial batches, and epoch evaluations are checkpointed and
exported but deliberately omitted from rollout clones.

Phase 1A does not refactor the established learner-specific training
certificate path: PC clamp construction can refresh before its training
settling, while backprop uses the scheduled interval. The run metadata records
the effective training certificate policy. Standardizing controller
observation state is intentionally deferred to the adaptive phase.

The result-bearing default layout is:

```text
runs/experiments/full_mnist_architecture/
    full_mnist_backprop_bank<BANK>/seed_<n>/
    full_mnist_pc_bank<BANK>/seed_<n>/
```

The PC arm was named `pc_local` before 2026-09-19, when its update omitted the
parent-child and composer terms. Existing `full_mnist_pc_local_bank*` bundles
keep that name: they record the older contract and are not comparable to a
full-native run.

At batch size 256 and five epochs, the 54,000-example fit split produces 211
updates per epoch and 1,055 updates total. The runner records expected and
completed counts and fails rather than silently accepting a shortened run.

## 5. Important Hyperparameters

### 5.1 Shared Defaults and Split-MNIST Overrides

The `default` mode's values are defined by `HiBaCaMLConfig` and nested
configs:

- seed: `0`
- input shape: `(28, 28, 1)`
- patch size: `(7, 7)`
- number of tasks: `5`
- batch size in config: `256`
- experiment script override: `OVERRIDES["batch_size"] = 768`
- epochs per task: `5`
- inference steps: `16` in normal mode and `4` in smoke mode
- inference rate: `0.05`
- optimizer: AdamW with learning rate `0.001` and weight decay `0.05`
- columns: `20 = 2 shared + 15 adaptive + 3 reserve`
- active nonshared columns: `3`
- shell dimensions: kernel `8`, tiers `(4, 6, 8)`
- composer hidden dimension: `64`
- composer top-k: `3`
- exact boundary shortlist: config default `8`
- experiment script static/neighbor support batch sizes: `16`
- experiment script audit data batch caps: `128` for all six of
  `boundary_current`, `rollout_train`, `boundary_worst_old`,
  `boundary_mixed_old`, `local_swap_audit`, and `demotion_audit`
  (`*_data_batch_size`); these bound how many examples each audit bundle draws,
  and the pre-run banner prints them as "search batch caps"
- maintenance interval: `64`
- demotion audit interval: `64`

The script-level constants at the top of `experiments/split_mnist.py` override
some config defaults for notebook/script runs. Documentation readers should
distinguish config defaults from script overrides.

`infer_steps` and `eta_infer` remain ordinary configuration hyperparameters.
The value 16 is the corrected normal-mode starting point, not a claim that it
is optimal for every graph or accelerator configuration.

### 5.2 Full-MNIST Phase 1A Overrides

`experiments/mnist.py` declares a separate static protocol over the same
20-column graph:

- seed: `0`;
- batch size: `256`;
- epochs: `5`;
- one ten-class task with `task_local_heads=False` and `num_tasks=1`;
- PC settling: `infer_steps=16`, `eta_infer=0.05`;
- active columns: 2 shared plus all 15 adaptive columns;
- inactive reserve columns: `(17, 18, 19)`;
- `topk_nonshared=15`, giving `active_support_size=17`;
- stage-1 and active-feature aggregation scale: `1/17`;
- composer top-k: exactly `3`;
- exact search, structural edits, precision resistance, demotion audits, and
  selector-state writes: disabled.

The bank size is the runner's one architectural knob. `BANK` in
`experiments/mnist.py` derives `adaptive_count = BANK - 2`,
`reserve_count = 20 - BANK`, and `topk_nonshared = adaptive_count`, so `BANK=17`
leaves columns `(17, 18, 19)` in reserve while `BANK=20` puts every column in
the bank at aggregation scale `1/20`. The two are architecturally identical --
the same 210 nodes and the same parameter shapes -- and differ only in the
aggregation scale and which columns are gated on.

The runner enforces one invariant, `topk_nonshared == adaptive_count`: every
non-shared column in the pool must be active. That is what distinguishes a full
bank of size `BANK` from a sparse subset of a larger pool, which this runner
rejects. Run directories are named from the bank (`full_mnist_backprop_bank17`),
so two bank sizes never collide.

The 54,000-example fit split yields 211 updates per epoch at batch size 256,
or 1,055 updates across five epochs. These values are protocol declarations,
not replacements for the Split-MNIST defaults above, and the Full-MNIST runner
records both expected and completed update counts in its artifacts.

## 6. Exact Matches, Deviations, and Missing Ideas

### Exact or Close Matches

- Two-level architecture: implemented through support control plus internal
  shell/certificate control.
- Split-MNIST five-task setup: implemented exactly as binary digit pairs.
- Patch tokenization: implemented as 16 non-overlapping `7 x 7` patches with
  learned embeddings and coordinates.
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
- The three paper micro-column roles map to `b_micro`, recurrent `k_micro`
  depth nodes, and `l_micro`, with implementation-specific lateral wiring.
- Internal Bayesian state is a precision/certificate approximation rather than
  full posterior inference.
- Backprop is the default runner in the script, although the graph and PC runner
  remain available.
- The stage-2 composer operates over pooled column features rather than directly
  exposing a final per-token attention output.
- Paper-reported offline audit result tables are not part of the implementation
  documentation and are not used here.

### Missing or Future-Facing Ideas

- Learned top-level selector policy: not implemented as the primary controller.
  Exact search and replay-bank proposal are current mechanisms.
- CIFAR scaling configuration: discussed in the paper, not implemented in this
  repository.
- Reinforcement-learning extension: conceptual only.
- Transformer/wavelet integration: conceptual only.
- Quantitative verification of adequacy parameters `epsilon_app`, `g`, and `h`:
  not implemented as a measurement suite.
- Explicit confusion-graph analysis over tasks: only indirect overlap/support
  diagnostics exist.

## 7. Pseudocode Summary

The implementation can be summarized as:

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
        grads, losses, state = trainer.compute_training_gradients(batch, support)
        grads = zero inactive-column gradients
        params = optimizer_update(params, precision_weighted_grads)
        params = shell_controller.apply_structural_edits(params, state, support, phi)
        refresh certificates periodically
        audit demotion swaps periodically
        audit local support one-swaps periodically

    freeze support snapshot
    evaluate all frozen supports
    export_task_artifacts(trainer, task_id)
    save_checkpoint(trainer, task_id)
```

## 8. Reader Guide

To understand the code as a concrete realization of the paper, read in this
order:

1. `hibacaml/config/defaults.py` for the paper-scale architectural constants.
2. `hibacaml/data/mnist.py` for Split-MNIST and Full-MNIST construction. Both
   protocols go through the same `_make_loader`/`_make_task` pair and differ
   only in which splits they supply.
3. `hibacaml/graph/builder.py` for how columns, gates, composer, and hierarchy
   heads become a FabricPC graph.
4. `hibacaml/control/support.py`, `hibacaml/control/scoring.py`, and
   `hibacaml/control/search.py` for support
   control and teacher-first search.
5. `hibacaml/control/certificates.py` for internal certificates, then
   `hibacaml/control/shells.py` for the shell edits they describe.
6. `hibacaml/training/shared.py` for clamp assembly, objective primitives,
   and evaluation scoring/accumulation — everything both learners compute
   without touching trainer state.
7. `hibacaml/training/trainer.py` for the shared base and the update
   template, then `hibacaml/training/pc.py` and
   `hibacaml/training/backprop.py` for the two learner contracts.
8. `hibacaml/experiment.py` for the graph/trainer/run-root wiring both runners
   share.
9. `experiments/split_mnist.py` for the executable experiment orchestration.
   Use `experiments/mnist.py` for the static ten-class full-bank study.
   Each runner now contains only its own protocol: task construction,
   protocol assertions, metric aggregation, and reporting.

The shortest conceptual summary is: the paper propose two coupled probabilistic
levels for continual learning, and the code realizes them as a sparse gated
column graph plus an audited support/shell controller around that graph.
