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
- `hibacaml/training/trainer.py:HiBaCaMLTrainer`
- `hibacaml/control/search.py:ExactSearchService`
- `hibacaml/control/shells.py:ShellController`

**Implementation details.**

- `N`: each column is a FabricPC subgraph created inside
  `create_hibacaml_structure`. Column metadata records nodes such as
  `b_micro`, `k_micro`, `l_micro`, `feature_pool`, gates, logits, and
  certificate inputs.
- `C`: the combiner is split into a simple stage-1 averaged logit path and a
  stage-2 attention-style composer. `ScaledAddNode` combines `stage1_logits`
  with the `ComposerStage2Node` correction and applies softmax.
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

- `hibacaml/nodes/core.py`
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
- `hibacaml/control/shells.py:ShellController.refresh_certificates`
- `hibacaml/control/shells.py:ShellController.certificate_matrix`
- `hibacaml/control/search.py:_certificate_reuse_score`

**Implementation details.**

`ShellController.refresh_certificates` updates EMAs and computes a
`ColumnCertificate` per column. Certificate fields include:

- `q_mean`, derived from sigmoid shell precisions;
- `prec_mean`, derived from exponentiated shell precision parameters;
- `pred_mean`, from mean predicted activation magnitude;
- `live_frac`, `tier_q`, and `tier_occ`;
- `shared_abstraction_mass`;
- `specificity_load`;
- `demotion_pressure`;
- `saturation`;
- `similarity_signature`, computed as cosine similarity between certificate
  vectors.

The certificate vector used by the composer is constructed by
`certificate_matrix`, masked by the active support. The static support scoring
path also uses `_certificate_reuse_score`, which rewards `q_mean` and
`shared_abstraction_mass` while penalizing specificity, demotion pressure, and
saturation. In `paper_faithful` mode, `certificate_support_weight` is set to
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

- `hibacaml/nodes/core.py:ShellBankInputNode`
- `hibacaml/nodes/core.py:ShellBankRecurrentNode`
- `hibacaml/nodes/core.py:ShellBankResidualNode`
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

- `hibacaml/data/split_mnist.py:_hierarchy_targets`
- `hibacaml/graph/builder.py:create_hibacaml_structure`
- `hibacaml/nodes/core.py:ComposerStage2Node`
- `hibacaml/nodes/core.py:ScaledAddNode`
- `hibacaml/training/trainer.py:_hierarchy_parent_child_penalty`

**Implementation details.**

`_hierarchy_targets` builds quadrant-aware soft labels by measuring normalized
image mass in each quadrant. `hier_mid` receives shape
`(mid_targets, output_dim)`, while `hier_global` receives the global task target.

The graph adds:

- `active_feature_summary` as the mean active column feature summary;
- `hier_mid`, a weighted softmax head over midpoint/quadrant targets;
- `hier_global`, a weighted softmax head over the global target;
- a parent-child penalty that encourages the mean midpoint prediction to match
  the global prediction.

`ComposerStage2Node` stacks per-column gated features and certificate vectors,
builds a certificate-derived prior, applies query-conditioned residual attention
over active columns, and produces a correction vector. `ScaledAddNode` combines
stage-1 logits with this correction.

**Alignment status.** Faithful in purpose. The implementation's composer works
over per-column feature vectors and certificates rather than an explicitly
per-token representation at the final combiner. Patch tokens are processed
inside the column graph before feature pooling.

## 3. System Architecture

### 3.1 Main Modules

`hibacaml/config/defaults.py`
: Defines all public hyperparameters and mode presets. `make_hibacaml_config`
  supports `paper_faithful` and `smoke`.

`hibacaml/data/split_mnist.py`
: Builds deterministic task loaders for task-incremental Split-MNIST,
  including task-local targets, task queries, and hierarchy targets.

`hibacaml/graph/builder.py`
: Builds the static FabricPC graph and initializes persistent HiBaCaML state.

`hibacaml/nodes/core.py`
: Defines custom FabricPC node types for patch tokenization, gated columns,
  shell-bank processing, attention-style composition, and final output
  combination.

`hibacaml/control/support.py`
: Encodes support-set arithmetic: shared-column inclusion, support masks,
  adaptive support enumeration, reserve recruitment candidates, and one-swap
  neighbors.

`hibacaml/control/shells.py`
: Owns shell statistics, certificates, semantic penalties, structural edits,
  demotion-swap candidates, and precision-weighted gradients.

`hibacaml/control/search.py`
: Owns exact boundary search, rollout scoring, posterior ranking, reserve
  diagnostics, replay-bank reselection, local one-swap maintenance, and
  demotion-swap auditing.

`hibacaml/training/trainer.py`
: Implements predictive-coding inference/gradient training with support
  selection, gradient masking, structural edits, evaluation, artifact export,
  checkpoints, and snapshots.

`hibacaml/training/backprop.py`
: Implements an end-to-end autodiff runner that preserves the same support,
  certificate, shell-edit, and audit semantics.

`experiments/split_mnist.py`
: Wires configuration overrides, task construction, graph review, training,
  evaluation aggregation, summaries, and plots.

### 3.2 Data and Control Flow

The end-to-end flow is:

```text
run_experiment_notebook(...)
  -> make_hibacaml_config(mode)
  -> apply_notebook_overrides(...)
  -> build_split_mnist_tasks(cfg)
  -> create_hibacaml_structure(cfg, inference)
  -> initialize_params(...)
  -> HiBaCaMLBackpropRunner or HiBaCaMLTrainer
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
       export_task_artifacts()
       save_checkpoint()
  -> aggregate accuracy/forgetting/support trajectories
  -> write run summary/table and plots
```

The support mask is the central runtime control variable. It is clamped into the
graph as `support_mask`, and each `ElementwiseGateNode` reads its column's
scalar mask entry to zero or retain token, feature, and logit pathways.

### 3.3 Training Modes

The experiment script accepts `learning = "pc"` or `"backprop"`.

- `HiBaCaMLTrainer` uses FabricPC inference and local predictive-coding weight
  gradients through `compute_local_weight_gradients`.
- `HiBaCaMLBackpropRunner` uses feedforward state initialization and JAX
  autodiff for supervised losses, while preserving support masks, certificate
  refresh, gradient masking, shell edits, boundary search, local swaps, and
  evaluation semantics.

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

`build_split_mnist_tasks` implements the five tasks from `_TASK_CLASS_PAIRS`.
It loads MNIST via TensorFlow/Keras when available, otherwise a cached
`mnist.npz` fallback. Images are normalized by MNIST mean/std and reshaped to
`28 x 28 x 1`.

For each task:

- labels are filtered to the task's two classes;
- task-local targets map the first class to index 0 and the second to index 1
  when `task_local_heads` is true;
- hierarchy targets are computed;
- `_ArrayTaskLoader` provides deterministic iteration with optional shuffling
  and batch limits;
- a one-hot `task_query` vector is attached to the task.

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

`experiments/split_mnist.py` constructs the trainer in `build_trainer`:

- `InferenceSGD` is always configured for graph inference;
- `FeedforwardStateInit` is used for the backprop runner;
- `create_hibacaml_structure` builds the graph;
- FabricPC `initialize_params` initializes graph parameters;
- trainer class is selected from `learning`.

For each task, `run_experiment` calls `trainer.train_task(task)`. Training:

1. selects a support at the task boundary if no support is set;
2. computes gradients on each training batch;
3. masks gradients for inactive columns;
4. applies optimizer updates with precision resistance;
5. applies shell structural edits to active columns;
6. periodically refreshes certificates;
7. periodically performs demotion and local one-swap audits;
8. freezes the task support as a `SupportSnapshot`;
9. evaluates the task using its saved support.

### 4.6 Evaluation Setup

After each task, `run_experiment` calls `evaluate_all_saved_supports`, producing
an accuracy matrix over all completed tasks under their frozen saved supports.
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

## 5. Important Hyperparameters

Default `paper_faithful` values are defined by `HiBaCaMLConfig` and nested
configs:

- seed: `0`
- input shape: `(28, 28, 1)`
- patch size: `(7, 7)`
- number of tasks: `5`
- batch size in config: `256`
- experiment script override: `BATCH_SIZE = 768`
- epochs per task: `5`
- inference steps: `8`
- optimizer: AdamW with learning rate `0.001` and weight decay `0.05`
- columns: `20 = 2 shared + 15 adaptive + 3 reserve`
- active nonshared columns: `3`
- shell dimensions: kernel `8`, tiers `(4, 6, 8)`
- composer hidden dimension: `64`
- composer top-k: `3`
- exact boundary shortlist: config default `8`
- experiment script static/neighbor support batch sizes: `16`
- maintenance interval: `64`
- demotion audit interval: `64`

The script-level constants at the top of `experiments/split_mnist.py` override
some config defaults for notebook/script runs. Documentation readers should
distinguish config defaults from script overrides.

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
cfg = make_hibacaml_config("paper_faithful")
tasks = build_split_mnist_tasks(cfg)
structure = create_hibacaml_structure(cfg, inference)
params = initialize_params(structure, seed)
trainer = HiBaCaMLBackpropRunner or HiBaCaMLTrainer(cfg, structure, params, tasks)

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
        grads, losses, state = trainer.compute_gradients(batch, support)
        grads = zero inactive-column gradients
        params = optimizer_update(params, precision_weighted_grads)
        params = shell_controller.apply_structural_edits(params, state, support, phi)
        refresh certificates periodically
        audit demotion swaps periodically
        audit local support one-swaps periodically

    freeze support snapshot
    evaluate all frozen supports
    export summaries/checkpoint
```

## 8. Reader Guide

To understand the code as a concrete realization of the paper, read in this
order:

1. `hibacaml/config/defaults.py` for the paper-scale architectural constants.
2. `hibacaml/data/split_mnist.py` for the task and target construction.
3. `hibacaml/graph/builder.py` for how columns, gates, composer, and hierarchy
   heads become a FabricPC graph.
4. `hibacaml/control/support.py` and `hibacaml/control/search.py` for support
   control and teacher-first search.
5. `hibacaml/control/shells.py` for internal certificates and shell edits.
6. `hibacaml/training/trainer.py` plus `hibacaml/training/backprop.py` for the
   sequential training loop.
7. `experiments/split_mnist.py` for the executable experiment orchestration.

The shortest conceptual summary is: the paper propose two coupled probabilistic
levels for continual learning, and the code realizes them as a sparse gated
column graph plus an audited support/shell controller around that graph.
