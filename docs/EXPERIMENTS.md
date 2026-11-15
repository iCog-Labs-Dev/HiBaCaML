# HiBaCaML: Experiments

This document describes the four experiment runners in `experiments/`: the data each
uses, its configuration, how to run it, what it writes, and a reference result. The
method is explained in [ALGORITHM.md](ALGORITHM.md), the modules in
[CODE_STRUCTURE.md](CODE_STRUCTURE.md), and the Split-CIFAR-10 controllers in
[COLUMN_REUSE_AND_SHELL_ADAPTATION.md](COLUMN_REUSE_AND_SHELL_ADAPTATION.md).

| Runner | Protocol | Columns active | Controller | Command (from the repository root) |
|---|---|---|---|---|
| `experiments/split_mnist.py` | Split-MNIST: 5 binary tasks in sequence | 2 shared + 3 per task | Exact support search and shell edits | `python -m experiments.split_mnist` |
| `experiments/mnist.py` | Full-MNIST: one ten-class task | 2 shared + 15 (all adaptive) | None (fixed support) | `python -m experiments.mnist` |
| `experiments/cifar.py` | Full-CIFAR-10: one ten-class task | 4 shared + 5 (all adaptive) | None (fixed support) | `python -m experiments.cifar --seed 0` |
| `experiments/split_cifar.py` | Split-CIFAR-10: 5 binary tasks in sequence | 3 shared + 4 per task | Restricted column reuse and audited shell adaptation | `python -m experiments.split_cifar --seed 0` |

Install the dependencies with `pip install -r requirements.txt`. The MNIST runners
are configured by editing the constants at the top of their file (`LEARNING`,
`TASKS_LIMIT`, `BANK`, `OVERRIDES`, ...). The CIFAR runners take only `--seed` on the
command line, require a GPU with JIT enabled, and read their remaining choices from
the settings block in the file. All outputs go under `runs/`.

## 1. Data and Tasks

### 1.1 Task-incremental splits

Both split protocols use the same five class pairs, `TASK_CLASS_PAIRS` in
`hibacaml/data/common.py`:

```text
(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
```

Each task's official training examples are split by a seed-controlled stratified
draw (`validation_fraction = 0.10`) into a *fit* split used for training and a
*validation* split on which every controller decision is judged. The official test split is
read only for reporting. `Task.loader(split)` returns a named split and raises when
it is missing rather than falling back.

For each task:

- labels are filtered to the task's two classes;
- task-local targets map the first class to index 0 and the second to index 1
  when `task_local_heads` is true;
- hierarchy targets are computed per batch;
- `TaskLoader` provides deterministic iteration with optional shuffling,
  augmentation and batch limits;
- a one-hot `task_query` vector is attached to the task.

The code implements task-local binary classification by target remapping and task
queries rather than by creating five separate physical classifier modules:
`HiBaCaMLConfig.output_dim` is `2` when `task_local_heads = True`. This keeps the
papers' intent, separating shared-representation forgetting from ten-way class-head
interference.

### 1.2 MNIST

`hibacaml/data/mnist.py` sets `KERAS_BACKEND=jax`, loads MNIST through
`keras.datasets.mnist.load_data()`, and exposes `build_split_mnist_tasks` and
`build_full_mnist_task`. Images stay uint8 until batching; each batch is normalized
by the MNIST mean/std and reshaped to `28 x 28 x 1`. Split-MNIST uses split seed
`cfg.seed + task_id`. MNIST loaders reuse one fixed shuffle order on every pass.

`build_full_mnist_task` creates one ten-class task with `task_local_heads=False`.
A seed-controlled stratified split partitions the official 60,000 training examples
into 54,000 fit and 6,000 validation examples; the official 10,000-example test set
remains separate.

### 1.3 CIFAR-10

`hibacaml/data/cifar.py` loads CIFAR-10 through Keras and normalizes each channel.
The fit split is augmented by a random crop of the zero-padded (4 pixels) image and
a random horizontal flip, reshuffled on every pass, and its last partial batch is
dropped. Held-out loaders are never augmented; the CIFAR runners evaluate them in batches
of 500.

`build_split_cifar_tasks` gives each task independent seed streams from
`task_seeds(seed, task_id)`: one for its fit/validation split, one for training
order and augmentation, one for support trials and one for the bootstrap. Each task
has 9,000 fit, 1,000 validation and 2,000 test images. `build_full_cifar_task`
builds one ten-class task with 45,000 fit, 5,000 validation and 10,000 test images.

### 1.4 Tokens

`PatchTokenizerNode` turns an image or feature map into tokens without parameters:

```text
input: (H, W, C)
  -> reshape into a grid of patch_size patches
  -> flatten each patch
  -> append normalized (y, x) coordinates
  -> output: (num_patches, ph * pw * C + 2)
```

There is no learned projection: a shared embedding would be trained by every task
outside the support mask, so each column's `b_micro` learns its own projection
instead.

- **MNIST.** The config default `patch_size = (7, 7)` gives 16 tokens of 51
  dimensions (used by Full-MNIST). Split-MNIST overrides it to `(28, 28)`: one
  flattened raw-pixel token of 786 dimensions.
- **CIFAR-10.** A shared convolutional stem (five 3x3 convolutions with channels
  32, 64, 64, 128, 128 and strides 1, 2, 1, 2, 1, then a 2x2 average pool) turns
  the 32x32x3 image into a 4x4x128 map. Each cell is one token: 16 tokens of 130
  dimensions.

## 2. Training and Evaluation Procedure

All runners construct their trainer through `hibacaml.experiment.build_trainer`:

- `HiBaCaMLPCInference` is configured for graph inference;
- `FeedforwardStateInit` is used explicitly;
- `create_hibacaml_structure` builds the graph;
- FabricPC `initialize_params` initializes graph parameters with `cfg.seed`;
- the trainer class is selected from `learning`.

For predictive coding, feedforward initialization only supplies a coherent starting
state. Supervised targets are then clamped, FabricPC performs the configured
iterative settling, and `compute_local_weight_gradients` computes local graph-energy
gradients. Backprop uses the same initializer but skips settling and differentiates
its external objective end to end. The graph builder requires an explicit
initializer and the trainer rejects non-feedforward initialization.

For each task, `trainer.train_task(task)`:

1. selects a support at the task boundary if no support is set: Split-MNIST
   searches for one, Full-MNIST and Full-CIFAR-10 take the default support, and
   Split-CIFAR-10 sets the support beforehand;
2. runs the compiled learner update on each training batch;
3. masks inactive-column gradients inside that update using the dynamic support,
   then applies precision resistance (if enabled) and an installed write policy's
   gradient mask (Split-CIFAR-10);
4. applies the optimizer transition, optionally behind a global-norm gradient
   clip, then masks the updates too, because AdamW momentum and weight decay would
   otherwise still move inactive columns; an installed write policy masks and
   scales the updates, and after the parameters are applied it restores every
   protected value;
5. applies shell structural edits to active columns (Split-MNIST only);
6. performs demotion audits periodically (Split-MNIST only);
7. refreshes certificates every `cert_refresh_interval` updates (also after every
   structural edit, and on every batch with the predictive-coding learner), then
   performs local one-swap audits periodically (Split-MNIST only);
8. at the end of each epoch, calls the runner's `on_epoch_end` hook (Split-CIFAR-10
   runs shell adaptation there) and then evaluates the validation split with the
   RNG restored afterwards;
9. freezes the task support as a `SupportSnapshot`;
10. evaluates the task and every earlier task under its saved support.

`train_task` owns task-level setup and completion, `_train_epoch` owns epoch
aggregation and held-out evaluation, and `_train_batch` owns the per-update hook
order: gradient production, structural edits, step advancement, demotion audit,
composer reporting, certificate refresh, local maintenance.

### 2.1 Predictive-coding clamping contract

The predictive-coding learner implements standard discriminative predictive coding.
Clamped nodes are held fixed for the whole settling process; every non-clamped node
is updated by `InferenceBase.update_latents` through `InferenceSGD`, which applies
`z_latent -= eta_infer * latent_grad` (latent decay is zero) once per iteration for `infer_steps`
iterations under `jax.lax.fori_loop`.

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
inputs stay clamped, so `y` settles freely and the prediction is read from its
`z_mu`. The class target is applied afterward as an external cross-entropy and never
enters the hidden-state trajectory.

The clamp policy is expressed by the `include_targets` flag on
`HiBaCaMLTrainer._build_clamps`. It defaults to `False`, so target-free is the
default behavior and supervised PC training opts in explicitly. The base trainer
rejects target clamps for a learner whose `CLAMPS_MAY_INCLUDE_TARGETS` contract is
false, so backprop targets remain external in every path.

### 2.2 Predictive-coding auxiliary factors

The parent-child and composer auxiliary terms enter PC learning as explicit factors
rather than as node energies, because neither fits FabricPC's node-local
`energy(z_latent, z_mu)` contract: parent-child couples two hierarchy predictions,
and the composer penalty depends on gate probabilities derived from every active
column's features, certificates, and the task query.

Each settling step therefore runs the ordinary FabricPC sweep, then pushes each
factor's `dE/dz_mu` back through the producing node's own `forward` with `jax.vjp`,
accumulating into the source nodes' `latent_grad` before latents are updated.
Parent-child reaches `active_feature_summary` through both hierarchy heads; the
composer reaches each column's feature branch through its `feature_gate`.

Two numerical boundaries are deliberate. Factor gradients differentiate the **batch
sum**, matching FabricPC, which differentiates `jnp.sum(energy)` in both
`forward_and_latent_grads` and `forward_and_weight_grads`; the reported components
remain batch means. And the factors inherit FabricPC's one-step offset: settling
factors act on fresh in-step `z_mu`, the weight phase recomputes predictions from
the settled `z_latent`, and reporting keeps reading the stored `z_mu`, exactly as
the three cross-entropies do.

Both factors read `z_mu`, never `z_latent`. During supervised training the hierarchy
and output nodes are clamped, so their `z_latent` holds the labels; a factor reading
those would measure agreement between targets rather than between predictions.
Certificates, the task query, and the support mask are clamped, so cotangents
reaching them are inert by construction.

### 2.3 Evaluation

Every evaluation uses target-free inference with external supervision: the graph is
clamped only to the image, support mask, task query, and certificate inputs. The
class target and the two class-derived hierarchy targets are kept outside inference
and used only afterward to compute external cross-entropies. This separation is
essential for the predictive-coding learner, where clamping test targets during
settling would leak the answer into the hidden-state trajectory.

Reporting evaluations read the test split and never refresh certificates, so
observing a run cannot change it. Controller decisions are judged on the
validation split only. Reports include evaluated and correct example counts, a task-local confusion
matrix, per-class precision/recall, cross-entropy, the composite loss and per-column
composer use.

`HiBaCaMLTrainer` owns the evaluation request and its stateful effects. It
normalizes ordered supports, constructs single- or multi-support clamps, splits the
evaluation RNG once for each executed batch, and refreshes certificates on the first
loader batch only when requested. `pc.py` and `backprop.py` provide pure inference and
detailed-scoring functions, which `build_evaluation_programs` compiles into raw,
detailed and score-only programs; `shared.py` scores
an already-produced graph state and accumulates example-weighted metrics.

The split runners derive, from the task-by-task accuracy matrix:

- mean seen accuracy after each task;
- mean forgetting after each task;
- the support sequence;
- the phi and support-entropy trajectories;
- best one-swap gains (Split-MNIST).

Split-CIFAR-10 also reports acquisition (each task's accuracy when it finishes),
backward transfer, and mean forgetting over old tasks only, for both the test and
validation matrices (`continual_extras`).

## 3. Split-MNIST

The paper's Split-MNIST setup: five binary tasks trained in sequence with
task-local two-way readouts, a 20-column pool with 2 shared and 3 selected columns
active per task, and the teacher-first support controller of
[ALGORITHM.md](ALGORITHM.md) (section 2.5).

**Configuration.** The `default` preset (`HiBaCaMLConfig` and its nested configs)
plus the runner's `OVERRIDES`:

- seed: `0`
- input shape: `(28, 28, 1)`
- patch size: `(28, 28)` (runner override; one flattened token)
- number of tasks: `5`
- batch size: `1024` (runner override; config default `256`)
- validation fraction: `0.10` of each task's training examples
- epochs per task: `5`
- inference steps: `16`; inference rate: `0.05`
- optimizer: AdamW with learning rate `0.001` and weight decay `0.05`
- columns: `20 = 2 shared + 15 adaptive + 3 reserve`; active nonshared: `3`
- shell dimensions: kernel `8`, tiers `(4, 6, 8)`, kernel depth `2`
- composer hidden dimension: `64`; composer top-k: `3`
- exact boundary shortlist: `8`
- static/neighbor support scoring batch sizes: `32` (runner override)
- audit data batch caps: `128` for each of `boundary_current`, `rollout_train`,
  `boundary_worst_old`, `boundary_mixed_old`, `local_swap_audit` and
  `demotion_audit` (`*_data_batch_size`, runner override); these cap each batch an
  audit bundle draws
- maintenance interval: `64`; demotion audit interval: `64`

**Running.** Edit `LEARNING` (`"backprop"` or `"pc"`), `TASKS_LIMIT` and `OVERRIDES`
at the top of `experiments/split_mnist.py`, then run
`python -m experiments.split_mnist`.

**Outputs.** `runs/SMNIST_<learning>_<UTC>/` contains `run_summary.json`,
`task_summary.csv`, `events.jsonl`, `heartbeat.json`, `task_summaries.json`, a
`task_<k>/` artifact bundle and checkpoint per task, a `final/` bundle (evaluations,
support tables, controller tables, certificates and composer diagnostics as JSON and
CSV), the run's selector bank under `selector_state/`, and `plots/`.

## 4. Full-MNIST

A static capacity control: one ten-class task on a fixed bank of columns, with no
support search, structural editing or continual-learning pressure. It establishes
what each learner can represent before any forgetting is attributable.

**Configuration.** `experiments/mnist.py` declares the protocol over the same
20-column graph:

- seed: `0`; batch size: `256`; epochs: `5`;
- one ten-class task with `task_local_heads=False` and `num_tasks=1`;
- patch size: `(7, 7)` (16 tokens of 51 dimensions);
- PC settling: `infer_steps=16`, `eta_infer=0.05`;
- active columns: 2 shared plus all 15 adaptive columns (`BANK = 17`);
- inactive reserve columns: `(17, 18, 19)`;
- `topk_nonshared=15`, giving `active_support_size=17`;
- column-logit and active-feature aggregation scale: `1/17`;
- composer top-k: exactly `3`;
- exact search, structural edits, precision resistance, demotion audits, and
  selector-state writes: disabled.

The bank size is the runner's one architectural knob. `BANK` derives
`adaptive_count = BANK - 2`, `reserve_count = 20 - BANK`, and
`topk_nonshared = adaptive_count`, so `BANK=17` leaves columns `(17, 18, 19)` in
reserve while `BANK=20` puts every column in the bank at aggregation scale `1/20`.
The graphs are identical (the same 210 nodes and parameter shapes) and differ only
in the aggregation scale and which columns are gated on. The runner enforces
`topk_nonshared == adaptive_count`: every non-shared column in the pool must be
active.

`pooled_logits` and `active_feature_summary` keep their 20 gated input edges, but
their aggregation scale is the true bank mean `1/17` rather than the Split-MNIST
`1/5`. The default support path inside `train_task` selects adaptive columns 2
through 16; the runner asserts that support, derived from `topk_nonshared` and
`active_support_size`.

At batch size 256 and five epochs, the 54,000-example fit split produces 211
updates per epoch and 1,055 in total. The runner records expected and completed
counts and fails rather than accept a shortened run. Epoch loss means are weighted
by actual batch size, including partial batches.

**Running.** Set `LEARNING` and `BANK` at the top of `experiments/mnist.py`, then run
`python -m experiments.mnist`.

**Outputs.** `runs/full_mnist_<learner>_bank<BANK>/seed_<n>/` contains
`run_summary.json`, `task_summary.csv`, `events.jsonl`, `heartbeat.json`,
`task_summaries.json`, the `task_0/` and `final/` bundles, a checkpoint, and
`plots/` (learning curve, test confusion matrix, composer usage). The PC and
backprop runs use different directory names, as do different bank sizes. The
directory name has no timestamp, so remove an earlier run's directory before
repeating the same learner, bank and seed.

## 5. Full-CIFAR-10

A static capacity control for CIFAR-10: one ten-class task on a fixed layout of
4 shared + 5 adaptive columns, all active, with the other 31 columns held inactive
in reserve. No controller runs and nothing is protected: the stem, composer and all
active columns train throughout.

**Configuration.** The `cifar` preset plus the runner's `OVERRIDES`:

- input `32 x 32 x 3`; convolutional stem as in section 1.4 (16 tokens of 130
  dimensions);
- columns: `40 = 4 shared + 5 adaptive + 31 reserve`; all 9 active;
- shell dimensions: kernel `32`, tiers `(10, 20, 30)`, kernel depth `3`;
- shared-column K inputs: `dense`;
- composer top-k: `3`;
- optimizer: AdamW with learning rate `3e-4` and weight decay `0.05`, behind a
  global-norm gradient clip of `1.0`;
- batch size `256`, evaluation batch `500`, `60` epochs;
- fit/validation split `45,000 / 5,000`, augmentation on;
- exact search, structural edits, precision resistance and demotion audits: off.

One epoch is 175 updates, 10,500 in total. At the end the runner checks the update,
optimizer-step and parameter-revision counts, the support, that every reserve
column is byte-identical to its initialization, and that the composer selected at
most three columns on the test split.

**Running.** `python -m experiments.cifar --seed 0` (`--seed` also sets the
fit/validation split).

**Outputs.** `runs/CIFAR10_backprop_static_seed<S>_<UTC>/` contains `run_summary.json`
(architecture and parameter count, runtime identity, schedule and counts, per-epoch
validation, final validation and test metrics), `task_summary.csv`, `events.jsonl`,
`heartbeat.json`, `task_summaries.json` and `plots/` (learning curve, test confusion
matrix, composer usage).

## 6. Split-CIFAR-10

The continual-learning experiment for CIFAR-10: the five class pairs trained in
sequence with task-local two-way heads. From the second task on, each task's support is
chosen by restricted column reuse at its boundary, and shell adaptation runs
during its training; after
the first task, exact write protection keeps the stem, the composer maps and the
shared columns' protected units unchanged. The mechanisms are specified in
[COLUMN_REUSE_AND_SHELL_ADAPTATION.md](COLUMN_REUSE_AND_SHELL_ADAPTATION.md).

**Configuration.** The `cifar` preset plus the runner's `OVERRIDES`:

- input `32 x 32 x 3`; convolutional stem as in section 1.4;
- columns: `40 = 3 shared + 31 adaptive + 6 reserve`; 4 nonshared active per task;
- shell dimensions: kernel `32`, tiers `(10, 20, 30)`, kernel depth `3`;
- shared-column K inputs: `kernel-only` (shared kernel units read only kernel
  coordinates);
- composer top-k: `3`;
- optimizer: AdamW with learning rate `3e-4` and weight decay `0.05`, behind a
  global-norm gradient clip of `1.0`, with per-epoch clipping statistics recorded;
- batch size `256`, evaluation batch `500`, `30` epochs per task;
- per task: 9,000 fit / 1,000 validation / 2,000 test images, augmentation on;
- exact search, structural edits, precision resistance and demotion audits: off.

One epoch is 35 updates, so each task trains for 1,050 updates and the sequence for
5,250. Restricted column reuse adds at most 1,680 discarded trial updates on
isolated model copies (35 per trial), so a run performs at most 6,930 optimizer
updates. Shell adaptation adds none.

**Settings.** The settings block at the top of `experiments/split_cifar.py` selects
the condition:

| `SUPPORT_POLICY` | `STRUCTURE` | `PROMOTIONS` | Condition label | Meaning |
|---|---|---|---|---|
| `restricted_reuse` | `on` | `on` | `full` | The full method (default) |
| `restricted_reuse` | `on` | `off` | `no_promotions` | Shell adaptation without relabel promotions |
| `restricted_reuse` | `off` | any | `no_shell_adaptation` | Column reuse only; shell adaptation observes but installs nothing |
| `fixed_fresh` | `off` | any | `fresh_only` | Lowest-index fresh columns for every task; no trials, no shell adaptation |
| `replay` | any | any | `replay` | Install the supports recorded in `REPLAY_SOURCE`, a previous `run_summary.json` |

Other valid combinations run and are labelled `custom`. `STRUCTURE = "off"`
overrides `PROMOTIONS`; the summary records both the requested and the effective
setting. A replay source is validated against the allocation rules before any
training.

**Running.** `python -m experiments.split_cifar --seed 0`. The seed sets model
initialization and every task's data, trial and bootstrap streams.

**Checks.** After every task the runner requires: exact update, optimizer-step,
parameter-revision (updates plus installed parameter edits) and role-revision
(installed label exchanges) counts; saved supports equal to the committed
allocation; no column outside the task's support changed; every protected
parameter byte-identical to its reference; finite model state and results; and at
most three columns selected by the composer. Any violation stops the run with an
`IntegrityError` naming what changed. At the end it checks the number of
shell-adaptation rounds and the trial-update bound.

**Outputs.** `runs/SCIFAR10_backprop_seed<S>_<UTC>/`, rewritten after every task:

- `run_summary.json`: `status` (`running`, then `complete`), the condition and
  settings, runtime identity, architecture and parameter count, schedule, the test
  and validation accuracy matrices, mean seen accuracy, forgetting, continual
  metrics, per-task boundary reports and counts, `support_selection` (every
  boundary's candidates, trial results, bootstrap margin, decision and update
  accounting) and `structural_control` (every shell-adaptation round, proposal,
  installed or shadow label exchange, harm ledger and activity counts);
- `task_summary.csv`: one row per task;
- `clipping_epoch_summaries.csv`: gradient-norm and clipping statistics per epoch;
- `events.jsonl`, `heartbeat.json`, `task_summaries.json`;
- `plots/`: accuracy/forgetting curves and the chosen-support table.

## 7. Reference Results

Single runs at seed 0 on one GPU (JAX 0.10.1), with the shipped settings. Expect
differences of the order of a percentage point between seeds and GPU runtimes.

| Runner | Result |
|---|---|
| Split-MNIST (backprop) | final mean accuracy 98.37%; mean forgetting 0.67 percentage points |
| Full-MNIST (backprop, bank 17) | test accuracy 97.40% (cross-entropy 0.082); validation 97.37% |
| Full-CIFAR-10 | test accuracy 82.16% (cross-entropy 0.511); validation 81.98% |
| Split-CIFAR-10 (full method) | final mean test accuracy 89.99%; validation 89.90%; mean old-task forgetting 0.09 percentage points |

In that Split-CIFAR-10 run, restricted column reuse kept fresh columns at every
boundary (1,680 trial updates, 6,930 optimizer updates in total), and shell
adaptation installed 252 label exchanges and six parameter edits (three shared-column
and three fresh-column inhibitions); pruning never triggered.
