# HiBaCaML: Code Structure

This document lists every module, what it owns, and how a run flows through them.
The method itself is explained in [ALGORITHM.md](ALGORITHM.md); the runners, their
settings and how to launch them are in [EXPERIMENTS.md](EXPERIMENTS.md); the
Split-CIFAR-10 controllers are specified in
[COLUMN_REUSE_AND_SHELL_ADAPTATION.md](COLUMN_REUSE_AND_SHELL_ADAPTATION.md).

## 1. Modules

### 1.1 Configuration, data and types

`hibacaml/__init__.py`
: The public package surface the runners import from: configuration, task
  builders, `build_trainer`, `prepare_run_root`, the graph builders, the learners
  and `export_run_artifacts`.

`hibacaml/config/defaults.py`
: Defines the configuration hyperparameters and mode presets (the Split-CIFAR-10
  controller constants are module constants in `control/reuse.py` and
  `control/structure.py`). `make_hibacaml_config`
  supports `default` (MNIST) and `cifar`; `override(cfg, section__field=value)`
  returns a modified copy. `StemConfig` describes the CIFAR convolutional stem.

`hibacaml/data/common.py`
: The dataset-independent task path: `TaskLoader` (deterministic batching of raw
  uint8 images, normalized one batch at a time), `make_task`, the stratified
  fit/validation split, task-local target remapping, hierarchy targets, task
  queries and the five class pairs `TASK_CLASS_PAIRS`.

`hibacaml/data/mnist.py`
: Loads and normalizes MNIST and builds the Split-MNIST tasks
  (`build_split_mnist_tasks`) and the single ten-class Full-MNIST task
  (`build_full_mnist_task`). MNIST reuses one fixed shuffle order on every pass.

`hibacaml/data/cifar.py`
: Loads CIFAR-10 and builds the Split-CIFAR-10 tasks (`build_split_cifar_tasks`)
  and the Full-CIFAR-10 task (`build_full_cifar_task`): per-channel normalization,
  random crop (pad 4) and horizontal flip on the fit split, a fresh shuffle each
  pass, and, for Split-CIFAR-10, independent per-task seed streams (`task_seeds`:
  split, training, trial and bootstrap).

`hibacaml/types.py`
: Shared data types: `Task`, `SupportSnapshot`, `TaskSummary`, `ColumnCertificate`,
  `ShellStats`, the per-run `PersistentHiBaCaMLState` (supports, certificates,
  revisions, role labels and controller tables) and `clone_persistent_state`.

### 1.2 Graph and nodes


`hibacaml/graph/builder.py`
: Builds the static FabricPC graph and initializes persistent HiBaCaML state:
  the optional convolutional stem, patch tokens, each column's B/K/L shell nodes,
  feature pool, gates and logits, the composer and the hierarchy heads. The
  `shared_k_connectivity` option is `dense` (default) or `kernel-only`.
  `column_logits` uses `LinearExplicitGrad` because it has a single input edge
  and an identity activation under unit-precision Gaussian energy, so its
  explicit derivative reproduces autodiff exactly. It passes
  `weight_init=KaimingInitializer()` because that subclass defaults to Normal
  where `Linear` defaults to Kaiming, and the subclass default would draw
  different initial weights. `feature_pool` keeps `Linear`: it flattens its
  2 + `kernel_depth` inputs (B, L and each kernel node) through a tanh, where the explicit-gradient form would
  not be equivalent.

`hibacaml/nodes/pathways.py`
: Defines patch-token preparation (`PatchTokenizerNode`) and support-gated graph
  pathways (`ElementwiseGateNode`).

`hibacaml/nodes/micro_columns.py`
: Defines the input, recurrent, and residual shell-bank micro-column nodes.

`hibacaml/nodes/composer.py`
: Defines the composer (column-mixing gate) mathematics, its exact top-k
  selection, and final output integration. The numerical detail helpers stay
  beside `ColumnComposerNode` so training, evaluation, and graph execution share
  one implementation.

`hibacaml/nodes/energy.py` and `hibacaml/nodes/common.py`
: The weighted cross-entropy energy used by the output and hierarchy heads, and
  small node helpers.

### 1.3 Control

The MNIST support controller is `support.py`, `ranking.py`, `scoring.py`,
`replay_bank.py`, `search.py` and `shells.py`; it runs when exact search and
structural edits are enabled (Split-MNIST). The Split-CIFAR-10 controllers are
`roles.py`, `reuse.py` and `structure.py`; they never import the MNIST search
stack directly. `certificates.py` serves both.

`hibacaml/control/support.py`
: Encodes support-set arithmetic: shared-column inclusion, support masks,
  adaptive support enumeration, reserve recruitment candidates, and one-swap
  neighbors.

`hibacaml/control/certificates.py`
: `CertificateController` — certificate observation of the graph's shells: shell-EMA
  statistics, per-column `ColumnCertificate` construction and similarity
  signatures, the revision-scoped certificate stack the composer consumes, and
  the semantic penalty. Observations are reduced by an un-jitted projection, copied to
  the host in one grouped transfer, and the ordered recurrence runs on the host. It also owns
  the shell geometry helpers
  (`shell_slices`, `shell_node_names`, `effective_precision`) used by the control
  layer. When role labels are installed, tier statistics
  are aggregated over label masks, cached by label bytes.

`hibacaml/control/shells.py`
: `ShellController` — the MNIST controller's shell edits: inhibition, outside-in
  pruning, promotion swaps, and audited demotion swaps. Structural and demotion
  projections are separate and un-jitted, and each copies its decision values to
  the host once before the thresholds and ordering are applied. It is constructed with
  the certificate controller because `apply_structural_edits` returns with
  certificates already refreshed for the edited parameters; the disabled path
  returns first and deliberately does not refresh. `precision_weight_gradients`
  stays a module-level function so the training update can close over plain
  configuration values instead of a controller object.

`hibacaml/control/ranking.py`
: Stateless calculations, defined by a property rather than a topic: a function
  belongs here if it is pure over its arguments and imports no controller,
  trainer, bank, or reporting module. Posterior ranking and summaries, reserve
  recruitment diagnosis, the phi neighbourhood, candidate chunking, and support
  set geometry.

`hibacaml/control/scoring.py`
: `SupportScorer` — audit-bundle construction, the boundary objective,
  batched static support rows, and the old-task audit cache. The trainer is a
  parameter on every call rather than an attribute, because rollout scoring
  evaluates candidates on cloned trainers and the clone should be visible at the
  call site. The audit cache is the class's only state; its key covers trainer
  identity, bundle identity, parameter revision, and certificate revision.

`hibacaml/control/replay_bank.py`
: `SelectorBank` — the file-backed row store and its cosine
  retrieval. A row's context is copied into a private read-only float32 array at
  every ingress: `np.asarray` does not copy an array that is already float32, a
  frozen dataclass blocks reassigning the field rather than writing through it,
  and `add` and `load` therefore copy again even after `make_replay_row`. Row norms are
  memoized per bank revision, which `add` and `load` bump. The epsilon is added in
  float32 at its own site: there it vanishes for a unit-magnitude norm, while
  widening the norm first would keep it and could reorder proportional contexts.

`hibacaml/control/search.py`
: `ExactSearchService` — the decisions and their consequences: rollout
  coordination, replay-bank proposal and acceptance policy, boundary search,
  local one-swap and demotion-swap orchestration, installing accepted supports
  and parameters, controller tables, timing, and events.

`hibacaml/control/roles.py`
: Per-unit shell tier labels (roles) and their arithmetic: physical labels,
  validation, the per-label write scales (`LABEL_WRITE_SCALES`) and the per-unit
  table built from them, unit usage,
  same-label redundancy, multi-context promotion evidence, adjacent relabelling,
  labelled inhibition and the pruning trigger. Pure host NumPy.

`hibacaml/control/reuse.py`
: Restricted column reuse for Split-CIFAR-10: `Condition`, `ColumnAllocator`
  (unused capacity and committed history), replay validation, the teacher
  objective, harm check, paired bootstrap and decision rule, write-policy
  construction for a condition, validation evidence collection, and
  `BoundarySelector`, which runs the isolated trials and commits the choice.

`hibacaml/control/structure.py`
: Audited shell adaptation for Split-CIFAR-10: the round schedule, sparse
  parameter/moment patches (`Proposal`, `Patch`, `build_patch`, `apply_patch`,
  `verify_patch_scope`), the cumulative `HarmLedger`, and `StructuralController`,
  which runs relabel, inhibition and pruning rounds from the epoch hook.

### 1.4 Training

`hibacaml/training/trainer.py`
: The abstract `HiBaCaMLTrainer` base and the state all trainer methods share:
  construction, cloning, write-policy installation, support selection, the
  optimizer, complete-update construction, dynamic gradient masking, and the
  per-batch update with its structural edits, certificate refresh, and audits.
  Three operations form the abstract learner boundary — `build_programs`,
  `_training_inputs`, and `_gradients` — and everything else is shared by both
  learners. `HiBaCaMLTrainer` inherits its evaluation and loop methods from
  `TrainerEvaluationMixin` and `TrainerLoopMixin` (below). The mixins define
  methods only and hold no state, so a trainer and each clone are single objects.
  Learners and callers import `HiBaCaMLTrainer`, the optimizer helpers, and the
  program builders from `trainer.py`; only `trainer.py` imports the two mixin
  modules.

  `build_update_programs` closes over stable graph, optimizer, and resistance
  configuration while keeping parameters, optimizer state, clamps, support
  masks, write-policy inputs, and RNG dynamic. Its compiled core performs, in
  order: learner gradients, support masking, precision treatment, the
  write-policy gradient mask, the optimizer
  transition (optionally a global-norm clip before AdamW), update masking, the
  write-policy update mask and scale, parameter application, and finally the
  restoration of every protected coordinate's original bytes. The write-policy
  steps run only when a policy is installed. Program dictionaries are built once
  and shared unchanged with rollout clones.

  `build_evaluation_programs` constructs compiled raw-inference, detailed, and
  score-only programs from each learner's pure numerical functions. Detailed
  evaluation serves metrics and batch outputs; controller loss APIs use the
  score-only route. Its compiled core keeps the six-value loss vector as an
  unread auxiliary output, which keeps its multi-support totals identical to the
  detailed route. Candidate means stay
  outside JIT and cross the host boundary as one ordered vector.

  Two class constants declare where the learner contracts genuinely differ:
  `COMPOSER_BEFORE_PARENT` (objective summation order),
  `CLAMPS_MAY_INCLUDE_TARGETS` (whether supervised targets may be clamped into
  the graph at all). Public `training_update` is state-producing for
  controller rollouts. The ordinary task loop uses BP's lean update only when
  no structural edit, refresh, audit, maintenance, or reporting consumer needs
  the state; PC always returns its settled state.

`hibacaml/training/trainer_evaluation.py`
: `TrainerEvaluationMixin`: certificate refresh and clamp preparation for one
  or several supports (used by training and evaluation), dispatch to the
  compiled programs, composer diagnostics, controller support scoring, and
  loader and task evaluation. It prepares target-free clamps and external
  targets, owns support ordering and the RNG split for each evaluated batch,
  refreshes certificates on the first requested loader batch, and installs the
  graph state of raw inference.

`hibacaml/training/trainer_loop.py`
: `TrainerLoopMixin`: `train_batches` (fixed batches, used by isolated
  trials), `_train_epoch` (epoch aggregation, the `on_epoch_end` hook, then
  validation with the RNG restored), and `train_task` (boundary support,
  epochs, the saved support, the completion evaluation, task summaries, and
  events).

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
  `(logits, per_sample_total, loss_vector, composer)`. `composer_before_parent` preserves the
  learners' difference in objective summation order.

  Training loss reporting materializes the six-element loss vector once before
  converting it to a Python dictionary. Evaluation candidate means, loader
  finalization, and composer reporting each copy their values to the host in
  one grouped transfer.

  The module's identity is what it refuses to import: no trainer, controller,
  selector bank, persistent state, or reporting. Everything stateful is passed
  in as a plain argument.

`hibacaml/training/pc.py`
: `HiBaCaMLPCTrainer`, the compiled state-producing PC update, evaluation and
  inference programs, and the auxiliary-factor
  gradients. Feedforward initialization, the latent-state relaxation
  with supervised targets clamped, and local graph-energy weight gradients. For
  evaluation it supplies pure target-free inference and detailed-scoring
  functions registered by the trainer's shared program builder.
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
  training; a partial target set is rejected. `build_programs` refuses a graph
  built with stock `InferenceSGD`, since the two auxiliary terms would then
  appear in the reported loss without affecting learning. The auxiliary weights
  are the whole control surface: with `parent_child_loss_weight` and the three
  composer penalty weights at zero, both factors take an early return and the
  update reduces to ordinary three-term predictive coding.

`hibacaml/training/backprop.py`
: `HiBaCaMLBackpropTrainer`, an end-to-end autodiff learner preserving the same
  support, certificate, shell-edit, and audit semantics. Its forward pass is
  exactly the PC learner's initialization without the relaxation that follows
  it there. Its pure gradient step feeds both compiled complete-update variants:
  a state-producing form and a lean form that omits the graph-state result. For
  evaluation its pure feedforward and detailed-scoring functions feed the same
  compiled evaluation builder as PC.

`hibacaml/training/write_policy.py`
: `WritePolicy` (trainable columns, frozen feature pools and the per-unit write
  scale table) and the masking, scaling and byte-restoring functions the compiled
  update applies. Installing any policy also freezes the stem and the composer's
  feature/output maps. Backprop only.

Both learners are siblings under the abstract base rather than one inheriting
the other. The stage order of an update is fixed in the shared compiled-update
builder, and rollout trials run the same
state-producing template, so no trial can substitute a different update
algorithm.

### 1.5 Runner wiring and reporting

`hibacaml/experiment.py`
: The shared runner wiring: `build_trainer` (inference, graph,
  parameter initialization, trainer class), `prepare_run_root` (exclusive run
  directory plus config re-rooting), `log_runtime` (backend, devices, git
  revision and JAX versions), `architecture_metadata` (layout, tokens and
  parameter count, for the CIFAR runners) and `require_production_runtime` (GPU with JIT, for the CIFAR
  runners).

`hibacaml/reporting/`
: Four modules, one job each. `logger.py` is the progress logger and
  imports only the standard library; `rollout_logging()` demotes clone progress lines to DEBUG. `export.py` owns
  snapshot assembly, task artifact export, checkpoint serialization, the final
  bundle, and the two files written while a run is in flight (`events.jsonl`,
  `heartbeat.json`). `build_run_snapshot` is deliberately stateful: it evaluates
  every saved support and advances evaluation RNG. It never refreshes
  certificates, and neither does any other reporting evaluation.
  `export.py` also holds the run records every runner writes
  (`write_run_records`: `run_summary.json`, `task_summary.csv` and extra CSV
  tables) and the shared continual-learning metrics (`mean_seen_accuracy`,
  `mean_forgetting`, `continual_extras`). `plots.py` holds the figures for
  static and sequential runs and selects the headless Agg backend on import.
  `integrity.py` holds the run-invariant checks (`require`, `IntegrityError`,
  `require_finite`) and byte-exact digests used for protection and isolation
  checks (`fingerprint`, `column_fingerprints`, `leaf_digests`,
  `learning_state_signature`, `preserve_reporting_state`).

### 1.6 Runners

`experiments/split_mnist.py`
: Split-MNIST: configuration overrides, task construction, training with the
  MNIST support controller, evaluation aggregation, summaries, and plots.

`experiments/mnist.py`
: Full-MNIST: one ten-class task on a fixed bank of columns, epoch-level
  validation, a configurable bank size, backprop or predictive coding.

`experiments/cifar.py`
: Full-CIFAR-10: one ten-class task on a fixed 4 shared + 5 adaptive layout,
  backprop, with checks that reserve columns never change.

`experiments/split_cifar.py`
: Split-CIFAR-10: restricted column reuse at each task boundary, audited shell
  adaptation from the epoch hook, exact write protection after the first task,
  and run records rewritten after every task. The controls are chosen in its
  settings block (see [EXPERIMENTS.md](EXPERIMENTS.md)).

## 2. Data and Control Flow

### 2.1 Split-MNIST

```text
make_hibacaml_config("default")
override(cfg, **OVERRIDES)
run_experiment(cfg, ...)                      # experiments/split_mnist.py
  -> log_runtime()
  -> build_split_mnist_tasks(cfg, limit=tasks_limit, validation_fraction)
  -> prepare_run_root(cfg, run_id)            # hibacaml/experiment.py
  -> build_trainer(cfg, tasks, learning)      # hibacaml/experiment.py
       -> create_hibacaml_structure(cfg, inference, graph_state_initializer)
       -> log protocol / graph / schedule summary
       -> initialize_params(...)
       -> HiBaCaMLBackpropTrainer or HiBaCaMLPCTrainer
  -> _run_tasks(trainer, ...)
       for each task:
         train_task(task)
           -> boundary support search if needed
           -> per batch: compiled update (gradient, support mask, precision,
              optimizer), shell structural edits (which refresh certificates),
              demotion audit, scheduled certificate refresh, one-swap audit
           -> per-epoch validation
           -> saved support, selector-bank save, completion evaluation
         evaluate_all_saved_supports(refresh_certificates=False)
         export_task_artifacts(trainer, task_id)
         save_checkpoint(trainer, task_id)
       build_run_snapshot(trainer), export_run_artifacts(snapshot, final)
       aggregate accuracy/forgetting/support trajectories
       write_run_records(...)
  -> plots
```

Full-MNIST and Full-CIFAR-10 train one task with a fixed support and all
controllers off. Full-MNIST then evaluates the test split with `evaluate_task`
and writes the snapshot, task artifacts and a checkpoint; Full-CIFAR-10 takes
its test metrics from the completion report and writes its run records and
plots.

With the predictive-coding learner, certificates also refresh on every training
batch while the clamps are built.

### 2.2 Split-CIFAR-10

```text
make_hibacaml_config("cifar")
override(cfg, **OVERRIDES)
run_experiment(cfg, support_policy, structure, promotions, replay_supports)   # experiments/split_cifar.py
  -> build_split_cifar_tasks(cfg, ...)
  -> prepare_run_root / build_trainer (backprop, kernel-only shared K inputs)
  -> install initial role labels
  -> BoundarySelector (control/reuse.py), StructuralController (control/structure.py)
  -> for each task:
       before the second task: record protected leaves (stem, composer maps, shared columns)
       condition = selector.select(...)       # isolated trials for restricted reuse
       selector.commit(...)                   # allocation + write policy on the real model
       train_task(task, on_epoch_end=hook)
         -> compiled update with support mask and write policy
         -> certificate refresh on schedule
         -> hook: protection check, scheduled shell-adaptation round, protection check
         -> per-epoch validation
       boundary checks: update/revision counts, saved supports, column isolation, protection
       evaluate test (completion pass) and validation for every seen task
       write_run_records(...)
  -> check round count and trial budget, rewrite run records with final status
  -> plots
```

The support mask is the central runtime control variable. It is clamped into the
graph as `support_mask`, and each `ElementwiseGateNode` reads its column's
scalar mask entry to zero or retain token, feature, and logit pathways.

## 3. Learners

The MNIST runners accept `learning = "pc"` or `"backprop"`; both CIFAR runners
use backprop only (Split-CIFAR-10's write policies are backprop-only).

- `HiBaCaMLPCTrainer` uses FabricPC inference and local predictive-coding weight
  gradients through `compute_local_weight_gradients`, plus the two auxiliary
  factors described in [EXPERIMENTS.md](EXPERIMENTS.md) (training procedure).
- `HiBaCaMLBackpropTrainer` uses feedforward state initialization and JAX
  autodiff for supervised losses, while preserving support masks, certificate
  refresh, gradient masking, shell edits, boundary search, local swaps, and
  evaluation semantics. Its target-free evaluation path is also feedforward-only
  and never falls through to iterative predictive-coding inference.

The runners default to `"backprop"`. This is a practical deviation from a purely
predictive-coding training story, but it keeps the HiBaCaML controller and
architecture unchanged.

## 4. Reading Order

To understand the code as a concrete realization of the paper, read in this
order:

1. `hibacaml/config/defaults.py` for the architectural constants and the two
   presets.
2. `hibacaml/data/common.py` for the shared task path, then `data/mnist.py` and
   `data/cifar.py` for the dataset-specific loading, transforms and seeds.
3. `hibacaml/graph/builder.py` for how the stem, columns, gates, composer, and
   hierarchy heads become a FabricPC graph, with the node definitions in
   `hibacaml/nodes/`.
4. `hibacaml/control/certificates.py` for internal certificates.
5. For the MNIST controller: `hibacaml/control/support.py`,
   `hibacaml/control/ranking.py`, `hibacaml/control/scoring.py`,
   `hibacaml/control/replay_bank.py`, `hibacaml/control/search.py`, then
   `hibacaml/control/shells.py`.
6. For the Split-CIFAR-10 controllers: `hibacaml/control/roles.py`,
   `hibacaml/training/write_policy.py`, `hibacaml/control/reuse.py`, then
   `hibacaml/control/structure.py`.
7. `hibacaml/training/shared.py` for clamp assembly, objective primitives,
   and evaluation scoring/accumulation — everything both learners compute
   without touching trainer state.
8. `hibacaml/training/trainer.py` for the shared base and the update
   template, `hibacaml/training/trainer_loop.py` for the epoch and task loop,
   and `hibacaml/training/trainer_evaluation.py` for clamps and evaluation,
   then `hibacaml/training/pc.py` and
   `hibacaml/training/backprop.py` for the two learner contracts.
9. `hibacaml/experiment.py` for the shared runner wiring, and
   `hibacaml/reporting/` for run records and plots.
10. The runners in `experiments/`. Each contains only its own protocol: task
    construction, protocol assertions, metric aggregation, and reporting.

The shortest conceptual summary is: the paper proposes two coupled probabilistic
levels for continual learning, and the code realizes them as a sparse gated
column graph plus audited support and shell controllers around that graph.
