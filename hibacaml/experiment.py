"""Shared experiment wiring for the HiBaCaML runners.

Graph construction, parameter initialization, runner selection, and run-root
preparation are identical for Split- and Full-MNIST; only the task set and the
recorded metadata differ. Keeping them here means a runner is just a protocol
description.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

from hibacaml.config import HiBaCaMLConfig
from hibacaml.reporting.logger import log_progress
from hibacaml.graph import create_hibacaml_structure
from hibacaml.training import HiBaCaMLBackpropTrainer, HiBaCaMLPCTrainer
from hibacaml.training.pc import HiBaCaMLPCInference
from hibacaml.types import MnistTask

LEARNERS = ("pc", "backprop")


def prepare_run_root(
    cfg: HiBaCaMLConfig,
    run_id: str,
    *,
    selector_state_root: Optional[str] = None,
) -> Tuple[HiBaCaMLConfig, Path]:
    """Create an exclusive run directory and re-root the config onto it.

    Refusing an existing directory is deliberate: run bundles are evidence, so a
    new run never writes over an old one.
    """
    run_output_root = cfg.experiment_root_path() / run_id
    if run_output_root.exists():
        raise FileExistsError(
            f"run directory already exists: {run_output_root} "
            "(choose a new run_id for a new declared protocol)"
        )
    run_output_root.mkdir(parents=True)
    cfg = dataclasses.replace(
        cfg,
        reporting=dataclasses.replace(
            cfg.reporting,
            experiment_root=str(run_output_root),
            selector_state_root=(
                selector_state_root
                or cfg.reporting.selector_state_root
                or str(run_output_root / "selector_state")
            ),
        ),
    )
    return cfg, run_output_root


def _log_run_summary(cfg, tasks, learning: str, structure) -> None:
    """Log the derived facts worth catching wrong in a run's first second.

    Everything here is computed from cfg/tasks/structure. Anything static about
    the algorithm belongs in the docs, where it gets reviewed.
    """
    pool = cfg.column_pool
    classes = len(tasks[0].classes) if tasks else 0
    protocol = "Full-MNIST" if len(tasks) == 1 and classes == 10 else "Split-MNIST"
    log_progress(
        f"protocol={protocol} tasks={len(tasks)} classes={classes} "
        f"learning={learning}",
        component="runner",
    )
    log_progress(
        f"graph nodes={len(structure.nodes)} columns={pool.total_columns} "
        f"({pool.shared_count} shared + {pool.adaptive_count} adaptive + "
        f"{pool.reserve_count} reserve) active={pool.active_support_size} "
        f"scale={1.0 / pool.active_support_size:.6f}",
        component="runner",
    )
    log_progress(
        f"schedule epochs={cfg.epochs_per_task} batch={cfg.batch_size} "
        f"seed={cfg.seed} infer_steps={cfg.infer_steps} eta={cfg.eta_infer}",
        component="runner",
    )


def build_trainer(
    cfg: HiBaCaMLConfig,
    tasks: Sequence[MnistTask],
    learning: str,
    *,
    run_id: str = "",
    experiment_metadata: Optional[Dict[str, object]] = None,
):
    """Build the graph, initialize parameters, and construct one runner."""
    if learning not in LEARNERS:
        raise ValueError(f"learning must be one of {LEARNERS}")

    # Both learners get the same graph. Backprop reaches it only through
    # feedforward_state, which never runs the inference algorithm, so carrying
    # the PC one costs it nothing.
    inference = HiBaCaMLPCInference(
        eta_infer=cfg.eta_infer,
        infer_steps=cfg.infer_steps,
    )
    structure = create_hibacaml_structure(
        cfg,
        inference,
        graph_state_initializer=FeedforwardStateInit(),
    )
    _log_run_summary(cfg, tasks, learning, structure)

    params = initialize_params(structure, jax.random.PRNGKey(cfg.seed))
    log_progress("parameter initialization complete", component="runner")

    trainer_cls = (
        HiBaCaMLBackpropTrainer if learning == "backprop" else HiBaCaMLPCTrainer
    )
    trainer = trainer_cls(
        cfg,
        structure,
        params,
        tasks=tasks,
        run_id=run_id,
        experiment_metadata=experiment_metadata,
    )
    log_progress(
        f"trainer constructed class={type(trainer).__name__}", component="runner"
    )
    return structure, trainer
