"""Shared experiment wiring for the HiBaCaML runners.

Graph construction, parameter initialization, runner selection, and run-root
preparation are identical for every protocol (Split-/Full-MNIST and CIFAR-10);
only the task set and the recorded metadata differ. Keeping them here means a
runner is just a protocol description.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import numpy as np
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

from hibacaml.config import HiBaCaMLConfig
from hibacaml.reporting.logger import log_progress
from hibacaml.graph import create_hibacaml_structure
from hibacaml.training import HiBaCaMLBackpropTrainer, HiBaCaMLPCTrainer
from hibacaml.training.pc import HiBaCaMLPCInference
from hibacaml.types import Task

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
    dataset = "CIFAR-10" if cfg.input_shape[-1] == 3 else "MNIST"
    protocol = f"{'Full' if len(tasks) == 1 and classes == 10 else 'Split'}-{dataset}"
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
    rows, cols = cfg.token_grid
    stem = cfg.stem
    log_progress(
        (f"stem channels={stem.channels} strides={stem.strides} pool={stem.pool_window} " if stem else "")
        + f"tokens={rows}x{cols} token_dim={cfg.patch_token_dim} "
        f"shared_k={structure.config['hibacaml']['shared_k_connectivity']}",
        component="runner",
    )
    log_progress(
        f"schedule epochs={cfg.epochs_per_task} batch={cfg.batch_size} "
        f"seed={cfg.seed} infer_steps={cfg.infer_steps} eta={cfg.eta_infer}",
        component="runner",
    )


def build_trainer(
    cfg: HiBaCaMLConfig,
    tasks: Sequence[Task],
    learning: str,
    *,
    run_id: str = "",
    experiment_metadata: Optional[Dict[str, object]] = None,
    shared_k_connectivity: str = "dense",
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
        shared_k_connectivity=shared_k_connectivity,
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


class IntegrityError(AssertionError):
    """A broken run invariant; `details` is recorded with the failure."""

    def __init__(self, message: str, details: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.details = details or {}


def require(condition, message: str, **details) -> None:
    if not condition:
        raise IntegrityError(message, details)


def require_finite(tree, *, label: str) -> None:
    """Reject nonfinite numeric leaves in model/report pytrees at task boundaries."""
    for path, value in jax.tree_util.tree_leaves_with_path(jax.device_get(tree)):
        if isinstance(value, (float, complex, np.number, np.ndarray)):
            require(np.all(np.isfinite(value)),
                    f"nonfinite value in {label}{jax.tree_util.keystr(path)}")


DETERMINISTIC_XLA_FLAG = "--xla_gpu_deterministic_ops=true"


def require_production_runtime(*, deterministic_xla: bool = False) -> None:
    """Production runs need a GPU with JIT (and, for paired runs, deterministic XLA ops)."""
    require(jax.default_backend() == "gpu" and not jax.config.jax_disable_jit,
            "production runs require a GPU backend with JIT enabled")
    if deterministic_xla:
        require(DETERMINISTIC_XLA_FLAG in os.environ.get("XLA_FLAGS", "").split(),
                f"production runs require XLA_FLAGS to contain {DETERMINISTIC_XLA_FLAG}")


def log_runtime() -> Dict[str, object]:
    """Log the backend and devices; return the runtime identity a GPU run is paired by."""
    log_progress(f"backend={jax.default_backend()} devices={jax.devices()}", component="runner")
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        revision = None
    return {
        "git_revision": revision,
        "jax": metadata.version("jax"),
        "jaxlib": metadata.version("jaxlib"),
        "backend": jax.default_backend(),
    }


def architecture_metadata(cfg: HiBaCaMLConfig, structure, params) -> Dict[str, object]:
    """Column layout, token layout and parameter count of a built CIFAR graph."""
    pool, meta = cfg.column_pool, structure.config["hibacaml"]
    return {
        "total_columns": pool.total_columns,
        "shared_columns": pool.shared_indices,
        "adaptive_columns": pool.adaptive_indices,
        "reserve_columns": pool.reserve_indices,
        "topk_nonshared": pool.topk_nonshared,
        "active_support_size": pool.active_support_size,
        "aggregation_scale": 1.0 / pool.active_support_size,
        "composer_topk": cfg.composer.topk,
        "stem": cfg.stem,
        "token_grid": cfg.token_grid,
        "token_dim": cfg.patch_token_dim,
        "shared_k_connectivity": meta["shared_k_connectivity"],
        "parameter_count": int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))),
    }
