"""MNIST task construction for HiBaCaML."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.data.common import (
    TASK_CLASS_PAIRS,
    _stratified_train_validation_indices,
    make_task,
)
from hibacaml.reporting.logger import log_progress
from hibacaml.types import Task


def _load_mnist_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw uint8 MNIST arrays."""

    # JAX is already required; selecting it keeps keras off a TensorFlow backend
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from keras.datasets import mnist

    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    return train_x, train_y, test_x, test_y


def _normalize_images(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32) / 255.0
    if images.ndim == 3:
        images = images[..., None]
    mean = np.float32(0.1307)
    std = np.float32(0.3081)
    return ((images - mean) / std).astype(np.float32)


def _make_task(task_id, classes, cfg, *, train, validation, test, shuffle_seed) -> Task:
    # TODO: Use a deterministic epoch-dependent seed so training reshuffles between epochs.
    return make_task(
        task_id,
        classes,
        cfg,
        train=train,
        validation=validation,
        test=test,
        normalize=_normalize_images,
        train_seed=shuffle_seed,
        reshuffle_each_pass=False,
    )


def build_split_mnist_tasks(
    cfg: HiBaCaMLConfig,
    *,
    limit: int | None = None,
    validation_fraction: float = 0.10,
) -> Tuple[Task, ...]:
    """Build the task-incremental Split-MNIST tasks.

    `limit` caps how many tasks are built. It deliberately does not go through
    `cfg.num_tasks`, which also sizes the replay-bank context vector's task
    one-hot (see `hibacaml/control/replay_bank.py`).

    Each task's validation split is carved from its own training examples. The
    structural audits read it; they never read test data.
    """
    train_x, train_y, test_x, test_y = _load_mnist_arrays()
    task_count = cfg.num_tasks if limit is None else min(cfg.num_tasks, limit)

    tasks = []
    for task_id, classes in enumerate(TASK_CLASS_PAIRS[:task_count]):
        train_mask = np.isin(train_y, np.asarray(classes))
        test_mask = np.isin(test_y, np.asarray(classes))
        task_x, task_y = train_x[train_mask], train_y[train_mask]
        fit, validation = _stratified_train_validation_indices(
            task_y,
            validation_fraction=validation_fraction,
            seed=cfg.seed + task_id,
        )
        task = _make_task(
            task_id,
            classes,
            cfg,
            train=(task_x[fit], task_y[fit]),
            validation=(task_x[validation], task_y[validation]),
            test=(test_x[test_mask], test_y[test_mask]),
            shuffle_seed=cfg.seed + task_id,
        )
        tasks.append(task)
        log_progress(
            f"task={task_id} built classes={classes} "
            f"train_examples={task.train_loader.images.shape[0]} "
            f"validation_examples={task.validation_loader.images.shape[0]} "
            f"test_examples={task.test_loader.images.shape[0]} "
            f"train_batches={len(task.train_loader)} "
            f"test_batches={len(task.test_loader)}",
            component="data",
        )
    return tuple(tasks)


def build_full_mnist_task(
    cfg: HiBaCaMLConfig,
    *,
    validation_fraction: float = 0.10,
    split_seed: int | None = None,
) -> Task:
    """Build one ten-class MNIST task with a held-out validation split."""
    if cfg.task_local_heads:
        raise ValueError("Full-MNIST requires task_local_heads=False")
    split_seed = cfg.seed if split_seed is None else int(split_seed)
    train_x, train_y, test_x, test_y = _load_mnist_arrays()
    fit_indices, validation_indices = _stratified_train_validation_indices(
        train_y,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )

    task = _make_task(
        0,
        tuple(range(10)),
        cfg,
        train=(train_x[fit_indices], train_y[fit_indices]),
        validation=(train_x[validation_indices], train_y[validation_indices]),
        test=(test_x, test_y),
        shuffle_seed=cfg.seed,
    )
    log_progress(
        "full task built classes=(0,...,9) "
        f"fit_examples={task.train_loader.images.shape[0]} "
        f"validation_examples={task.validation_loader.images.shape[0]} "
        f"test_examples={task.test_loader.images.shape[0]} "
        f"fit_batches={len(task.train_loader)} "
        f"validation_batches={len(task.validation_loader)} "
        f"test_batches={len(task.test_loader)}",
        component="data",
    )
    return task
