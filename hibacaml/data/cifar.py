"""CIFAR-10 task construction for HiBaCaML."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.data.common import (
    TASK_CLASS_PAIRS,
    _stratified_train_validation_indices,
    make_task,
)
from hibacaml.reporting.logger import log_progress
from hibacaml.types import Task

_MEAN = np.asarray((0.4914, 0.4822, 0.4465), dtype=np.float32)
_STD = np.asarray((0.2470, 0.2435, 0.2616), dtype=np.float32)
_CROP_PAD = 4
SEED_ROLES = ("split", "training", "trial", "bootstrap")


def task_seeds(seed: int, task_id: int) -> Dict[str, int]:
    """Independent per-task stream seeds spawned from `[seed, task_id]`, one per role."""
    children = np.random.SeedSequence([int(seed), int(task_id)]).spawn(len(SEED_ROLES))
    return {
        role: int(child.generate_state(1, dtype=np.uint64)[0])
        for role, child in zip(SEED_ROLES, children)
    }


def _load_cifar_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the raw uint8 CIFAR-10 arrays with flat integer labels."""
    # JAX is already required; selecting it keeps keras off a TensorFlow backend
    os.environ.setdefault("KERAS_BACKEND", "jax")
    from keras.datasets import cifar10

    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    return train_x, train_y.reshape(-1), test_x, test_y.reshape(-1)


def _normalize(images: np.ndarray) -> np.ndarray:
    """uint8 NHWC images to per-channel standardized float32."""
    return (images.astype(np.float32) / np.float32(255.0) - _MEAN) / _STD


def _augment(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random crop of the zero-padded uint8 image, then a random horizontal flip."""
    batch, height, width, _ = images.shape
    pad = ((0, 0), (_CROP_PAD, _CROP_PAD), (_CROP_PAD, _CROP_PAD), (0, 0))
    padded = np.pad(images, pad)
    # The draw order (rows, columns, flips) is part of the protocol.
    rows = rng.integers(0, 2 * _CROP_PAD + 1, size=batch)[:, None] + np.arange(height)
    cols = rng.integers(0, 2 * _CROP_PAD + 1, size=batch)[:, None] + np.arange(width)
    flip = rng.random(batch) < 0.5
    cols = np.where(flip[:, None], cols[:, ::-1], cols)
    return padded[np.arange(batch)[:, None, None], rows[:, :, None], cols[:, None, :]]


def _preflight(
    cfg: HiBaCaMLConfig,
    *,
    split: bool,
    limit: int | None,
    validation_fraction: float,
    eval_batch_size: int | None,
) -> Tuple[int, int]:
    """Reject settings that would build a different protocol; returns (tasks, eval batch)."""
    if cfg.task_local_heads != split:
        raise ValueError(
            "Split-CIFAR-10 requires task_local_heads=True"
            if split
            else "Full-CIFAR-10 requires task_local_heads=False"
        )
    if tuple(cfg.input_shape) != (32, 32, 3):
        raise ValueError("CIFAR-10 requires input_shape=(32, 32, 3)")
    count = cfg.num_tasks if split else 1
    if not 1 <= count <= len(TASK_CLASS_PAIRS):
        raise ValueError("num_tasks must be between 1 and 5")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        count = min(count, limit)
    eval_batch = cfg.batch_size if eval_batch_size is None else eval_batch_size
    for name, value in (
        ("batch_size", cfg.batch_size),
        ("eval_batch_size", eval_batch),
        ("train_batches_limit", cfg.train_batches_limit),
        ("test_batches_limit", cfg.test_batches_limit),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if cfg.seed < 0:
        raise ValueError("seed must be non-negative")
    if cfg.composer.query_dim < count:
        raise ValueError(f"composer.query_dim must be >= {count} task queries")
    return count, eval_batch


def build_full_cifar_task(
    cfg: HiBaCaMLConfig,
    *,
    validation_fraction: float = 0.10,
    augment: bool = True,
    eval_batch_size: int | None = None,
) -> Task:
    """Build one ten-class CIFAR-10 task; split and loader seeds are both `cfg.seed`.

    Training drops its last partial batch; each pass reshuffles, so no example is
    permanently excluded. Held-out loaders use `eval_batch_size` (default `cfg.batch_size`).
    """
    _, eval_batch = _preflight(
        cfg,
        split=False,
        limit=None,
        validation_fraction=validation_fraction,
        eval_batch_size=eval_batch_size,
    )
    train_x, train_y, test_x, test_y = _load_cifar_arrays()
    fit, validation = _stratified_train_validation_indices(
        train_y, validation_fraction=validation_fraction, seed=cfg.seed
    )
    task = make_task(
        0,
        tuple(range(10)),
        cfg,
        train=(train_x[fit], train_y[fit]),
        validation=(train_x[validation], train_y[validation]),
        test=(test_x, test_y),
        normalize=_normalize,
        train_seed=cfg.seed,
        augment=_augment if augment else None,
        drop_last=True,
        eval_batch_size=eval_batch,
    )
    log_progress(
        "full task built classes=(0,...,9) "
        f"fit_examples={len(task.train_loader.images)} "
        f"validation_examples={len(task.validation_loader.images)} "
        f"test_examples={len(task.test_loader.images)} "
        f"fit_batches={len(task.train_loader)} "
        f"validation_batches={len(task.validation_loader)} "
        f"test_batches={len(task.test_loader)} augment={bool(augment)}",
        component="data",
    )
    return task


def build_split_cifar_tasks(
    cfg: HiBaCaMLConfig,
    *,
    limit: int | None = None,
    validation_fraction: float = 0.10,
    augment: bool = True,
    eval_batch_size: int | None = None,
) -> Tuple[Task, ...]:
    """Build an ordered Task-IL prefix of the five CIFAR-10 class pairs.

    Each pair keeps official order, is split stratified on its original labels and
    maps its first class to output 0. `task_seeds` gives each task independent split
    and training streams; model initialization still uses `cfg.seed`. `limit` caps
    the prefix without changing `cfg.num_tasks`.
    """
    count, eval_batch = _preflight(
        cfg,
        split=True,
        limit=limit,
        validation_fraction=validation_fraction,
        eval_batch_size=eval_batch_size,
    )
    train_x, train_y, test_x, test_y = _load_cifar_arrays()

    tasks = []
    for task_id, classes in enumerate(TASK_CLASS_PAIRS[:count]):
        train_mask = np.isin(train_y, classes)
        test_mask = np.isin(test_y, classes)
        pair_x, pair_y = train_x[train_mask], train_y[train_mask]
        if not len(pair_y) or not np.any(test_mask):
            raise ValueError(f"task {task_id} classes={classes} has an empty training or test split")
        seeds = task_seeds(cfg.seed, task_id)
        fit, validation = _stratified_train_validation_indices(
            pair_y, validation_fraction=validation_fraction, seed=seeds["split"]
        )
        if not len(fit) or not len(validation):
            raise ValueError(f"task {task_id} classes={classes} has an empty fit or validation split")
        task = make_task(
            task_id,
            classes,
            cfg,
            train=(pair_x[fit], pair_y[fit]),
            validation=(pair_x[validation], pair_y[validation]),
            test=(test_x[test_mask], test_y[test_mask]),
            normalize=_normalize,
            train_seed=seeds["training"],
            augment=_augment if augment else None,
            drop_last=True,
            eval_batch_size=eval_batch,
        )
        tasks.append(task)
        log_progress(
            f"task={task_id} built classes={classes} "
            f"split_seed={seeds['split']} training_seed={seeds['training']} "
            f"fit_examples={len(task.train_loader.images)} "
            f"validation_examples={len(task.validation_loader.images)} "
            f"test_examples={len(task.test_loader.images)} "
            f"train_batches={len(task.train_loader)} "
            f"test_batches={len(task.test_loader)} augment={bool(augment)}",
            component="data",
        )
    return tuple(tasks)
