"""MNIST task construction for HiBaCaML."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.reporting.logger import log_progress
from hibacaml.types import MnistTask

_TASK_CLASS_PAIRS: Tuple[Tuple[int, int], ...] = ( (0, 1),  (2, 3),  (4, 5),  (6, 7),  (8, 9) )


@dataclass
class _ArrayTaskLoader:
    """Deterministic lazy loader over in-memory task arrays."""

    images: np.ndarray
    targets: np.ndarray
    hierarchy_mid: np.ndarray
    hierarchy_global: np.ndarray
    batch_size: int
    shuffle: bool
    seed: Optional[int]
    max_batches: Optional[int] = None

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        indices = np.arange(self.images.shape[0], dtype=np.int32)
        if self.shuffle:
            # TODO: Use a deterministic epoch-dependent seed so training reshuffles between epochs.
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for batch_idx in range(len(self)):
            start = batch_idx * self.batch_size
            stop = min(start + self.batch_size, self.images.shape[0])
            batch_ids = indices[start:stop]
            if batch_ids.size == 0:
                continue
            yield {
                "x": self.images[batch_ids],
                "y": self.targets[batch_ids],
                "hier_mid": self.hierarchy_mid[batch_ids],
                "hier_global": self.hierarchy_global[batch_ids],
            }

    def __len__(self) -> int:
        total = (self.images.shape[0] + self.batch_size - 1) // self.batch_size
        if self.max_batches is None:
            return total
        return min(total, self.max_batches)

    @property
    def examples(self) -> int:
        """Examples this loader actually yields, honouring `max_batches`."""
        return min(int(self.images.shape[0]), len(self) * self.batch_size)


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


def _task_query(task_id: int, query_dim: int) -> np.ndarray:
    if task_id >= query_dim:
        raise ValueError(f"task_id {task_id} >= query_dim {query_dim}: composer query would alias")
    query = np.zeros((query_dim,), dtype=np.float32)
    query[task_id] = 1.0
    return query


def _task_targets(
    labels: np.ndarray,
    classes: Tuple[int, ...],
    output_dim: int,
    task_local_heads: bool,
) -> np.ndarray:
    if task_local_heads:
        mapped = np.where(labels == classes[0], 0, 1)
        return np.eye(output_dim, dtype=np.float32)[mapped]
    return np.eye(output_dim, dtype=np.float32)[labels]


def _hierarchy_targets(
    images: np.ndarray,
    targets: np.ndarray,
    mid_targets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build quadrant-aware soft targets plus the global class target."""
    if mid_targets != 4:
        mid = np.broadcast_to(
            targets[:, None, :],
            (targets.shape[0], mid_targets, targets.shape[1]),
        )
        return mid.astype(np.float32), targets.astype(np.float32)

    batch, height, width, _ = images.shape
    h_mid = height // 2
    w_mid = width // 2
    quadrants = (
        images[:, :h_mid, :w_mid, :],
        images[:, :h_mid, w_mid:, :],
        images[:, h_mid:, :w_mid, :],
        images[:, h_mid:, w_mid:, :],
    )
    masses = np.stack([np.mean(np.abs(q), axis=(1, 2, 3)) for q in quadrants], axis=1)
    masses = masses / np.maximum(np.max(masses, axis=1, keepdims=True), 1e-6)
    uniform = np.full((batch, targets.shape[1]), 1.0 / targets.shape[1], dtype=np.float32)
    alpha = masses[..., None].astype(np.float32)
    mid = uniform[:, None, :] + alpha * (targets[:, None, :] - uniform[:, None, :])
    return mid.astype(np.float32), targets.astype(np.float32)


def _stratified_train_validation_indices(
    labels: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint stratified fit and validation indices."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    labels = np.asarray(labels)
    classes = np.unique(labels)
    target_validation = int(round(labels.shape[0] * validation_fraction))

    # Largest-remainder apportionment: floor each class, then hand the leftover
    # slots to the largest fractional parts, ties broken by ascending class id.
    ideal = np.array(
        [np.sum(labels == class_id) for class_id in classes],
        dtype=np.float64,
    ) * validation_fraction
    counts = np.floor(ideal).astype(np.int64)
    order = np.argsort(-(ideal - counts), kind="stable")
    counts[order[: target_validation - int(counts.sum())]] += 1

    rng = np.random.default_rng(seed)
    fit_parts, validation_parts = [], []
    for class_id, validation_count in zip(classes, counts):
        class_indices = np.flatnonzero(labels == class_id).astype(np.int32)
        rng.shuffle(class_indices)
        validation_parts.append(class_indices[:validation_count])
        fit_parts.append(class_indices[validation_count:])

    fit_indices = np.concatenate(fit_parts).astype(np.int32, copy=False)
    validation_indices = np.concatenate(validation_parts).astype(np.int32, copy=False)
    rng.shuffle(fit_indices)
    rng.shuffle(validation_indices)
    if validation_indices.shape[0] != target_validation:
        raise RuntimeError("stratified validation allocation has the wrong size")
    if fit_indices.shape[0] + validation_indices.shape[0] != labels.shape[0]:
        raise RuntimeError("stratified split lost or duplicated examples")
    return fit_indices, validation_indices


def _make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    classes: Tuple[int, ...],
    cfg: HiBaCaMLConfig,
    *,
    shuffle: bool,
    seed: Optional[int],
    max_batches: int | None,
) -> _ArrayTaskLoader:
    """Turn one raw uint8 split into a deterministic loader.

    `images` stays uint8 until here: normalizing is elementwise, so normalizing
    per split is identical per element while avoiding a float32 copy of the whole
    dataset alongside every per-split copy.
    """
    normalized = _normalize_images(images)
    targets = _task_targets(labels, classes, cfg.output_dim, cfg.task_local_heads)
    hierarchy_mid, hierarchy_global = _hierarchy_targets(
        normalized,
        targets,
        cfg.hierarchy.mid_targets,
    )
    return _ArrayTaskLoader(
        images=normalized,
        targets=targets,
        hierarchy_mid=hierarchy_mid,
        hierarchy_global=hierarchy_global,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        seed=seed,
        max_batches=max_batches,
    )


def _make_task(
    task_id: int,
    classes: Tuple[int, ...],
    cfg: HiBaCaMLConfig,
    *,
    train: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    validation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    shuffle_seed: int,
) -> MnistTask:
    """Build one task from its raw `(images, labels)` splits.

    The only difference between the Split- and Full-MNIST protocols is which
    splits arrive here: Split passes train/test, Full adds validation.
    """

    def loader(split, shuffle, seed, limit):
        if split is None:
            return None
        return _make_loader(
            *split, classes, cfg, shuffle=shuffle, seed=seed, max_batches=limit
        )

    held_out = cfg.test_batches_limit
    return MnistTask(
        task_id=task_id,
        classes=classes,
        train_loader=loader(train, True, shuffle_seed, cfg.train_batches_limit),
        validation_loader=loader(validation, False, None, held_out),
        test_loader=loader(test, False, None, held_out),
        task_query=_task_query(task_id, cfg.composer.query_dim),
        output_dim=cfg.output_dim,
    )


def build_split_mnist_tasks(
    cfg: HiBaCaMLConfig,
    *,
    limit: int | None = None,
) -> Tuple[MnistTask, ...]:
    """Build the task-incremental Split-MNIST tasks.

    `limit` caps how many tasks are built. It deliberately does not go through
    `cfg.num_tasks`, which also sizes the replay-bank context vector's task
    one-hot (see `hibacaml/control/replay_bank.py`).
    """
    train_x, train_y, test_x, test_y = _load_mnist_arrays()
    task_count = cfg.num_tasks if limit is None else min(cfg.num_tasks, limit)

    tasks = []
    for task_id, classes in enumerate(_TASK_CLASS_PAIRS[:task_count]):
        train_mask = np.isin(train_y, np.asarray(classes))
        test_mask = np.isin(test_y, np.asarray(classes))
        task = _make_task(
            task_id,
            classes,
            cfg,
            train=(train_x[train_mask], train_y[train_mask]),
            test=(test_x[test_mask], test_y[test_mask]),
            shuffle_seed=cfg.seed + task_id,
        )
        tasks.append(task)
        log_progress(
            f"task={task_id} built classes={classes} "
            f"train_examples={task.train_loader.images.shape[0]} "
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
) -> MnistTask:
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
