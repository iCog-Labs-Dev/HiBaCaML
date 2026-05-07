"""Split-MNIST task construction for HiBaCaML."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.debug import log_progress
from hibacaml.types import SplitMnistTask

_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_TASK_CLASS_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


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


def _mnist_cache_path() -> Path:
    override = os.environ.get("FABRICPC_MNIST_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "fabricpc" / "mnist.npz"


def _download_mnist_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MNIST_URL, path)


def _load_mnist_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST arrays with a lightweight no-framework fallback."""
    try:
        from tensorflow.keras.datasets import mnist  # type: ignore

        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        return train_x, train_y, test_x, test_y
    except Exception:
        pass

    try:
        from keras.datasets import mnist  # type: ignore

        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        return train_x, train_y, test_x, test_y
    except Exception:
        pass

    cache_path = _mnist_cache_path()
    if not cache_path.exists():
        _download_mnist_npz(cache_path)

    with np.load(cache_path) as data:
        train_x = data["x_train"]
        train_y = data["y_train"]
        test_x = data["x_test"]
        test_y = data["y_test"]
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
    classes: Tuple[int, int],
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


def build_split_mnist_tasks(cfg: HiBaCaMLConfig) -> Tuple[SplitMnistTask, ...]:
    """Build the five task-incremental Split-MNIST tasks."""
    train_x, train_y, test_x, test_y = _load_mnist_arrays()
    train_x = _normalize_images(train_x)
    test_x = _normalize_images(test_x)

    tasks = []
    for task_id, classes in enumerate(_TASK_CLASS_PAIRS[: cfg.num_tasks]):
        train_mask = np.isin(train_y, np.asarray(classes))
        test_mask = np.isin(test_y, np.asarray(classes))

        task_train_x = train_x[train_mask]
        task_test_x = test_x[test_mask]
        task_train_targets = _task_targets(
            train_y[train_mask],
            classes,
            cfg.output_dim,
            cfg.task_local_heads,
        )
        task_test_targets = _task_targets(
            test_y[test_mask],
            classes,
            cfg.output_dim,
            cfg.task_local_heads,
        )
        task_train_mid, task_train_global = _hierarchy_targets(
            task_train_x,
            task_train_targets,
            cfg.hierarchy.mid_targets,
        )
        task_test_mid, task_test_global = _hierarchy_targets(
            task_test_x,
            task_test_targets,
            cfg.hierarchy.mid_targets,
        )

        train_loader = _ArrayTaskLoader(
            images=task_train_x,
            targets=task_train_targets,
            hierarchy_mid=task_train_mid,
            hierarchy_global=task_train_global,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed + task_id,
            max_batches=cfg.train_batches_limit,
        )
        test_loader = _ArrayTaskLoader(
            images=task_test_x,
            targets=task_test_targets,
            hierarchy_mid=task_test_mid,
            hierarchy_global=task_test_global,
            batch_size=cfg.batch_size,
            shuffle=False,
            seed=None,
            max_batches=cfg.test_batches_limit,
        )
        tasks.append(
            SplitMnistTask(
                task_id=task_id,
                classes=classes,
                train_loader=train_loader,
                test_loader=test_loader,
                task_query=_task_query(task_id, cfg.composer.query_dim),
                output_dim=cfg.output_dim,
            )
        )
        log_progress(
            f"task={task_id} built classes={classes} "
            f"train_examples={task_train_x.shape[0]} test_examples={task_test_x.shape[0]} "
            f"train_batches={len(train_loader)} test_batches={len(test_loader)}",
            component="data",
        )
    return tuple(tasks)
