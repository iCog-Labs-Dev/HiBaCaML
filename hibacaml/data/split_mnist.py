"""Split-MNIST task construction for HiBaCaML."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np

from hibacaml.types import SplitMnistTask

_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
_TASK_CLASS_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


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
    return images


def _task_query(task_id: int, query_dim: int) -> jnp.ndarray:
    if task_id >= query_dim:
        raise ValueError(f"task_id {task_id} >= query_dim {query_dim}: composer query would alias")
    return jnp.asarray(np.eye(query_dim, dtype=np.float32)[task_id])


def _one_hot(labels: np.ndarray, depth: int) -> np.ndarray:
    return np.eye(depth, dtype=np.float32)[labels]


def _build_hierarchy_targets(images: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build quadrant-level soft targets plus global task targets.

    Uses 4 quadrant midscale targets and 1 global target with a deterministic
    quadrant-sensitive encoding that keeps results reproducible and task-local.
    """
    batch = images.shape[0]
    one_hot = _one_hot(labels, 2)
    complement = _one_hot(1 - labels, 2)

    h_mid = images.shape[1] // 2
    w_mid = images.shape[2] // 2
    quadrants = (
        images[:, :h_mid, :w_mid, :],
        images[:, :h_mid, w_mid:, :],
        images[:, h_mid:, :w_mid, :],
        images[:, h_mid:, w_mid:, :],
    )
    quad_mass = np.stack(
        [np.mean(q, axis=(1, 2, 3)) for q in quadrants],
        axis=1,
    ).astype(np.float32)
    quad_mass = quad_mass / (np.max(quad_mass, axis=1, keepdims=True) + 1e-6)
    strength = 0.5 + 0.5 * quad_mass
    hier_mid = strength[..., None] * one_hot[:, None, :] + (1.0 - strength[..., None]) * complement[:, None, :]
    hier_mid = hier_mid / (np.sum(hier_mid, axis=-1, keepdims=True) + 1e-8)
    hier_global = one_hot.astype(np.float32).reshape(batch, 2)
    return hier_mid.astype(np.float32), hier_global


def _task_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    classes: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isin(labels, np.asarray(classes))
    task_x = images[mask]
    raw_labels = labels[mask]
    task_y = np.where(raw_labels == classes[0], 0, 1).astype(np.int32)
    hier_mid, hier_global = _build_hierarchy_targets(task_x, task_y)
    return task_x, _one_hot(task_y, 2), hier_mid, hier_global


def _batch_loader(
    x: np.ndarray,
    y: np.ndarray,
    hier_mid: np.ndarray,
    hier_global: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    batch_limit: int | None,
) -> Tuple[dict, ...]:
    indices = np.arange(x.shape[0], dtype=np.int32)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    batches = []
    for batch_idx, start in enumerate(range(0, len(indices), batch_size)):
        if batch_limit is not None and batch_idx >= batch_limit:
            break
        batch_ids = indices[start : start + batch_size]
        if batch_ids.size == 0:
            continue
        batches.append(
            {
                "x": jnp.asarray(x[batch_ids], dtype=jnp.float32),
                "y": jnp.asarray(y[batch_ids], dtype=jnp.float32),
                "hier_mid": jnp.asarray(hier_mid[batch_ids], dtype=jnp.float32),
                "hier_global": jnp.asarray(hier_global[batch_ids], dtype=jnp.float32),
            }
        )
    return tuple(batches)


def build_split_mnist_tasks(cfg) -> Tuple[SplitMnistTask, ...]:
    """Build the five task-incremental Split-MNIST tasks."""
    train_x, train_y, test_x, test_y = _load_mnist_arrays()
    train_x = _normalize_images(train_x)
    test_x = _normalize_images(test_x)

    tasks = []
    for task_id, classes in enumerate(_TASK_CLASS_PAIRS[: cfg.num_tasks]):
        task_train_x, task_train_y, task_train_mid, task_train_global = _task_arrays(train_x, train_y, classes)
        task_test_x, task_test_y, task_test_mid, task_test_global = _task_arrays(test_x, test_y, classes)
        task = SplitMnistTask(
            task_id=task_id,
            classes=classes,
            train_loader=_batch_loader(
                task_train_x,
                task_train_y,
                task_train_mid,
                task_train_global,
                batch_size=cfg.batch_size,
                seed=cfg.seed + task_id,
                shuffle=True,
                batch_limit=cfg.train_batches_limit,
            ),
            test_loader=_batch_loader(
                task_test_x,
                task_test_y,
                task_test_mid,
                task_test_global,
                batch_size=cfg.batch_size,
                seed=cfg.seed + 10_000 + task_id,
                shuffle=False,
                batch_limit=cfg.test_batches_limit,
            ),
            task_query=_task_query(task_id, cfg.composer.query_dim),
            output_dim=cfg.output_dim,
        )
        tasks.append(task)
    return tuple(tasks)
