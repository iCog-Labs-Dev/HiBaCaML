"""MNIST task construction and pixel routing profiles for HiBaCaML."""

from __future__ import annotations

import os
from functools import partial
from typing import Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.control.router import RouterProfile, image_tangents
from hibacaml.data.common import (
    TASK_CLASS_PAIRS,
    _stratified_train_validation_indices,
    make_task,
)
from hibacaml.reporting.integrity import fingerprint
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


_ROWS, _COLS = np.indices((28, 28), dtype=np.float64)
ROUTER_HISTORIES = {(28, 28): ("anchor", 0), (7, 7): ("patch7", 1)}
DESKEWED_EXTENDED_TANGENT = "deskewed_extended_tangent"
SHIFT_ROTATION_TANGENT = "shift_rotation_tangent"
_STUDY_LABELS = {DESKEWED_EXTENDED_TANGENT: "D20", SHIFT_ROTATION_TANGENT: "T20"}
_MODE_FROM_STUDY_LABEL = {label: mode for mode, label in _STUDY_LABELS.items()}


def _deskew(image):
    """Deskew one digit about its intensity centroid and recenter it at (14,14)."""
    from scipy.ndimage import affine_transform

    image = image.astype(np.float64) / 255
    mass = image.sum()
    if not mass:
        return image
    center = np.array([(_ROWS * image).sum(), (_COLS * image).sum()]) / mass
    dr, dc = _ROWS - center[0], _COLS - center[1]
    variance = (dr * dr * image).sum() / mass
    covariance = (dr * dc * image).sum() / mass
    matrix = np.array([[1., 0.], [covariance / variance if variance > 0 else 0., 1.]])
    upright = affine_transform(image, matrix, offset=center - matrix @ [14., 14.],
                               order=1, mode="constant", cval=0.)
    return np.clip(upright, 0., 1.) * 255


def _router_pixels(images, *, deskew):
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.shape[1:] not in ((28, 28), (28, 28, 1)):
        raise ValueError("MNIST routing requires a batch of raw uint8 28x28 images")
    images = images.reshape(len(images), 28, 28)
    if not deskew:
        return _normalize_images(images).reshape(len(images), 784)
    features = np.empty((len(images), 784), dtype=np.float32)
    for i, image in enumerate(images):
        features[i] = _normalize_images(_deskew(image)).ravel()
    return features


def router_profile(mode, patch_size):
    """Build a 20-centroid tangent profile, accepting earlier study labels on input."""
    mode = _MODE_FROM_STUDY_LABEL.get(mode, mode)
    patch_size = tuple(patch_size)
    if mode not in _STUDY_LABELS:
        raise ValueError(f"expected one of {', '.join(_STUDY_LABELS)} for the MNIST router, got {mode!r}")
    if patch_size not in ROUTER_HISTORIES:
        raise ValueError("MNIST routing requires patch size (28,28) or (7,7)")
    extended = mode == DESKEWED_EXTENDED_TANGENT
    bounds = (1., 1., np.pi / 18)
    if extended:
        bounds += (.10, .10, .5)
    return RouterProfile(
        identity={"format": "split_mnist_tangent_router_v1", "mode": mode, "patch_size": list(patch_size)},
        feature_dim=784, encode=partial(_router_pixels, deskew=extended),
        tangents=partial(image_tangents, image_shape=(28, 28, 1),
                         background=float(_normalize_images(np.uint8(0))), extended=extended),
        bounds=bounds, seed_prefix=(0, ROUTER_HISTORIES[patch_size][1], 0, 1), centroids_per_support=20,
        identity_aliases=({"mode": _STUDY_LABELS[mode]},),
    )


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
    row_digests: dict[int, dict[str, str]] | None = None,
) -> Tuple[Task, ...]:
    """Build the task-incremental Split-MNIST tasks.

    `limit` caps how many tasks are built. It deliberately does not go through
    `cfg.num_tasks`, which also sizes the replay-bank context vector's task
    one-hot (see `hibacaml/control/replay_bank.py`).

    Each task's validation split is carved from its own training examples. The
    structural audits read it; they never read test data.
    When requested, record digests of the ordered original fit and validation row IDs.
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
        if row_digests is not None:
            original = np.flatnonzero(train_mask)
            row_digests[task_id] = {
                "fit_rows_sha256": fingerprint(original[fit]),
                "validation_rows_sha256": fingerprint(original[validation]),
            }
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
