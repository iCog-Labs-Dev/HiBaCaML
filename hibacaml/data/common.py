"""Dataset-agnostic task construction shared by the data modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np

from hibacaml.config import HiBaCaMLConfig
from hibacaml.types import Task

TASK_CLASS_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))

Normalize = Callable[[np.ndarray], np.ndarray]
Augment = Callable[[np.ndarray, np.random.Generator], np.ndarray]


@dataclass
class TaskLoader:
    """Deterministic loader over raw uint8 images, normalized one batch at a time.

    A shuffled or augmented loader draws each pass from `default_rng((seed, passes))`;
    `reshuffle_each_pass=False` reuses `seed` every pass instead. `copy.copy` keeps the
    pass count, `dataclasses.replace` restarts it at zero.
    """

    images: np.ndarray
    targets: np.ndarray
    normalize: Normalize
    mid_targets: int
    batch_size: int
    shuffle: bool = False
    augment: Optional[Augment] = None
    reshuffle_each_pass: bool = True
    drop_last: bool = False
    seed: Optional[int] = None
    max_batches: Optional[int] = None
    passes: int = field(default=0, init=False)

    def __post_init__(self):
        if (self.shuffle or self.augment is not None) and self.seed is None:
            raise ValueError("a shuffled or augmented loader needs a seed")

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        indices = np.arange(self.images.shape[0])
        rng = None
        if self.shuffle or self.augment is not None:
            rng = np.random.default_rng(
                (self.seed, self.passes) if self.reshuffle_each_pass else self.seed
            )
            self.passes += 1
            if self.shuffle:
                rng.shuffle(indices)
        for batch_idx in range(len(self)):
            batch_ids = indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
            images = self.images[batch_ids]
            if self.augment is not None:
                images = self.augment(images, rng)
            x = self.normalize(images)
            y = self.targets[batch_ids]
            hier_mid, hier_global = _hierarchy_targets(x, y, self.mid_targets)
            yield {"x": x, "y": y, "hier_mid": hier_mid, "hier_global": hier_global}

    def __len__(self) -> int:
        count = self.images.shape[0]
        total = count // self.batch_size if self.drop_last else -(-count // self.batch_size)
        return total if self.max_batches is None else min(total, self.max_batches)

    @property
    def examples(self) -> int:
        """Examples one pass yields, honouring `drop_last` and `max_batches`."""
        return min(int(self.images.shape[0]), len(self) * self.batch_size)


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


def make_task(
    task_id: int,
    classes: Tuple[int, ...],
    cfg: HiBaCaMLConfig,
    *,
    train: Tuple[np.ndarray, np.ndarray],
    validation: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    normalize: Normalize,
    train_seed: int,
    augment: Optional[Augment] = None,
    reshuffle_each_pass: bool = True,
    drop_last: bool = False,
    eval_batch_size: int | None = None,
) -> Task:
    """Build one task from raw `(images, labels)` splits; held-out loaders stay clean and ordered."""

    def loader(split, batch_size, limit, **options):
        images, labels = split
        return TaskLoader(
            images=images,
            targets=_task_targets(labels, classes, cfg.output_dim, cfg.task_local_heads),
            normalize=normalize,
            mid_targets=cfg.hierarchy.mid_targets,
            batch_size=batch_size,
            max_batches=limit,
            **options,
        )

    held_out = cfg.batch_size if eval_batch_size is None else eval_batch_size
    return Task(
        task_id=task_id,
        classes=classes,
        train_loader=loader(
            train,
            cfg.batch_size,
            cfg.train_batches_limit,
            shuffle=True,
            augment=augment,
            reshuffle_each_pass=reshuffle_each_pass,
            drop_last=drop_last,
            seed=train_seed,
        ),
        validation_loader=loader(validation, held_out, cfg.test_batches_limit),
        test_loader=loader(test, held_out, cfg.test_batches_limit),
        task_query=_task_query(task_id, cfg.composer.query_dim),
        output_dim=cfg.output_dim,
    )
