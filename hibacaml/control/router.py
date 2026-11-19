"""Prototype support routing with a caller-supplied input representation and tangent basis.

Create Router(profile, history), acquire one entry per saved support from current
fit inputs, then call route(inputs). Only the profile interprets the input batch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations, product
from operator import index
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hibacaml.reporting.integrity import fingerprint


@dataclass(frozen=True)
class RouterProfile:
    """Define a versioned representation for acquisition and inference.

    encode(batch) returns [examples, feature_dim]. tangents(centers) returns
    [centroids, directions, feature_dim] in the same coordinates. The JSON
    identity names the format and its preprocessing variants for bank loading.
    Change that identity whenever the encoder or tangent definitions change.
    Identity aliases supply earlier spellings of existing metadata fields during loading.
    """

    identity: dict
    feature_dim: int
    encode: Callable[[np.ndarray], np.ndarray]
    tangents: Callable[[np.ndarray], np.ndarray]
    bounds: tuple[float, ...]
    seed_prefix: tuple[int, ...]
    centroids_per_support: int = 20
    identity_aliases: tuple[dict, ...] = ()

    def __post_init__(self):
        # Round-trip the descriptor once, so JSON lists and Python tuples identify
        # the same bank and a caller's original metadata dictionary can be reused.
        object.__setattr__(self, "identity", json.loads(json.dumps(self.identity, allow_nan=False)))
        reserved = {"history", "k", "bounds", "boundary", "entries", "checkpoint_sha256"}
        format_name = self.identity.get("format")
        if not isinstance(format_name, str) or not format_name or reserved.intersection(self.identity):
            raise ValueError("profile identity needs a versioned format and distinct metadata keys")
        aliases = json.loads(json.dumps(self.identity_aliases, allow_nan=False))
        if any(not isinstance(alias, dict) or not alias.keys() <= self.identity.keys() for alias in aliases):
            raise ValueError("profile aliases may only rename existing identity fields")
        object.__setattr__(self, "identity_aliases", tuple(aliases))
        object.__setattr__(self, "feature_dim", index(self.feature_dim))
        object.__setattr__(self, "centroids_per_support", index(self.centroids_per_support))
        object.__setattr__(self, "bounds", tuple(map(float, self.bounds)))
        object.__setattr__(self, "seed_prefix", tuple(map(index, self.seed_prefix)))
        if self.feature_dim < 1 or self.centroids_per_support < 1:
            raise ValueError("feature dimension and centroid count must be positive")
        if any(seed < 0 for seed in self.seed_prefix):
            raise ValueError("fit seed components must be nonnegative")
        if not self.bounds or not all(np.isfinite(b) and b > 0 for b in self.bounds):
            raise ValueError("each tangent direction needs a finite positive bound")

    def fit_seed(self, boundary):
        return [*self.seed_prefix, boundary]

    def bank_identity(self, history):
        return dict(self.identity, history=history, k=self.centroids_per_support, bounds=list(self.bounds))


def _sqdist(x, x_norm, centers):
    return np.maximum(x_norm[:, None] - 2 * (x @ centers.T) + (centers * centers).sum(axis=1), 0.)


def _fit_centers(features, seed, count):
    """Pooled Euclidean k-means++ followed by at most 100 Lloyd iterations."""
    x = features.astype(np.float64)
    if len(x) < count:
        raise ValueError(f"each router entry requires at least {count} fit examples")
    rng = np.random.default_rng(seed)
    x_norm = (x * x).sum(axis=1)
    centers = np.empty((count, x.shape[1]), dtype=np.float64)
    centers[0] = x[rng.integers(len(x))]
    closest = np.full(len(x), np.inf)
    for i in range(1, count):
        closest = np.minimum(closest, _sqdist(x, x_norm, centers[i - 1:i])[:, 0])
        mass = closest.sum()
        row = rng.choice(len(x), p=closest / mass) if mass else rng.integers(len(x))
        centers[i] = x[row]

    previous = None
    for iteration in range(1, 101):
        assigned = _sqdist(x, x_norm, centers).argmin(axis=1)
        if np.array_equal(assigned, previous):
            break
        for i in range(count):
            members = x[assigned == i]
            if len(members):
                centers[i] = members.mean(axis=0)
        previous = assigned
    inertia = _sqdist(x, x_norm, centers).min(axis=1).sum()
    return centers.astype(np.float32), iteration, float(inertia)


def image_tangents(centers, *, image_shape, background, extended=False):
    """Translation/rotation tangents for flattened HWC images, with optional scale, shear and thickness.

    Spatial transforms share coefficients across channels. Supply each channel's
    background value in the encoder's coordinates for centered differences at the border.
    """
    height, width, channels = image_shape
    images = np.empty((len(centers), height + 2, width + 2, channels), dtype=np.float64)
    images[:] = background
    images[:, 1:-1, 1:-1] = centers.reshape(-1, *image_shape)
    gx = (images[:, 1:-1, 2:] - images[:, 1:-1, :-2]) / 2
    gy = (images[:, 2:, 1:-1] - images[:, :-2, 1:-1]) / 2
    x = (np.arange(width) - (width - 1) / 2)[None, :, None]
    y = (np.arange(height) - (height - 1) / 2)[:, None, None]
    rows = [-gx, -gy, y * gx - x * gy]
    if extended:
        rows.extend([-(x * gx + y * gy), -y * gx, np.hypot(gx, gy)])
    return np.stack(rows, axis=1).reshape(len(centers), len(rows), -1)


class _TangentDistances:
    """Cached centroid geometry for min ||(x-c) - B.T @ a||² with joint box bounds."""

    def __init__(self, centers, basis, bounds):
        self.centers = centers
        self.basis = basis
        self.bounds = np.asarray(bounds)
        self.bc = np.einsum("cpi,ci->cp", basis, centers)
        gram = basis @ basis.transpose(0, 2, 1)
        # On each face q fixes some coordinates at their bounds. The remaining
        # normal equations share one pseudoinverse across all signs and images.
        self.faces = []
        axes = range(len(bounds))
        for count in range(len(bounds) + 1):
            for fixed in combinations(axes, count):
                free = [i for i in axes if i not in fixed]
                inverse = np.linalg.pinv(gram[:, free][:, :, free], hermitian=True) if free else None
                for signs in product((-1, 1), repeat=count):
                    q = np.zeros(len(bounds))
                    q[list(fixed)] = self.bounds[list(fixed)] * signs
                    g = gram @ q
                    self.faces.append((free, inverse, q, g @ q, g[:, free]))

    def __call__(self, x):
        base = _sqdist(x, (x * x).sum(axis=1), self.centers)
        h = (x @ self.basis.reshape(-1, x.shape[1]).T).reshape(len(x), *self.bc.shape) - self.bc
        best = base.copy()  # zero deformation is always feasible
        for free, inverse, q, norm, coupling in self.faces:
            cost = base + norm - 2 * (h @ q)
            if free:
                rhs = h[:, :, free] - coupling
                solution = np.einsum("cij,ncj->nci", inverse, rhs)
                feasible = (np.abs(solution) <= self.bounds[free] + 1e-12).all(axis=2)
                cost -= np.einsum("nci,nci->nc", rhs, solution)
                np.minimum(best, cost, out=best, where=feasible)
            else:
                np.minimum(best, cost, out=best)
        return np.maximum(best, 0.)


class RoutingDecision(NamedTuple):
    catalogue: tuple[tuple[int, ...], ...]
    scores: np.ndarray
    choices: np.ndarray
    margins: np.ndarray


def canonical_support(values):
    """Return the sorted tuple of distinct nonshared column indices."""
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError(f"invalid nonshared support {values}")
    support = tuple(sorted(map(int, array)))
    if len(set(support)) != len(support):
        raise ValueError(f"repeated column in support {support}")
    return support


class Router:
    """One immutable centroid entry per acquired support. Loaded banks are inference-only."""

    def __init__(self, profile: RouterProfile, history: str):
        self.profile = profile
        self.history = history
        self._entries = {}  # acquisition order: support -> (float32 centroids, fit record)
        self._geometry = None
        self._frozen = False

    @property
    def supports(self):
        return tuple(self._entries)

    def _features(self, inputs):
        features = np.asarray(self.profile.encode(inputs))
        if (features.shape != (len(inputs), self.profile.feature_dim)
                or features.dtype.kind not in "fiu" or not np.isfinite(features).all()):
            raise ValueError("encoder must return a finite example-by-feature matrix matching the profile")
        return features

    def _append(self, support, centers, record):
        if support in self._entries:
            raise ValueError(f"duplicate whole support {support} at boundary {len(self.supports)}")
        shape = (self.profile.centroids_per_support, self.profile.feature_dim)
        if centers.shape != shape or centers.dtype != np.float32 or not np.isfinite(centers).all():
            raise ValueError(f"invalid centroids for support {support}")
        if fingerprint(centers) != record["centroid_sha256"]:
            raise ValueError(f"centroid digest mismatch for support {support}")
        centers.setflags(write=False)
        self._entries[support] = (centers, deepcopy(record))
        self._geometry = None

    def acquire(self, boundary, support, fit_inputs, provenance=None):
        """Append one observed nonshared tuple using the current task's ordered fit inputs."""
        if self._frozen or boundary != len(self.supports):
            raise ValueError("acquisition requires a live bank at the next boundary")
        support = canonical_support(support)
        if support in self._entries:
            raise ValueError(f"duplicate whole support {support} at boundary {boundary}")
        seed = self.profile.fit_seed(boundary)
        features = self._features(fit_inputs)
        centers, iterations, inertia = _fit_centers(features, seed, self.profile.centroids_per_support)
        record = dict(provenance or {}, support=support, seed=seed, examples=len(fit_inputs),
                      iterations=iterations, inertia=inertia, centroid_sha256=fingerprint(centers))
        self._append(support, centers, record)

    def route(self, inputs):
        """Score a batch against the lexicographically ordered bank. Margins use distinct supports."""
        catalogue = tuple(sorted(self._entries))
        if not catalogue:
            raise ValueError("cannot route with an empty support catalogue")
        x = self._features(inputs)
        if self._geometry is None:
            centers = np.concatenate([self._entries[s][0] for s in catalogue]).astype(np.float64)
            basis = np.asarray(self.profile.tangents(centers))
            shape = (len(centers), len(self.profile.bounds), self.profile.feature_dim)
            if basis.shape != shape or basis.dtype.kind not in "fiu" or not np.isfinite(basis).all():
                raise ValueError("tangent basis must match the profile's centroids, bounds and feature dimension")
            self._geometry = _TangentDistances(centers, basis.astype(np.float64, copy=False), self.profile.bounds)
        scores = np.empty((len(x), len(catalogue)), dtype=np.float64)
        for start in range(0, len(x), 256):
            chunk = x[start:start + 256].astype(np.float64)
            costs = self._geometry(chunk).reshape(len(chunk), len(catalogue), self.profile.centroids_per_support)
            scores[start:start + len(chunk)] = costs.min(axis=2)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite tangent distance")
        margins = np.full(len(x), np.nan)
        if len(catalogue) > 1:
            first, second = np.partition(scores, 1, axis=1)[:, :2].T
            margins = (second - first) / np.maximum(second, 1e-12)
        return RoutingDecision(catalogue, scores, scores.argmin(axis=1), margins)

    def save(self, path, checkpoint_sha256):
        """Save the complete prefix in acquisition order, bound to its model checkpoint."""
        if not self._entries:
            raise ValueError("cannot save an empty support catalogue")
        centers, records = zip(*self._entries.values())
        manifest = dict(self.profile.bank_identity(self.history), boundary=len(records) - 1, entries=records,
                        checkpoint_sha256=checkpoint_sha256)
        metadata = json.dumps(manifest, allow_nan=False)
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, manifest=metadata, centroids=np.stack(centers))

    @classmethod
    def load(cls, path, *, profile, history, supports, checkpoint_sha256):
        """Restore a frozen prefix after verifying its checkpoint and acquisition-order supports."""
        expected = tuple(map(canonical_support, supports))
        with np.load(path, allow_pickle=False) as bundle:
            manifest = json.loads(bundle["manifest"].item())
            centers = bundle["centroids"]
        identity = dict(profile.bank_identity(history), boundary=len(expected) - 1,
                        checkpoint_sha256=checkpoint_sha256)
        stored = {k: v for k, v in manifest.items() if k != "entries"}
        if not any(stored == (identity | alias) for alias in ({}, *profile.identity_aliases)):
            raise ValueError("router bank and requested model, profile or boundary disagree")
        records = manifest["entries"]
        saved = tuple(canonical_support(record["support"]) for record in records)
        if not expected or saved != expected or len(centers) != len(records):
            raise ValueError("router bank and saved support sequence disagree")
        return cls.from_entries(profile, history, zip(centers, records))

    @classmethod
    def from_entries(cls, profile, history, entries):
        """Build a frozen bank from (centroids, fit record) pairs in acquisition order."""
        bank = cls(profile, history)
        for boundary, (centers, record) in enumerate(entries):
            if record["seed"] != profile.fit_seed(boundary):
                raise ValueError(f"invalid acquisition seed at boundary {boundary}")
            support = canonical_support(record["support"])
            bank._append(support, np.array(centers, copy=True), dict(record, support=support))
        if not bank.supports:
            raise ValueError("cannot import an empty support catalogue")
        bank._frozen = True
        return bank
