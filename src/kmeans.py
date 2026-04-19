"""[HAND-IMPLEMENTED] K-Means++ clustering with elbow analysis for choosing k.

Implemented with numpy only. Supports dense numpy matrices. For the recipe
application, we run K-Means on 2-D or low-dim embeddings — not on the sparse
TF-IDF matrix directly — so dense arithmetic is fine and appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KMeansResult:
    centroids: np.ndarray       # (k, d)
    labels: np.ndarray          # (n,)
    inertia: float              # sum of squared distances to nearest centroid
    n_iter: int


def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-Means++ seeding: pick first center uniformly, subsequent centers with prob ∝ D²."""
    n, d = X.shape
    centroids = np.empty((k, d), dtype=X.dtype)
    first = int(rng.integers(0, n))
    centroids[0] = X[first]
    # Squared distance of each point to its nearest already-chosen centroid.
    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            # Degenerate case — all points coincide; just repeat.
            centroids[i] = centroids[0]
            continue
        probs = closest_sq / total
        idx = int(rng.choice(n, p=probs))
        centroids[i] = X[idx]
        new_sq = np.sum((X - centroids[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centroids


def _assign(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign each point to the nearest centroid. Return (labels, inertia).

    Uses the identity ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c so the heavy
    lifting becomes a single BLAS matrix multiply of shape (n, d) × (d, k).
    The previous implementation materialized an (n, k, d) broadcast tensor,
    which blew up memory and wall time at n=39k, k up to 50, d=50.
    """
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]        # (n, 1)
    c_sq = np.einsum("ij,ij->i", centroids, centroids) # (k,)
    d2 = x_sq + c_sq[None, :] - 2.0 * (X @ centroids.T)
    # Float rounding can push diagonal terms slightly negative — clip.
    np.maximum(d2, 0.0, out=d2)
    labels = np.argmin(d2, axis=1)
    inertia = float(d2[np.arange(X.shape[0]), labels].sum())
    return labels, inertia


def _update(X: np.ndarray, labels: np.ndarray, k: int, prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Recompute centroids as the mean of assigned points. Re-seed empty clusters."""
    d = X.shape[1]
    new = np.empty((k, d), dtype=X.dtype)
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            # Empty cluster: re-seed with a random data point to avoid collapse.
            new[i] = X[int(rng.integers(0, X.shape[0]))]
        else:
            new[i] = X[mask].mean(axis=0)
    return new


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def fit_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 200,
    n_init: int = 5,
    tol: float = 1e-4,
    seed: int = 42,
    metric: str = "euclidean",
) -> KMeansResult:
    """Run K-Means++ `n_init` times and return the run with lowest inertia.

    `metric`:
      - "euclidean" (default): classical Lloyd's algorithm minimizing
        Σ ||x_i - μ_k||². Appropriate when magnitude carries meaning.
      - "cosine" → *Spherical K-Means*: pre-L2-normalize X to the unit
        sphere and re-normalize centroids after each update. On the unit
        sphere ||x-y||² = 2 - 2·cos(x,y), so Euclidean Lloyd's on normalized
        inputs minimizes cosine *dissimilarity*. This is the right objective
        for TF-IDF / ingredient-bag vectors where *direction* (which
        ingredients) matters, not *magnitude* (how many ingredients).
    """
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric: {metric!r}")
    if metric == "cosine":
        X = _l2_normalize(X)

    best: KMeansResult | None = None
    for run in range(n_init):
        rng = np.random.default_rng(seed + run)
        centroids = _kmeans_plus_plus_init(X, k, rng)
        if metric == "cosine":
            centroids = _l2_normalize(centroids)
        inertia_prev = np.inf
        for it in range(max_iter):
            labels, inertia = _assign(X, centroids)
            new_centroids = _update(X, labels, k, centroids, rng)
            if metric == "cosine":
                # Project means back onto the unit sphere — the spherical
                # K-Means step that makes centroid-assignment consistent with
                # cosine similarity in the next iteration.
                new_centroids = _l2_normalize(new_centroids)
            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if shift < tol or abs(inertia_prev - inertia) < tol:
                break
            inertia_prev = inertia
        labels, inertia = _assign(X, centroids)
        result = KMeansResult(centroids=centroids, labels=labels, inertia=inertia, n_iter=it + 1)
        if best is None or result.inertia < best.inertia:
            best = result
    assert best is not None
    return best


def elbow_analysis(
    X: np.ndarray,
    k_values: list[int],
    seed: int = 42,
    n_init: int = 1,
    sample_size: int | None = None,
    metric: str = "euclidean",
) -> dict[int, float]:
    """Return {k: inertia} for each k.

    Fitting a *good* K-Means model is expensive, but picking k via the elbow
    heuristic only needs a rough inertia curve. Two knobs for speed:

      - `n_init=1`  : skip the multi-restart loop during the elbow sweep.
      - `sample_size`: fit on a random subsample (common on large corpora).
                       If `sample_size` is given and smaller than `len(X)`,
                       the per-k inertia is rescaled back to the full-corpus
                       scale so the curve shape is preserved.
    """
    data = X
    scale = 1.0
    n = X.shape[0]
    if sample_size is not None and sample_size < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        data = X[idx]
        scale = n / float(sample_size)
    return {
        k: fit_kmeans(data, k, seed=seed, n_init=n_init, metric=metric).inertia * scale
        for k in k_values
    }


def choose_k_elbow(inertias: dict[int, float]) -> int:
    """Heuristic elbow selection: the k that maximizes distance from the line
    connecting (k_min, inertia_min) to (k_max, inertia_max)."""
    ks = sorted(inertias.keys())
    pts = np.array([[k, inertias[k]] for k in ks], dtype=np.float64)
    # Normalize so both axes have comparable scale.
    pts_n = (pts - pts.min(axis=0)) / (np.ptp(pts, axis=0) + 1e-12)
    start, end = pts_n[0], pts_n[-1]
    line = end - start
    line_norm = line / (np.linalg.norm(line) + 1e-12)
    distances = []
    for p in pts_n:
        v = p - start
        proj = np.dot(v, line_norm) * line_norm
        perp = v - proj
        distances.append(np.linalg.norm(perp))
    return int(ks[int(np.argmax(distances))])
