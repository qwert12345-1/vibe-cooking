"""K-Means++ clustering with elbow analysis for choosing k using NumPy

Supports dense numpy matrices. For the recipe application, 
we run K-Means on 2-D or low-dim embeddings — not on the sparse
TF-IDF matrix directly — so dense arithmetic is fine and appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# stores the resuult of clustering: centroids, labels, inertia, and number of iterations
# returned by `fit_kmeans`
@dataclass
class KMeansResult:
    centroids: np.ndarray       # (k, d)
    labels: np.ndarray          # (n,)
    inertia: float              # sum of squared distances to nearest centroid
    n_iter: int


# initializes centroids using the K-Means++ strategy
# chooses first center uniformly random, then chooses later centers with probability
# proportional to squared distance from existing centers (gives better clustering than random initialization)
def _kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = X.shape
    centroids = np.empty((k, d), dtype=X.dtype)
    first = int(rng.integers(0, n)) # uniformly random chosen the first center
    centroids[0] = X[first]
    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1) # Squared distance of each point to its nearest already-chosen centroid
    for i in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            # Degenerate case — all points coincide; just do it again
            centroids[i] = centroids[0]
            continue
        probs = closest_sq / total # probability proportional to squared distance from existing centers
        idx = int(rng.choice(n, p=probs))
        centroids[i] = X[idx] # choose the next center
        new_sq = np.sum((X - centroids[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq) # update the squared distances to the nearest centroid
    return centroids


# assigns each data point to its nearest centroid and computes the inertia
# uses the identity ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c so the heavy
# lifting becomes a single BLAS matrix multiply of shape (n, d) × (d, k) 
def _assign(X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]        # squares every row, and smush it into a column vector
    c_sq = np.einsum("ij,ij->i", centroids, centroids) # same thing to the 20 centroids, and smush them into flat vector
    d2 = x_sq + c_sq[None, :] - 2.0 * (X @ centroids.T) # squared L2 distance 

    np.maximum(d2, 0.0, out=d2) # safety clip for floating point errors

    labels = np.argmin(d2, axis=1) # assign to its nearest centroid
    inertia = float(d2[np.arange(X.shape[0]), labels].sum()) # sum of squared L2 distances to nearest centroid
    return labels, inertia


# recomputes each centroid as the mean of the points assigned to it
# if a cluster ends up empty, it reseeds that centroid with a random data point
# so basically all recipes have chosen their teams, and now the centroid itself has to physically move to the 
# center to the exact average center of all the recipes that has joined its team :)
def _update(X: np.ndarray, labels: np.ndarray, k: int, prev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = X.shape[1]
    new = np.empty((k, d), dtype=X.dtype)
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            new[i] = X[int(rng.integers(0, X.shape[0]))]
        else:
            new[i] = X[mask].mean(axis=0)
    return new

# math... L2 norm of the input vectors (to get all vectors with length of 1.0)
# and to calculate cosine similarity by calculating L2 distance on normalized vectors
# which is the best way to compare TF-IDF ingredient lists!
def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


# main clustering function that supports both Euclidean and spherical K-Means for cosine similarity
# runs multiple random initializations and returns the best one
# each run, it alternates between assigning points to clusters and updating centroids till convergence
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


# elbow analysis to determine the optimal number of clusters
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


# simple elbow heristic to pick the best k by treating the inertia curve as lots of 
# points on 2-D plane and connecting (k_min, inertia_min) to (k_max, inertia_max)
# then find the point that is farthest from that line
def choose_k_elbow(inertias: dict[int, float]) -> int:
    ks = sorted(inertias.keys())
    pts = np.array([[k, inertias[k]] for k in ks], dtype=np.float64)
    # Normalize so both axes have comparable scale.
    pts_n = (pts - pts.min(axis=0)) / (np.ptp(pts, axis=0) + 1e-12)

    # start and end points of the line
    start, end = pts_n[0], pts_n[-1]
    line = end - start
    line_norm = line / (np.linalg.norm(line) + 1e-12) 

    # calculate the distance of each point from the line
    distances = [] 
    for p in pts_n:
        v = p - start
        proj = np.dot(v, line_norm) * line_norm
        perp = v - proj
        distances.append(np.linalg.norm(perp))
    return int(ks[int(np.argmax(distances))])
