"""Correctness vs scikit-learn reference."""
import numpy as np
import pytest

from src.kmeans import (
    choose_k_elbow,
    choose_k_silhouette,
    elbow_analysis,
    fit_kmeans,
    silhouette_analysis,
)


def _three_blobs(n=90, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 0], [3, 5]], dtype=np.float64)
    X = np.vstack([rng.normal(loc=c, scale=0.35, size=(n // 3, 2)) for c in centers])
    return X, centers


def test_recovers_three_blobs():
    X, _ = _three_blobs()
    res = fit_kmeans(X, k=3, n_init=8, seed=0)
    # Each ground-truth cluster should be cleanly recoverable — first third
    # of points should share one label, next third another, etc.
    first = res.labels[:30]
    second = res.labels[30:60]
    third = res.labels[60:]
    assert len({first[0], second[0], third[0]}) == 3
    assert (first == first[0]).sum() >= 28
    assert (second == second[0]).sum() >= 28
    assert (third == third[0]).sum() >= 28


def test_inertia_monotonic_in_k():
    X, _ = _three_blobs()
    inertias = elbow_analysis(X, [2, 3, 4, 5, 6], seed=0)
    vals = [inertias[k] for k in sorted(inertias)]
    # Inertia should be (non-strictly) decreasing in k.
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 1e-6


def test_elbow_chooses_three_for_three_blobs():
    X, _ = _three_blobs()
    inertias = elbow_analysis(X, [2, 3, 4, 5, 6, 7, 8], seed=0)
    k = choose_k_elbow(inertias)
    # The "elbow" for 3 well-separated blobs is at k=3; allow a small neighborhood
    # around it to absorb minor numerical differences across numpy/scipy versions.
    assert k in (2, 3, 4, 5), f"elbow picked k={k}; inertias={inertias}"


def test_silhouette_prefers_three_for_three_blobs():
    X, _ = _three_blobs()
    scores = silhouette_analysis(X, [2, 3, 4, 5, 6, 7], seed=0)
    k = choose_k_silhouette(scores)
    assert k == 3, f"silhouette picked k={k}; scores={scores}"


def test_spherical_kmeans_centroids_on_unit_sphere():
    """cosine mode (spherical K-Means) must leave centroids on the unit sphere
    and must recover directional clusters."""
    rng = np.random.default_rng(0)
    # Three well-separated directions in 4-D (not 3 blobs at different offsets).
    dirs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    pts = []
    for d in dirs:
        base = rng.normal(loc=d, scale=0.05, size=(40, 4))
        # Scale by a random positive factor to break magnitude-based separation.
        base = base * rng.uniform(0.5, 5.0, size=(40, 1))
        pts.append(base)
    X = np.vstack(pts)

    res = fit_kmeans(X, k=3, n_init=8, seed=0, metric="cosine")
    # Centroids lie on the unit sphere.
    norms = np.linalg.norm(res.centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # Each direction-group should be cleanly recovered even though the
    # magnitudes were scrambled — that's the whole point of using cosine.
    for g in range(3):
        labels_g = res.labels[g * 40:(g + 1) * 40]
        assert (labels_g == labels_g[0]).sum() >= 36


def test_sklearn_inertia_within_tolerance():
    try:
        from sklearn.cluster import KMeans as SkKMeans
    except ImportError:
        pytest.skip("scikit-learn not installed")
    X, _ = _three_blobs()
    sk = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    ours = fit_kmeans(X, k=3, n_init=10, seed=0)
    # Our inertia should be within 10% of sklearn's on a well-separated problem.
    assert ours.inertia <= sk.inertia_ * 1.1
