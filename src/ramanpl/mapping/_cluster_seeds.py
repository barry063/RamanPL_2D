"""
_cluster_seeds.py
-----------------
Internal helpers for cluster-seeded warm-start fitting (v0.6.5).

All sklearn imports are lazy so that ``import ramanpl`` and default
``fit_spectra()`` calls do not require scikit-learn.  Only
``cluster_seeds=True`` triggers the import path.

Public surface: none.  Everything here is package-private.
"""

from math import sqrt as _sqrt

import numpy as np


# ---------------------------------------------------------------------------
# sklearn lazy import
# ---------------------------------------------------------------------------

def _require_sklearn_for_cluster_seeds():
    """Lazily import sklearn; raise an actionable ImportError if absent.

    Returns (StandardScaler, PCA, KMeans) on success.
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        return StandardScaler, PCA, KMeans
    except ImportError as exc:
        raise ImportError(
            "cluster_seeds=True requires scikit-learn. "
            "Install it with:  pip install ramanpl[ml]"
        ) from exc


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------

_VALID_CONFIG_KEYS = frozenset({"n_clusters", "n_components", "random_state"})


def _normalise_cluster_seed_config(cluster_seeds, X, Y):
    """Accept False, True, or a small dict; return a validated config dict or None.

    Parameters
    ----------
    cluster_seeds : False | True | dict
    X, Y : int  — map dimensions (columns, rows)

    Returns
    -------
    dict or None
        None when cluster_seeds is False (disabled).
        Dict with keys n_clusters, n_components, random_state otherwise.

    Raises
    ------
    ValueError  if unknown dict keys are present or values are out of range.
    TypeError   if dict values have wrong types or cluster_seeds has an
                unexpected type.
    """
    if cluster_seeds is False:
        return None

    if cluster_seeds is True:
        return {
            "n_clusters": min(8, max(1, int(_sqrt(X * Y)))),
            "n_components": 3,
            "random_state": None,
        }

    if isinstance(cluster_seeds, dict):
        unknown = set(cluster_seeds) - _VALID_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"Unknown key(s) in cluster_seeds dict: {sorted(unknown)}. "
                f"Accepted keys: {sorted(_VALID_CONFIG_KEYS)}."
            )
        cfg = {
            "n_clusters": min(8, max(1, int(_sqrt(X * Y)))),
            "n_components": 3,
            "random_state": None,
        }
        cfg.update(cluster_seeds)
        # Type checks (bool subclasses int, so accept it for n_clusters/n_components)
        if not isinstance(cfg["n_clusters"], int):
            raise TypeError(
                f"cluster_seeds['n_clusters'] must be an integer, "
                f"got {type(cfg['n_clusters']).__name__!r}: {cfg['n_clusters']!r}"
            )
        if not isinstance(cfg["n_components"], int):
            raise TypeError(
                f"cluster_seeds['n_components'] must be an integer, "
                f"got {type(cfg['n_components']).__name__!r}: {cfg['n_components']!r}"
            )
        if cfg["n_clusters"] < 1:
            raise ValueError(
                f"cluster_seeds['n_clusters'] must be >= 1, got {cfg['n_clusters']}"
            )
        if cfg["n_components"] < 1:
            raise ValueError(
                f"cluster_seeds['n_components'] must be >= 1, got {cfg['n_components']}"
            )
        return cfg

    raise TypeError(
        f"cluster_seeds must be False, True, or a dict; "
        f"got {type(cluster_seeds).__name__!r}"
    )


# ---------------------------------------------------------------------------
# Spectral feature matrix
# ---------------------------------------------------------------------------

def _spectral_feature_matrix(cube):
    """Convert a [Y, X, N] cube into a compressed feature matrix.

    Parameters
    ----------
    cube : ndarray [Y, X, N]

    Returns
    -------
    matrix : ndarray [P, N]
        Only rows where every spectral value is finite.  P = valid pixel count.
        Row order follows row-major (y outer, x inner) scan of the full grid.
    valid_mask : ndarray [Y, X], bool
        True at positions included in *matrix*.
    """
    Y, X, N = cube.shape
    flat = cube.reshape(Y * X, N)  # row-major: row idx = j*X + i
    valid = np.all(np.isfinite(flat), axis=1)
    valid_mask = valid.reshape(Y, X)
    matrix = flat[valid]
    return matrix, valid_mask


# ---------------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------------

def _cluster_spectra(cube, *, n_clusters, n_components, random_state):
    """Cluster preprocessed spectra with StandardScaler → PCA → KMeans.

    Parameters
    ----------
    cube : ndarray [Y, X, N]
    n_clusters : int
    n_components : int   PCA components; clamped to min(n_components, n_valid, N)
    random_state : int or None

    Returns
    -------
    labels : ndarray [Y, X], dtype intp
        Cluster index per pixel; -1 for invalid (non-finite) pixels.
    metadata : dict
        Keys: enabled, n_clusters_requested, n_clusters_effective,
              n_components, embedding, valid_coords, n_valid_pixels.
    """
    StandardScaler, PCA, KMeans = _require_sklearn_for_cluster_seeds()
    Y, X, N = cube.shape
    matrix, valid_mask = _spectral_feature_matrix(cube)
    n_valid = matrix.shape[0]

    labels = np.full((Y, X), -1, dtype=np.intp)

    if n_valid == 0:
        return labels, {
            "enabled": True,
            "n_clusters_requested": n_clusters,
            "n_clusters_effective": 0,
            "n_components": n_components,
            "embedding": np.empty((0, max(1, n_components))),
            "valid_coords": [],
            "n_valid_pixels": 0,
        }

    # Ordered (y, x) coordinates for valid pixels — row-major
    valid_coords = [
        (j, i) for j in range(Y) for i in range(X) if valid_mask[j, i]
    ]

    k = min(n_clusters, n_valid)
    n_comp = min(n_components, n_valid, N)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    pca = PCA(n_components=n_comp, random_state=random_state)
    embedding = pca.fit_transform(scaled)  # [n_valid, n_comp]

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embedding)  # [n_valid]

    for row_idx, (j, i) in enumerate(valid_coords):
        labels[j, i] = int(cluster_ids[row_idx])

    return labels, {
        "enabled": True,
        "n_clusters_requested": n_clusters,
        "n_clusters_effective": k,
        "n_components": n_comp,
        "embedding": embedding,
        "valid_coords": valid_coords,
        "n_valid_pixels": n_valid,
    }


# ---------------------------------------------------------------------------
# Representative pixel selection
# ---------------------------------------------------------------------------

def _representative_pixels(cube, labels, metadata):
    """Return one (x, y) representative per non-empty cluster.

    The representative is the valid pixel whose PCA embedding is nearest
    the cluster centroid (Euclidean distance in the PCA space).

    Parameters
    ----------
    cube : ndarray [Y, X, N]   (not used directly; kept for API symmetry)
    labels : ndarray [Y, X]
    metadata : dict   as returned by _cluster_spectra

    Returns
    -------
    representatives : list of (cluster_id, (x, y)) tuples, one per non-empty
        cluster, in ascending cluster-id order.
    """
    embedding = metadata["embedding"]     # [n_valid, n_comp]
    valid_coords = metadata["valid_coords"]  # [(y, x), ...]

    unique_clusters = sorted(
        int(c) for c in np.unique(labels) if c >= 0
    )
    representatives = []
    for k in unique_clusters:
        member_row_idxs = [
            row_idx for row_idx, (j, i) in enumerate(valid_coords)
            if labels[j, i] == k
        ]
        if not member_row_idxs:
            continue
        cluster_emb = embedding[member_row_idxs]
        centroid = cluster_emb.mean(axis=0)
        dists = np.sum((cluster_emb - centroid) ** 2, axis=1)
        best = member_row_idxs[int(np.argmin(dists))]
        j, i = valid_coords[best]
        representatives.append((k, (i, j)))  # (cluster_id, (x, y))

    return representatives


# ---------------------------------------------------------------------------
# Cluster fit schedule
# ---------------------------------------------------------------------------

def _build_cluster_schedule(labels, representatives):
    """Build the ordered per-pixel fit schedule for cluster-seeded fitting.

    Parameters
    ----------
    labels : ndarray [Y, X]
        Cluster assignment; -1 for invalid pixels.
    representatives : list of (cluster_id, (x, y))
        As returned by ``_representative_pixels``.

    Returns
    -------
    schedule : list of dict
        Each entry: ``{'cluster': int, 'seed': (x, y), 'members': [(x, y), ...]}``.
        Members are ordered row-major (y outer, x inner); seed is excluded.
        Clusters with no assigned pixels are omitted.
    """
    Y, X = labels.shape
    schedule = []

    for cluster_id, seed in representatives:
        # All valid pixels in this cluster, row-major order
        cluster_pixels = [
            (i, j) for j in range(Y) for i in range(X)
            if labels[j, i] == cluster_id
        ]
        if not cluster_pixels:
            continue
        members = [px for px in cluster_pixels if px != seed]
        schedule.append({"cluster": cluster_id, "seed": seed, "members": members})

    return schedule
