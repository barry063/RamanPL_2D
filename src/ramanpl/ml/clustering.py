"""
ramanpl.ml.clustering — PCA and k-means clustering on feature tables.

Operates on fitted peak descriptors produced by ``feature_table()`` (v0.5.2),
not on raw spectra.  These tools are for exploratory domain discovery only;
no claim of automatic material identification is made.

Both public functions require scikit-learn (``pip install RamanPL_2D[ml]``).
Importing this module without scikit-learn installed is safe; the
``ImportError`` is raised only when a public function is called.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_QA_COLUMNS = {"x", "y", "rmse", "ok", "n_starts", "n_params_at_bounds"}


def _require_sklearn():
    """Lazily import sklearn symbols; raise a clean error when absent."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        return StandardScaler, PCA, KMeans
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for ramanpl.ml. "
            "Install with: pip install RamanPL_2D[ml]"
        ) from exc


def _select_feature_columns(df: pd.DataFrame, feature_columns) -> list:
    """Resolve the list of feature columns to use for fitting.

    Parameters
    ----------
    df : DataFrame
    feature_columns : list[str] or None
        If None, auto-selects all numeric columns not in ``_QA_COLUMNS``.

    Returns
    -------
    list[str]
    """
    if feature_columns is None:
        cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in _QA_COLUMNS
        ]
        if not cols:
            raise ValueError("No numeric feature columns found in DataFrame.")
        return cols

    cols = list(feature_columns)
    for name in cols:
        if name not in df.columns:
            raise ValueError(f"Feature column not found in DataFrame: {name!r}")
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(
                f"Feature column {name!r} is not numeric (dtype={df[name].dtype})."
            )
    return cols


def pca_reduce(
    df: pd.DataFrame,
    n_components: int,
    *,
    feature_columns=None,
    scale: bool = True,
) -> pd.DataFrame:
    """Reduce a feature table to its principal components.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide feature table, e.g. from ``RamanMapping.feature_table()``.
    n_components : int
        Number of principal components to compute (>= 1).
    feature_columns : list[str] or None, optional
        Columns to use as features.  ``None`` (default) auto-selects all
        numeric columns that are not coordinate or QA columns
        (``x``, ``y``, ``rmse``, ``ok``, ``n_starts``, ``n_params_at_bounds``).
        To chain after :func:`pca_reduce`, pass ``["pc1", "pc2"]`` explicitly.
    scale : bool, optional
        If ``True`` (default), standardise features to zero mean and unit
        variance before fitting.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with ``pc1`` … ``pc{n_components}`` columns appended.
        Rows whose feature values contain NaN are dropped from the fit and
        re-injected as NaN in the PC columns.  The returned index always
        matches the input index.

    Attributes
    ----------
    out.attrs["explained_variance_ratio_"] : list[float]
        Fraction of variance explained by each component (length *n_components*).
    out.attrs["pca_feature_columns"] : list[str]
        Feature columns used for fitting.
    """
    StandardScaler, PCA, _ = _require_sklearn()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(n_components, int) or n_components < 1:
        raise ValueError(f"n_components must be an integer >= 1, got {n_components!r}.")

    cols = _select_feature_columns(df, feature_columns)

    if n_components > len(cols):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed number of feature "
            f"columns ({len(cols)})."
        )

    X = df[cols].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)
    kept_idx = df.index[mask]

    if mask.sum() == 0:
        raise ValueError(
            "No rows remain after dropping NaN features; cannot fit PCA."
        )

    Xk = X[mask]
    if scale:
        Xk = StandardScaler().fit_transform(Xk)

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(Xk)

    out = df.copy()
    for k in range(n_components):
        out[f"pc{k + 1}"] = np.nan
        out.loc[kept_idx, f"pc{k + 1}"] = Z[:, k]

    out.attrs["explained_variance_ratio_"] = pca.explained_variance_ratio_.tolist()
    out.attrs["pca_feature_columns"] = list(cols)

    return out


def kmeans_cluster(
    df: pd.DataFrame,
    n_clusters: int,
    *,
    feature_columns=None,
    scale: bool = True,
    random_state=None,
) -> pd.DataFrame:
    """Assign each pixel in a feature table to a k-means cluster.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide feature table, e.g. from ``RamanMapping.feature_table()``.
    n_clusters : int
        Number of clusters (>= 1).
    feature_columns : list[str] or None, optional
        Columns to use as features.  ``None`` (default) auto-selects all
        numeric columns that are not coordinate or QA columns.
    scale : bool, optional
        If ``True`` (default), standardise features before clustering.
    random_state : int or None, optional
        Passed to ``sklearn.cluster.KMeans`` for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with a ``cluster`` column (nullable ``Int64``) appended.
        Rows whose feature values contain NaN receive ``pd.NA``.  The returned
        index always matches the input index.
    """
    StandardScaler, _, KMeans = _require_sklearn()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError(f"n_clusters must be an integer >= 1, got {n_clusters!r}.")
    if random_state is not None and not isinstance(random_state, int):
        raise TypeError(
            f"random_state must be None or an int, got {type(random_state).__name__}."
        )

    cols = _select_feature_columns(df, feature_columns)

    X = df[cols].to_numpy(dtype=float)
    mask = np.isfinite(X).all(axis=1)
    kept_idx = df.index[mask]

    if mask.sum() == 0:
        raise ValueError(
            "No rows remain after dropping NaN features; cannot fit k-means."
        )

    Xk = X[mask]
    if scale:
        Xk = StandardScaler().fit_transform(Xk)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xk)

    out = df.copy()
    out["cluster"] = pd.array([pd.NA] * len(df), dtype="Int64")
    out.loc[kept_idx, "cluster"] = labels

    return out
