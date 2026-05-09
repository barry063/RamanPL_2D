"""
test_ml_clustering.py
---------------------
Unit tests for ramanpl.ml.clustering.

All tests use small synthetic DataFrames; no real spectra are loaded.
The entire module is skipped when scikit-learn is not installed.
"""

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn")

from ramanpl.ml.clustering import kmeans_cluster, pca_reduce  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_df(n_rows=20, seed=0):
    """Return a small numeric DataFrame without QA columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feat_a": rng.normal(size=n_rows),
            "feat_b": rng.normal(size=n_rows),
            "feat_c": rng.normal(size=n_rows),
        }
    )


def _df_with_qa(n_rows=20, seed=0):
    """DataFrame that includes coordinate and QA columns to be auto-excluded."""
    df = _simple_df(n_rows, seed)
    df["x"] = np.arange(n_rows, dtype=float)
    df["y"] = np.zeros(n_rows)
    df["rmse"] = np.ones(n_rows) * 0.01
    df["ok"] = True
    df["n_starts"] = 1
    df["n_params_at_bounds"] = 0
    return df


def _df_with_nan_row(n_rows=10, nan_row=0, seed=0):
    """DataFrame with one row where all features are NaN."""
    df = _simple_df(n_rows, seed)
    df.loc[nan_row, ["feat_a", "feat_b", "feat_c"]] = np.nan
    return df


# ---------------------------------------------------------------------------
# Tests for pca_reduce
# ---------------------------------------------------------------------------

def test_pca_reduce_returns_pc_columns_and_preserves_index():
    df = _simple_df()
    out = pca_reduce(df, n_components=2)
    assert "pc1" in out.columns
    assert "pc2" in out.columns
    assert "pc3" not in out.columns
    assert list(out.index) == list(df.index)


def test_pca_reduce_attaches_explained_variance():
    df = _simple_df()
    out = pca_reduce(df, n_components=2)
    evr = out.attrs["explained_variance_ratio_"]
    assert isinstance(evr, list)
    assert len(evr) == 2
    assert all(0.0 <= v <= 1.0 for v in evr)
    assert sum(evr) <= 1.0 + 1e-9
    assert evr[0] >= evr[1]


def test_pca_reduce_drops_nan_rows_and_reinjects_nan():
    df = _df_with_nan_row(nan_row=3)
    out = pca_reduce(df, n_components=2)
    # NaN row should have NaN PC values
    assert np.isnan(out.loc[3, "pc1"])
    assert np.isnan(out.loc[3, "pc2"])
    # All other rows should be finite
    other = out.drop(index=3)
    assert np.isfinite(other["pc1"].to_numpy()).all()
    assert np.isfinite(other["pc2"].to_numpy()).all()


def test_pca_reduce_invalid_feature_column_raises():
    df = _simple_df()
    with pytest.raises(ValueError, match="not found in DataFrame"):
        pca_reduce(df, n_components=1, feature_columns=["nonexistent"])


def test_pca_reduce_n_components_too_large_raises():
    df = _simple_df()  # 3 feature columns
    with pytest.raises(ValueError, match="n_components"):
        pca_reduce(df, n_components=4)

# Added test for QA: the auto-exclusion behaviour is part of the public contract
def test_pca_reduce_auto_excludes_qa_columns():
      df = _df_with_qa()
      out = pca_reduce(df, n_components=2)
      assert out.attrs["pca_feature_columns"] == ["feat_a", "feat_b", "feat_c"]

# ---------------------------------------------------------------------------
# Tests for kmeans_cluster
# ---------------------------------------------------------------------------

def test_kmeans_cluster_returns_cluster_column_and_preserves_index():
    df = _simple_df()
    out = kmeans_cluster(df, n_clusters=3, random_state=0)
    assert "cluster" in out.columns
    assert list(out.index) == list(df.index)
    valid = out["cluster"].dropna()
    assert set(valid.astype(int).unique()).issubset({0, 1, 2})


def test_kmeans_cluster_drops_nan_rows_and_reinjects_nan():
    df = _df_with_nan_row(nan_row=5)
    out = kmeans_cluster(df, n_clusters=2, random_state=0)
    assert pd.isna(out.loc[5, "cluster"])
    other = out.drop(index=5)
    assert other["cluster"].notna().all()


def test_kmeans_cluster_recovers_known_clusters():
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(42)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    labels_true = np.repeat([0, 1, 2], 30)
    pts = centres[labels_true] + rng.normal(scale=0.3, size=(90, 2))
    df = pd.DataFrame({"x_feat": pts[:, 0], "y_feat": pts[:, 1]})

    out = kmeans_cluster(df, n_clusters=3, random_state=42)
    labels_pred = out["cluster"].astype(int).to_numpy()
    ari = adjusted_rand_score(labels_true, labels_pred)
    assert ari >= 0.99


def test_kmeans_cluster_random_state_is_deterministic():
    from sklearn.metrics import adjusted_rand_score

    df = _simple_df(n_rows=60, seed=7)
    out1 = kmeans_cluster(df, n_clusters=3, random_state=99)
    out2 = kmeans_cluster(df, n_clusters=3, random_state=99)
    ari = adjusted_rand_score(
        out1["cluster"].astype(int).to_numpy(),
        out2["cluster"].astype(int).to_numpy(),
    )
    assert ari == 1.0


def test_kmeans_cluster_invalid_feature_column_raises():
    df = _simple_df()
    with pytest.raises(ValueError, match="not found in DataFrame"):
        kmeans_cluster(df, n_clusters=2, feature_columns=["missing_col"])


# ---------------------------------------------------------------------------
# Composition test
# ---------------------------------------------------------------------------

def test_pca_then_kmeans_chain_works_on_explicit_pc_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(size=30),
            "feat_b": rng.normal(size=30),
            "feat_c": rng.normal(size=30),
        }
    )
    df_pca = pca_reduce(df, n_components=2)
    assert "pc1" in df_pca.columns
    assert "pc2" in df_pca.columns

    df_final = kmeans_cluster(
        df_pca, n_clusters=3, feature_columns=["pc1", "pc2"], random_state=0
    )
    assert "pc1" in df_final.columns
    assert "pc2" in df_final.columns
    assert "cluster" in df_final.columns
    assert list(df_final.index) == list(df.index)


# ---------------------------------------------------------------------------
# Boundary tests (both functions)
# ---------------------------------------------------------------------------

def test_no_numeric_features_raises():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "ok": [True, False]})
    with pytest.raises(ValueError, match="No numeric feature columns"):
        pca_reduce(df, n_components=1)
    with pytest.raises(ValueError, match="No numeric feature columns"):
        kmeans_cluster(df, n_clusters=2)


def test_empty_after_nan_drop_raises():
    df = pd.DataFrame(
        {
            "feat_a": [np.nan, np.nan, np.nan],
            "feat_b": [np.nan, np.nan, np.nan],
        }
    )
    with pytest.raises(ValueError, match="No rows remain"):
        pca_reduce(df, n_components=1)
    with pytest.raises(ValueError, match="No rows remain"):
        kmeans_cluster(df, n_clusters=2)