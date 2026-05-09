# Unsupervised clustering on feature tables

**Interpretability note.** `ramanpl.ml.clustering` operates on fitted peak
descriptors (positions, widths, intensities, ratios) produced by
`feature_table()` — not on raw spectra.  It is a tool for exploratory domain
discovery only.  No claim of automatic material identification is made.

## Installation

The clustering utilities require scikit-learn, which is not part of the base
install.  Add the `[ml]` extra:

```bash
pip install RamanPL_2D[ml]
```

## Usage

### PCA reduction

`pca_reduce` adds principal-component columns (`pc1`, `pc2`, …) to a copy of
the feature table.  Rows with NaN features (failed-fit pixels) are dropped
from the fit and re-injected as NaN.

```python
from ramanpl.ml import pca_reduce

df_pca = pca_reduce(df, n_components=2)
print(df_pca.attrs["explained_variance_ratio_"])  # e.g. [0.61, 0.24]
```

`feature_columns=None` (the default) auto-selects all numeric columns that
are not coordinate or QA columns (`x`, `y`, `rmse`, `ok`, `n_starts`,
`n_params_at_bounds`).

### K-means clustering

`kmeans_cluster` adds an integer `cluster` column (nullable `Int64`).

```python
from ramanpl.ml import kmeans_cluster

df_clustered = kmeans_cluster(df, n_clusters=3, random_state=42)
```

### Chaining PCA → k-means

Pass `feature_columns` explicitly to cluster on the PC subspace:

```python
from ramanpl.ml import kmeans_cluster, pca_reduce

df_pca = pca_reduce(df, n_components=2)
df_final = kmeans_cluster(
    df_pca, n_clusters=3, feature_columns=["pc1", "pc2"], random_state=42
)
```

## Explained-variance accessor

After `pca_reduce`, the fraction of total variance captured by each component
is stored in the DataFrame's `attrs` dictionary:

```python
evr = df_pca.attrs["explained_variance_ratio_"]  # list[float], length = n_components
```

## API reference

See {mod}`ramanpl.ml.clustering` in the API reference for full parameter
documentation.
