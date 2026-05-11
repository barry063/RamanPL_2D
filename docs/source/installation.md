# Installation

## Requirements

- Python 3.9 or later
- No mandatory compiled dependencies — the core package runs on NumPy and SciPy

## Recommended install (editable)

Clone the repository and install in editable mode from the `src/` directory:

```bash
git clone https://github.com/barry063/RamanPL_2D.git
cd RamanPL_2D
pip install -e .
```

An editable install means changes to the source are reflected immediately without reinstalling.

## Optional RamanSPy backend

RamanSPy is an optional preprocessing backend for supported Raman workflows. It is not needed for PL workflows or native preprocessing.

```bash
pip install -e ".[ramanspy]"
```

If RamanSPy is not installed, `preprocessing_backend="auto"` silently falls back to native.

## Optional ML extra

```bash
pip install -e ".[ml]"
```

The base install does not require scikit-learn.
Importing `ramanpl.ml` and `ramanpl.ml.clustering` is safe in a base install;
the `ImportError` is raised only when a public function (`pca_reduce`,
`kmeans_cluster`) is called without scikit-learn present.

## Building the documentation locally

```bash
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser.
