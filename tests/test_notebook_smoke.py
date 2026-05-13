"""
test_notebook_smoke.py
----------------------
Smoke-execute the three canonical backend-behaviour notebooks.

Each notebook is executed headlessly with a bounded timeout. Execution
errors in any cell cause the test to fail. Output content is not asserted.

Requires: nbformat, nbconvert (both ship with a standard Jupyter install).
    pip install nbformat nbconvert

These tests are intentionally skipped when nbconvert is not available so
that the base pytest suite can still run without a full Jupyter environment.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_NOTEBOOKS = [
    REPO_ROOT / "example-usage" / "Ramanfit" / "Raman_backend_demo.ipynb",
    REPO_ROOT / "example-usage" / "Mapping" / "Raman_mapping_backend_demo.ipynb",
    REPO_ROOT / "example-usage" / "backend" / "Backend_fallback_cases.ipynb",
    REPO_ROOT / "example-usage" / "Mapping" / "Feature_Table_Example.ipynb",
    REPO_ROOT / "example-usage" / "Mapping" / "Peak_Proposal_Demo.ipynb",
    REPO_ROOT / "example-usage" / "Validation" / "Validation_v0.6.0_vs_v0.5.0.ipynb",
]

# Notebooks excluded from standard CI smoke: runtime exceeds 600 s/cell on typical hardware.
# Run with: pytest -m slow
_SLOW_NOTEBOOKS = [
    REPO_ROOT / "example-usage" / "Mapping" / "Clustering_Demo.ipynb",
]

# Notebooks that require the [ml] extra (scikit-learn)
_ML_NOTEBOOKS = {"Clustering_Demo.ipynb"}

# Seconds allowed per cell before the test fails with a timeout error.
# Set to 600 to accommodate notebooks that run real map fits on WDF datasets.
NOTEBOOK_TIMEOUT = 600

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    _NBCONVERT_AVAILABLE = True
except ImportError:
    _NBCONVERT_AVAILABLE = False


def _execute_notebook(notebook_path: Path, timeout: int = NOTEBOOK_TIMEOUT) -> None:
    """Execute a notebook in-process and raise on any cell execution error."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

    nb = nbformat.read(notebook_path, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})


@pytest.mark.skipif(not _NBCONVERT_AVAILABLE, reason="nbformat/nbconvert not installed")
@pytest.mark.parametrize("notebook", CANONICAL_NOTEBOOKS, ids=[p.name for p in CANONICAL_NOTEBOOKS])
def test_notebook_executes_without_error(notebook):
    if notebook.name in _ML_NOTEBOOKS and importlib.util.find_spec("sklearn") is None:
        pytest.skip(f"{notebook.name} requires [ml] extra (scikit-learn not installed)")
    assert notebook.exists(), f"Canonical notebook not found: {notebook}"
    try:
        _execute_notebook(notebook)
    except Exception as exc:
        pytest.fail(f"Notebook execution failed — {notebook.name}: {exc}")


@pytest.mark.slow
@pytest.mark.skipif(not _NBCONVERT_AVAILABLE, reason="nbformat/nbconvert not installed")
@pytest.mark.parametrize("notebook", _SLOW_NOTEBOOKS, ids=[p.name for p in _SLOW_NOTEBOOKS])
def test_slow_notebook_executes_without_error(notebook):
    if notebook.name in _ML_NOTEBOOKS and importlib.util.find_spec("sklearn") is None:
        pytest.skip(f"{notebook.name} requires [ml] extra (scikit-learn not installed)")
    assert notebook.exists(), f"Slow notebook not found: {notebook}"
    try:
        _execute_notebook(notebook)
    except Exception as exc:
        pytest.fail(f"Notebook execution failed — {notebook.name}: {exc}")
