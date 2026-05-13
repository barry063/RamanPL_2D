"""
benchmark_mapping_fit.py
-------------------------
Reproducible benchmark harness for v0.5.1.

Records a hardware-independent "before" performance baseline for the
RamanMapping fit pipeline across a grid of dataset sizes and fit-kwarg
variants.  Intended to be re-run on the same machine after algorithmic
changes to assess improvement.

Design choices
--------------
- Synthetic cubes use deterministic peak parameters with additive Gaussian
  noise (numpy.random.default_rng(42)).
- Wall-clock runtime is **advisory** because scipy.optimize.curve_fit
  iteration count, warm-start state propagation, and adaptive-multistart
  retries all introduce structural variability that is not noise but
  algorithmic state.  A 1.0x-1.2x spread between runs on identical inputs
  is expected and does not constitute a regression or improvement.
- The primary metric is n_curve_fit_calls, a hardware-independent count of
  fit attempts.  Algorithmic improvements should reduce this; pure runtime
  gains without a corresponding reduction are likely measurement artefacts.
- Future builds claiming a speed-up must show a reduction in
  n_curve_fit_calls, not only in runtime_s.

Usage
-----
    python benchmarks/benchmark_mapping_fit.py

Output
------
    benchmarks/results/mapping_fit_benchmark.csv
"""

import sys
import csv
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Make the src package importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import scipy.optimize as _scipy_opt
from ramanpl.mapping import RamanMapping

# ---------------------------------------------------------------------------
# Result record field names (order preserved in CSV)
# ---------------------------------------------------------------------------

_RESULT_FIELDS = [
    "dataset_name",
    "cube_shape",
    "warm_start",
    "n_starts",
    "random_state",
    "runtime_s",
    "n_curve_fit_calls",
    "success_rate",
    "mean_rmse_finite",
    "n_failed_pixels",
]

_DEFAULT_OUTPUT = Path(__file__).parent / "results" / "mapping_fit_benchmark.csv"

_CUSTOM_PEAKS = {
    "Si": ([490.0, 1.0, 0.001], [560.0, 40.0, 5.0]),
}
_DATA_RANGE = (200.0, 1800.0)
_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Dataset and fit-kwargs builders
# ---------------------------------------------------------------------------

def build_mapping_fit_benchmark_cases():
    """Return list of (name, x, cube) tuples for three synthetic Raman mapping datasets."""
    rng = np.random.default_rng(_RANDOM_STATE)

    cases = []
    for name, n_pts, shape in [
        ("small_3x4",       150, (3,  4,  150)),
        ("extended_15x15",  150, (15, 15, 150)),
        ("tall_axis_10x12", 500, (10, 12, 500)),
    ]:
        x = np.linspace(200.0, 1800.0, n_pts)
        peak = np.exp(-0.5 * ((x - 520.0) / 15.0) ** 2)
        cube = (peak[None, None, :] + rng.uniform(0.0, 0.05, shape)).astype(float)
        cases.append((name, x, cube))
    return cases


def build_fit_kwargs_variants():
    """Return list of fit-kwargs dicts covering the 2x2 warm_start x n_starts grid."""
    variants = []
    for warm_start in (False, True):
        for n_starts in (1, 4):
            variants.append({
                "warm_start": warm_start,
                "fit_spectrum_kwargs": {
                    "diagnostics": "light",
                    "random_state": _RANDOM_STATE,
                    "n_starts": n_starts,
                },
            })
    return variants


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_mapping_fit_case(*, x, cube, fit_kwargs, dataset_name) -> dict:
    """
    Run one (dataset, fit_kwargs) benchmark case.

    Uses unittest.mock.patch on scipy.optimize.curve_fit inside a with
    block to count calls without modifying production code.  The patch is
    local to this function and never applied at module level.

    Parameters
    ----------
    x : ndarray
        Spectral axis.
    cube : ndarray
        Mapping cube with shape [Y, X, N].
    fit_kwargs : dict
        Keys: warm_start (bool), fit_spectrum_kwargs (dict).
    dataset_name : str
        Name tag for the result record.

    Returns
    -------
    dict
        Record with all _RESULT_FIELDS keys.
    """
    Y, X, _ = cube.shape
    warm_start = fit_kwargs.get("warm_start", False)
    fsw = dict(fit_kwargs.get("fit_spectrum_kwargs", {}))
    n_starts_val = fsw.get("n_starts", 1)

    _real_curve_fit = _scipy_opt.curve_fit

    counter = {"n": 0}

    def _wrapped_curve_fit(*args, **kwargs):
        counter["n"] += 1
        return _real_curve_fit(*args, **kwargs)

    def _build():
        return RamanMapping.from_arrays(
            cube, x, X, Y,
            custom_peaks=_CUSTOM_PEAKS,
            data_range=_DATA_RANGE,
            background_remove=False,
            smoothing=False,
            normalize=False,
        )

    with patch.object(_scipy_opt, "curve_fit", _wrapped_curve_fit):
        t0 = time.perf_counter()
        m = _build()
        m.fit_spectra(warm_start=warm_start, fit_spectrum_kwargs=fsw)
        runtime_s = time.perf_counter() - t0
        n_curve_fit_calls = counter["n"]

    # Compute summary statistics from fit results
    residuals = m.residual_map  # shape [Y, X]
    n_total = Y * X
    finite_mask = np.isfinite(residuals)
    n_ok = int(np.sum(finite_mask))
    n_failed = n_total - n_ok
    success_rate = n_ok / n_total if n_total > 0 else float("nan")
    mean_rmse_finite = float(np.mean(residuals[finite_mask])) if n_ok > 0 else float("nan")

    return {
        "dataset_name":       dataset_name,
        "cube_shape":         str(cube.shape),
        "warm_start":         warm_start,
        "n_starts":           n_starts_val,
        "random_state":       _RANDOM_STATE,
        "runtime_s":          round(runtime_s, 6),
        "n_curve_fit_calls":  int(n_curve_fit_calls),
        "success_rate":       round(success_rate, 6),
        "mean_rmse_finite":   round(mean_rmse_finite, 8),
        "n_failed_pixels":    n_failed,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_mapping_fit_benchmark_csv(results, output_path=_DEFAULT_OUTPUT):
    """Write benchmark results to CSV.  Creates parent directories if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def _print_summary(results):
    header = (
        f"{'dataset':<20} {'warm':>5} {'nst':>4} {'time(s)':>9} "
        f"{'calls':>7} {'ok%':>7} {'rmse':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in results:
        print(
            f"{row['dataset_name']:<20} {str(row['warm_start']):>5} "
            f"{row['n_starts']:>4} {row['runtime_s']:>9.4f} "
            f"{row['n_curve_fit_calls']:>7} "
            f"{row['success_rate'] * 100:>6.1f}% "
            f"{row['mean_rmse_finite']:>10.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all(output_path=_DEFAULT_OUTPUT):
    """Run the full benchmark grid and write results to CSV."""
    cases = build_mapping_fit_benchmark_cases()
    variants = build_fit_kwargs_variants()

    all_rows = []
    for dataset_name, x, cube in cases:
        for fit_kwargs in variants:
            try:
                row = run_mapping_fit_case(
                    x=x, cube=cube, fit_kwargs=fit_kwargs, dataset_name=dataset_name
                )
                all_rows.append(row)
            except Exception as exc:
                warnings.warn(
                    f"Benchmark failed for {dataset_name} / warm_start="
                    f"{fit_kwargs['warm_start']}: {exc}",
                    stacklevel=1,
                )

    write_mapping_fit_benchmark_csv(all_rows, output_path)
    _print_summary(all_rows)
    print(f"\nResults written to: {output_path}")
    return all_rows


if __name__ == "__main__":
    run_all()
