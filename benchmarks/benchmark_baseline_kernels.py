"""
benchmark_baseline_kernels.py
------------------------------
Microbenchmark for native 1D baseline routines.

Sweeps spectrum length and reports per-method runtime and peak memory.
Scientific correctness is NOT checked here — see test_baseline_numerical_parity.py.

Usage
-----
    python benchmarks/benchmark_baseline_kernels.py
"""

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.baselineAPI import BaselineAPI

# ---------------------------------------------------------------------------
# Synthetic spectrum builder
# ---------------------------------------------------------------------------

def _make_spectrum(n: int, seed: int = 0) -> tuple:
    """Return (x, y) — a synthetic Raman-like 1D spectrum of length n."""
    rng = np.random.default_rng(seed)
    x = np.linspace(200.0, 1800.0, n)
    baseline = 0.5 + 0.3 * np.sin(np.pi * (x - 200.0) / 1600.0)
    peak = np.exp(-0.5 * ((x - 520.0) / 15.0) ** 2)
    noise = rng.uniform(0.0, 0.02, n)
    return x, baseline + peak + noise


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _measure(func, *args, **kwargs) -> tuple[float, float]:
    """Return (runtime_s, peak_memory_mb) for func(*args, **kwargs)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    func(*args, **kwargs)
    runtime_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return runtime_s, peak_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

_METHODS = [
    ("poly",    {"method": "poly",    "poly_degree": 3}),
    ("gaussian",{"method": "gaussian","gaussian_sigma": 10}),
    ("asls",    {"method": "asls",    "lam": 1e5, "p": 0.01, "niter": 20}),
    ("arpls",   {"method": "arpls",   "lam": 1e5, "niter": 50}),
    ("airpls",  {"method": "airpls",  "lam": 1e5, "niter": 50}),
]

_SPECTRUM_LENGTHS = [128, 512, 1024]


def run_baseline_kernel_benchmarks() -> list[dict]:
    rows = []
    for n in _SPECTRUM_LENGTHS:
        x, y = _make_spectrum(n)
        for method_name, kwargs in _METHODS:
            method = kwargs.pop("method")
            runtime_s, peak_mb = _measure(
                BaselineAPI.subtract, x, y, method=method, **kwargs
            )
            kwargs["method"] = method  # restore for re-use
            rows.append({
                "spectrum_len": n,
                "method":       method_name,
                "runtime_s":    round(runtime_s, 6),
                "peak_mb":      round(peak_mb, 4),
            })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    header = f"{'n':>6}  {'method':<10}  {'time(s)':>10}  {'mem(MB)':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print(
            f"{row['spectrum_len']:>6}  {row['method']:<10}  "
            f"{row['runtime_s']:>10.6f}  {row['peak_mb']:>9.4f}"
        )
    print(sep)
    print("Note: scientific parity is not measured here.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = run_baseline_kernel_benchmarks()
    _print_table(rows)
