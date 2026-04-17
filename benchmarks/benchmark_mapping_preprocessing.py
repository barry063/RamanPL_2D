"""
benchmark_mapping_preprocessing.py
-----------------------------------
Reproducible benchmark harness for v0.4.3.

Compares native vs RamanSPy preprocessing backends on synthetic Raman
mapping cubes across six pipeline configurations. Records runtime,
peak memory (tracemalloc), cube shape, axis integrity, and numeric
parity between backends.

Usage
-----
    python benchmarks/benchmark_mapping_preprocessing.py

Output
------
    benchmarks/results/mapping_preprocess_benchmark.csv
"""

import sys
import time
import tracemalloc
import csv
import warnings
from pathlib import Path

import numpy as np

# Make the src package importable when run as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ramanpl.preprocessing import (
    apply_pipeline_to_mapping_cube,
    Pipeline,
    make_benchmark_pipeline_crop_only,
    make_benchmark_pipeline_savgol_only,
    make_benchmark_pipeline_poly_baseline,
    make_benchmark_pipeline_asls_baseline,
    make_benchmark_pipeline_airpls_baseline,
    make_benchmark_pipeline_arpls_baseline,
    pipeline_to_benchmark_name,
)
from ramanpl.integration.ramanspy_adapter import has_ramanspy

# ---------------------------------------------------------------------------
# Result record field names (order preserved in CSV)
# ---------------------------------------------------------------------------
_RESULT_FIELDS = [
    "dataset_name",
    "pipeline_name",
    "backend_requested",
    "backend_resolved",
    "runtime_s",
    "peak_memory_mb",
    "cube_shape",
    "axis_len",
    "axis_match",
    "rmse",
    "max_abs_err",
]

_DEFAULT_OUTPUT = Path(__file__).parent / "results" / "mapping_preprocess_benchmark.csv"


# ---------------------------------------------------------------------------
# Dataset and pipeline builders
# ---------------------------------------------------------------------------

def build_mapping_benchmark_cases():
    """Return a list of (name, x, cube) tuples for five synthetic Raman mapping datasets."""
    rng = np.random.default_rng(0)

    cases = []
    for name, n_pts, shape in [
        ("small_3x4",      150, (3,  4,  150)),
        ("medium_10x12",   150, (10, 12, 150)),
        ("larger_20x24",   150, (20, 24, 150)),
        ("extended_30x30", 150, (30, 30, 150)),
        ("tall_axis_10x12", 500, (10, 12, 500)),
    ]:
        x = np.linspace(200.0, 1800.0, n_pts)
        peak = np.exp(-0.5 * ((x - 520.0) / 15.0) ** 2)
        cube = (peak[None, None, :] + rng.uniform(0.0, 0.05, shape)).astype(float)
        cases.append((name, x, cube))
    return cases


def build_supported_mapping_pipelines():
    """Return the six benchmark pipelines (all native backend; caller overrides for ramanspy)."""
    return [
        make_benchmark_pipeline_crop_only(),
        make_benchmark_pipeline_savgol_only(),
        make_benchmark_pipeline_poly_baseline(),
        make_benchmark_pipeline_asls_baseline(),
        make_benchmark_pipeline_airpls_baseline(),
        make_benchmark_pipeline_arpls_baseline(),
    ]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_runtime_seconds(func, *args, **kwargs) -> float:
    """Return wall-clock seconds taken by func(*args, **kwargs)."""
    t0 = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - t0


def measure_peak_memory_mb(func, *args, **kwargs) -> float:
    """Return peak tracemalloc memory in MB incurred by func(*args, **kwargs)."""
    tracemalloc.start()
    try:
        func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 ** 2)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_mapping_preprocess_case(*, x, cube, pipeline, backend, dataset_name) -> dict:
    """
    Run one (dataset, pipeline, backend) benchmark case.

    Returns a dict with 11 keys matching _RESULT_FIELDS.  The ``rmse`` and
    ``max_abs_err`` fields are NaN by default; the caller fills them in via
    ``summarise_mapping_parity`` after comparing native vs ramanspy results.
    """
    p = Pipeline(steps=pipeline.steps, backend=backend, name=pipeline.name)

    def _runner():
        return apply_pipeline_to_mapping_cube(
            x=x,
            cube=cube,
            pipeline=p,
            modality="Raman",
            axis_kind="raman_shift_cm-1",
        )

    # Run once to capture result.
    result = _runner()

    # Measure runtime and memory on separate runs (avoids tracemalloc/timing
    # interaction and caching effects that could skew the first call).
    runtime_s = measure_runtime_seconds(_runner)
    peak_memory_mb = measure_peak_memory_mb(_runner)

    backend_resolved = result.meta.get("preprocessing_backend", "unknown")
    axis_match = bool(result.cube.shape[2] == len(result.x))

    return {
        "dataset_name":     dataset_name,
        "pipeline_name":    pipeline_to_benchmark_name(pipeline),
        "backend_requested": backend,
        "backend_resolved":  backend_resolved,
        "runtime_s":         round(runtime_s, 6),
        "peak_memory_mb":    round(peak_memory_mb, 4),
        "cube_shape":        str(result.cube.shape),
        "axis_len":          len(result.x),
        "axis_match":        axis_match,
        "rmse":              float("nan"),
        "max_abs_err":       float("nan"),
        # Store the result object for parity comparison; removed before CSV write.
        "_result_obj":       result,
    }


def summarise_mapping_parity(native_result, ramanspy_result) -> dict:
    """Return RMSE and max absolute error between two MappingPreprocessResult cubes."""
    if native_result.cube.shape != ramanspy_result.cube.shape:
        return {"rmse": float("nan"), "max_abs_err": float("nan")}
    diff = native_result.cube - ramanspy_result.cube
    return {
        "rmse":        float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_err": float(np.max(np.abs(diff))),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_mapping_benchmark_csv(results, output_path=_DEFAULT_OUTPUT):
    """Write benchmark results to CSV.  Creates parent directories if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip internal helper key before writing.
    clean_results = [{k: v for k, v in row.items() if k != "_result_obj"} for row in results]

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(clean_results)


def _print_summary(results):
    """Print a compact aligned summary table to stdout."""
    header = (
        f"{'dataset':<16} {'pipeline':<28} {'backend':>10} {'time(s)':>9} "
        f"{'mem(MB)':>9} {'rmse':>12}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in results:
        rmse_str = f"{row['rmse']:.2e}" if not (
            isinstance(row["rmse"], float) and row["rmse"] != row["rmse"]
        ) else "  n/a"
        print(
            f"{row['dataset_name']:<16} {row['pipeline_name']:<28} "
            f"{row['backend_resolved']:>10} {row['runtime_s']:>9.4f} "
            f"{row['peak_memory_mb']:>9.3f} {rmse_str:>12}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    cases = build_mapping_benchmark_cases()
    pipelines = build_supported_mapping_pipelines()
    backends = ["native", "ramanspy"] if has_ramanspy() else ["native"]

    if not has_ramanspy():
        warnings.warn(
            "RamanSPy is not installed — skipping ramanspy backend rows. "
            "Install with: pip install ramanspy",
            stacklevel=1,
        )

    all_rows = []

    for dataset_name, x, cube in cases:
        for pipeline in pipelines:
            native_row = None
            ramanspy_row = None

            # --- native ---
            try:
                native_row = run_mapping_preprocess_case(
                    x=x, cube=cube, pipeline=pipeline,
                    backend="native", dataset_name=dataset_name,
                )
                all_rows.append(native_row)
            except Exception as exc:
                warnings.warn(
                    f"Native benchmark failed for {dataset_name}/{pipeline.name}: {exc}",
                    stacklevel=1,
                )

            # --- ramanspy ---
            if has_ramanspy():
                try:
                    ramanspy_row = run_mapping_preprocess_case(
                        x=x, cube=cube, pipeline=pipeline,
                        backend="ramanspy", dataset_name=dataset_name,
                    )
                    # Patch parity metrics if native result is available.
                    if native_row is not None:
                        parity = summarise_mapping_parity(
                            native_row["_result_obj"], ramanspy_row["_result_obj"]
                        )
                        ramanspy_row.update(parity)
                    all_rows.append(ramanspy_row)
                except (ImportError, NotImplementedError, Exception) as exc:
                    warnings.warn(
                        f"RamanSPy benchmark failed for {dataset_name}/{pipeline.name}: {exc}",
                        stacklevel=1,
                    )
                    # Record a skipped row so the CSV is complete.
                    skipped = {
                        "dataset_name":      dataset_name,
                        "pipeline_name":     pipeline_to_benchmark_name(pipeline),
                        "backend_requested": "ramanspy",
                        "backend_resolved":  "skipped",
                        "runtime_s":         float("nan"),
                        "peak_memory_mb":    float("nan"),
                        "cube_shape":        "n/a",
                        "axis_len":          "n/a",
                        "axis_match":        "n/a",
                        "rmse":              float("nan"),
                        "max_abs_err":       float("nan"),
                        "_result_obj":       None,
                    }
                    all_rows.append(skipped)

    _print_summary(all_rows)
    write_mapping_benchmark_csv(all_rows)
    print(f"\nResults written to: {_DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
