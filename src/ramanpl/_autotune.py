"""
_autotune.py — Baseline auto-tuning helper for RamanPL_2D v0.6.2.

Opt-in diagnostic that scores a grid of candidate baseline configurations
on a representative spectrum (seed pixel for mapping, single spectrum for
single-fit), returns a ranked result, and lets the caller commit the winner
via apply_choice().

Default baseline grid (24 candidates):

| Method  | Parameter sweep                                  | Candidates |
|---------|--------------------------------------------------|------------|
| asls    | lam ∈ {1e3,1e4,1e5,1e6,1e7}, p=0.001, niter=20 | 5          |
| arpls   | lam ∈ {1e3,1e4,1e5,1e6,1e7}, niter=50           | 5          |
| airpls  | lam ∈ {1e3,1e4,1e5,1e6}, niter=50               | 4          |
| poly    | poly_order ∈ {1,2,3,4,5}                         | 5          |
| gaussian| gaussian_sigma ∈ {5,10,20,50,100}               | 5          |
| Total   |                                                  | 24         |
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BaselineCandidate:
    """One entry in the autotune grid."""
    method: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def as_spec(self) -> Dict[str, Any]:
        return {"method": self.method, **self.kwargs}


@dataclass
class BaselineAutotuneResult:
    """Container returned by autotune_baseline_for_object / autotune_baseline()."""
    ranking: List[Dict[str, Any]]   # sorted ascending by rmse; each dict has method, kwargs, rmse
    winner: Dict[str, Any]          # baseline spec dict ready for apply_choice()
    seed_coord: Optional[Tuple[int, int]]
    figure: Any                     # matplotlib.Figure or None
    meta: Dict[str, Any]


def _swap_baseline_step_in_pipeline(pipeline, new_baseline_spec: Dict[str, Any]):
    """
    Return a new Pipeline with the unique BaselineSubtract step replaced.

    Raises ValueError if there are zero or more than one BaselineSubtract steps.
    """
    try:
        from .preprocessing import Pipeline, BaselineSubtract
        from .schema import normalise_baseline_spec
    except Exception:
        from ramanpl.preprocessing import Pipeline, BaselineSubtract
        from ramanpl.schema import normalise_baseline_spec

    baseline_indices = [
        i for i, s in enumerate(pipeline.steps)
        if isinstance(s, BaselineSubtract)
    ]

    if len(baseline_indices) == 0:
        raise ValueError(
            "apply_choice requires exactly one BaselineSubtract step in the pipeline, "
            "but found none. Check that background_remove=True when constructing the object."
        )
    if len(baseline_indices) > 1:
        raise ValueError(
            f"apply_choice requires exactly one BaselineSubtract step in the pipeline, "
            f"but found {len(baseline_indices)}."
        )

    idx = baseline_indices[0]
    old_step = pipeline.steps[idx]
    new_step = BaselineSubtract(
        baseline_spec=normalise_baseline_spec(new_baseline_spec),
        clip_nonnegative=old_step.clip_nonnegative,
    )

    new_steps = list(pipeline.steps)
    new_steps[idx] = new_step

    return Pipeline(steps=new_steps, backend=pipeline.backend, name=pipeline.name)


def _score_candidate(
    x: np.ndarray,
    y_raw: np.ndarray,
    candidate_pipeline,
    lb: np.ndarray,
    ub: np.ndarray,
    model_fn,
    n_starts: int = 1,
    modality: str = "Raman",
    axis_kind: str = "raman_shift_cm-1",
) -> float:
    """
    Apply candidate_pipeline to (x, y_raw), fit with model_fn, return RMSE.

    Returns np.inf on any failure.
    """
    try:
        from .preprocessing import SpectralDataset
        from .single_fit._single_fit_core import run_multistart_curve_fit
    except Exception:
        from ramanpl.preprocessing import SpectralDataset
        from ramanpl.single_fit._single_fit_core import run_multistart_curve_fit

    try:
        ds0 = SpectralDataset(
            x=np.asarray(x, dtype=float).ravel(),
            y=np.asarray(y_raw, dtype=float).ravel(),
            modality=modality,
            axis_kind=axis_kind,
            meta={},
        )
        ds = candidate_pipeline.apply(ds0)
        x_proc = np.asarray(ds.x, dtype=float).ravel()
        y_proc = np.asarray(ds.y, dtype=float).ravel()

        scale = float(np.nanmax(y_proc))
        if not np.isfinite(scale) or scale <= 0:
            return np.inf

        y_norm = y_proc / scale

        base_p0 = 0.5 * (np.asarray(lb, dtype=float) + np.asarray(ub, dtype=float))
        _, _, diag = run_multistart_curve_fit(
            model=model_fn,
            x=x_proc,
            y=y_norm,
            lower_bound=lb,
            upper_bound=ub,
            base_p0=base_p0,
            n_starts=n_starts,
            p0_strategy="midpoint",
            maxfev=6400,
            diagnose_bounds=False,
        )
        return float(diag["rmse"])
    except Exception:
        return np.inf


def _make_comparison_figure(
    ranking: List[Dict[str, Any]],
    x: np.ndarray,
    y_raw: np.ndarray,
    axis_label: str = "x",
    top_k: int = 5,
):
    """
    Overlay preprocessed spectra for the top-K candidates on a single axes.

    Returns a matplotlib Figure, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        from .preprocessing import SpectralDataset, Pipeline, BaselineSubtract
        from .schema import normalise_baseline_spec
    except Exception:
        try:
            import matplotlib.pyplot as plt
            from ramanpl.preprocessing import SpectralDataset, Pipeline, BaselineSubtract
            from ramanpl.schema import normalise_baseline_spec
        except Exception:
            return None

    x = np.asarray(x, dtype=float).ravel()
    y_raw = np.asarray(y_raw, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(8, 5))

    modality = "Raman" if "cm" in axis_label.lower() else "PL"
    axis_kind = "raman_shift_cm-1" if modality == "Raman" else "energy_eV"

    shown = 0
    for entry in ranking[:top_k]:
        spec = {"method": entry["method"], **entry["kwargs"]}
        try:
            pipe = Pipeline(
                steps=[BaselineSubtract(baseline_spec=normalise_baseline_spec(spec))],
                backend="native",
                name="autotune_preview",
            )
            ds0 = SpectralDataset(x=x, y=y_raw, modality=modality, axis_kind=axis_kind, meta={})
            ds = pipe.apply(ds0)
            y_proc = np.asarray(ds.y, dtype=float).ravel()
            x_proc = np.asarray(ds.x, dtype=float).ravel()

            kwargs_str = ", ".join(f"{k}={v}" for k, v in entry["kwargs"].items())
            label = f"{entry['method']}({kwargs_str}) RMSE={entry['rmse']:.4f}"
            ax.plot(x_proc, y_proc, label=label, alpha=0.8)
            shown += 1
        except Exception:
            continue

    ax.set_xlabel(axis_label)
    ax.set_ylabel("Processed intensity")
    ax.set_title(f"Baseline autotune — top-{shown} candidates")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def _default_baseline_grid(
    methods: Optional[List[str]] = None,
    lam_grid: Optional[List[float]] = None,
) -> List[BaselineCandidate]:
    """
    Build the default 24-candidate baseline grid.

    Parameters
    ----------
    methods : list of str or None
        Subset of {'asls', 'arpls', 'airpls', 'poly', 'gaussian'}.
        None → all five methods.
    lam_grid : list of float or None
        Override the default lam sweep for iterative methods.
        None → use default {1e3,1e4,1e5,1e6,1e7} (asls/arpls) or {1e3,1e4,1e5,1e6} (airpls).
    """
    all_methods = {"asls", "arpls", "airpls", "poly", "gaussian"}
    if methods is None:
        methods = list(all_methods)
    else:
        unknown = set(methods) - all_methods
        if unknown:
            raise ValueError(f"Unknown baseline method(s): {unknown}. Choose from {all_methods}.")

    default_lam = lam_grid if lam_grid is not None else [1e3, 1e4, 1e5, 1e6, 1e7]
    airpls_lam = lam_grid if lam_grid is not None else [1e3, 1e4, 1e5, 1e6]

    candidates: List[BaselineCandidate] = []

    if "asls" in methods:
        for lam in default_lam:
            candidates.append(BaselineCandidate("asls", {"lam": lam, "p": 0.001, "niter": 20}))

    if "arpls" in methods:
        for lam in default_lam:
            candidates.append(BaselineCandidate("arpls", {"lam": lam, "niter": 50}))

    if "airpls" in methods:
        for lam in airpls_lam:
            candidates.append(BaselineCandidate("airpls", {"lam": lam, "niter": 50}))

    if "poly" in methods:
        for order in [1, 2, 3, 4, 5]:
            candidates.append(BaselineCandidate("poly", {"poly_order": order}))

    if "gaussian" in methods:
        for sigma in [5, 10, 20, 50, 100]:
            candidates.append(BaselineCandidate("gaussian", {"gaussian_sigma": sigma}))

    return candidates


def autotune_baseline_for_object(
    obj,
    *,
    seed_coord: Optional[Tuple[int, int]] = None,
    methods: Optional[List[str]] = None,
    lam_grid: Optional[List[float]] = None,
    plot: bool = True,
    fit_spectrum_kwargs: Optional[Dict[str, Any]] = None,
) -> BaselineAutotuneResult:
    """
    Score a grid of baseline candidates on a representative spectrum.

    This function is pure-read: it never modifies obj. Call obj.apply_choice()
    to commit the winner.

    Parameters
    ----------
    obj : RamanMapping | PLMapping | RamanFit | PLfit
        The fit object.  Must have a preprocessing pipeline with exactly one
        BaselineSubtract step.
    seed_coord : (j, i) or None
        Row-major pixel index for mapping objects (j=row, i=col).
        Ignored for single-fit objects.
    methods : list[str] or None
        Subset of baseline methods to scan. None → all five (24 candidates).
    lam_grid : list[float] or None
        Override lam sweep for iterative methods.
    plot : bool
        If True, build a comparison figure for the top-5 candidates.
    fit_spectrum_kwargs : dict or None
        Extra kwargs passed to run_multistart_curve_fit (e.g. n_starts).

    Returns
    -------
    BaselineAutotuneResult
    """
    try:
        from .preprocessing import Pipeline, BaselineSubtract
        from .peak_models import sum_peaks
    except Exception:
        from ramanpl.preprocessing import Pipeline, BaselineSubtract
        from ramanpl.peak_models import sum_peaks

    fskw = fit_spectrum_kwargs or {}
    n_starts = int(fskw.get("n_starts", 1))

    # ------------------------------------------------------------------
    # Dispatch: detect object type and extract seed spectrum + axis
    # ------------------------------------------------------------------
    is_mapping = hasattr(obj, "spectra")

    if is_mapping:
        # Mapping object
        x_attr = "xdata" if hasattr(obj, "xdata") else "wavenumber"
        x = np.asarray(getattr(obj, x_attr), dtype=float).ravel()

        if seed_coord is None:
            seed_coord = (0, 0)
        seed_j, seed_i = int(seed_coord[0]), int(seed_coord[1])
        y_raw = np.asarray(obj.spectra[seed_j, seed_i, :], dtype=float).ravel()

        # Build flat bounds from custom_peaks dict
        lb_list: List[float] = []
        ub_list: List[float] = []
        for name in obj.peak_params:
            lb_pk, ub_pk = obj.custom_peaks[name]
            lb_list.extend(lb_pk)
            ub_list.extend(ub_pk)

        modality = "PL" if x_attr == "xdata" else "Raman"
        axis_kind = "energy_eV" if modality == "PL" else "raman_shift_cm-1"
        axis_label = "Energy (eV)" if modality == "PL" else "Raman shift (cm⁻¹)"

    else:
        # Single-fit object (RamanFit or PLfit)
        seed_coord = None
        # x_attr drives modality detection below; always set it.
        x_attr = "energy" if hasattr(obj, "energy") else "wavenumber"

        # Use the pristine (pre-pipeline) arrays so x and y_raw are always aligned,
        # even when the pipeline contains a CropByRange step.
        if hasattr(obj, "_x_axis_pristine"):
            x = np.asarray(obj._x_axis_pristine, dtype=float).ravel()
        else:
            x = np.asarray(getattr(obj, x_attr), dtype=float).ravel()

        if hasattr(obj, "_raw_spectra_pristine"):
            y_raw = np.asarray(obj._raw_spectra_pristine, dtype=float).ravel()
        else:
            y_raw = np.asarray(obj.raw_spectra, dtype=float).ravel()

        lb_list = list(obj.lower_bound)
        ub_list = list(obj.upper_bound)

        modality = "PL" if x_attr == "energy" else "Raman"
        axis_kind = "energy_eV" if modality == "PL" else "raman_shift_cm-1"
        axis_label = "Energy (eV)" if modality == "PL" else "Raman shift (cm⁻¹)"

    lb = np.asarray(lb_list, dtype=float)
    ub = np.asarray(ub_list, dtype=float)
    peak_profile = getattr(obj, "peak_profile", "lorentzian")
    _stride = 4 if peak_profile == "pvoigt" else 3

    def model_fn(x_in, *params):
        return sum_peaks(x_in, np.asarray(params), profile=peak_profile, stride=_stride)

    # ------------------------------------------------------------------
    # Build candidate grid and score each candidate
    # ------------------------------------------------------------------

    # Pre-flight: reject ambiguous pipelines before scoring begins.
    # >1 BaselineSubtract steps: scoring would use a wrong pipeline and
    # apply_choice would fail anyway — raise immediately.
    # 0 BaselineSubtract steps: allowed; each candidate is scored as a
    # standalone single-step pipeline (useful when exploring baselines
    # before one has been added to the pipeline).
    try:
        from .preprocessing import BaselineSubtract as _BS
    except Exception:
        from ramanpl.preprocessing import BaselineSubtract as _BS
    _n_baseline_steps = sum(1 for s in obj.preprocessing.steps if isinstance(s, _BS))
    if _n_baseline_steps > 1:
        raise ValueError(
            f"autotune_baseline requires at most one BaselineSubtract step in the "
            f"pipeline, but found {_n_baseline_steps}. Remove duplicate baseline steps "
            f"before calling autotune_baseline."
        )
    _has_baseline_step = _n_baseline_steps == 1

    candidates = _default_baseline_grid(methods=methods, lam_grid=lam_grid)

    ranking_raw: List[Dict[str, Any]] = []

    for cand in candidates:
        # Build a pipeline that only swaps the baseline step, or a standalone
        # single-step pipeline when the object has no baseline step.
        if _has_baseline_step:
            new_pipe = _swap_baseline_step_in_pipeline(obj.preprocessing, cand.as_spec())
        else:
            from ramanpl.preprocessing import Pipeline, BaselineSubtract
            from ramanpl.schema import normalise_baseline_spec
            new_pipe = Pipeline(
                steps=[BaselineSubtract(baseline_spec=normalise_baseline_spec(cand.as_spec()))],
                backend="native",
                name="autotune_standalone",
            )

        score = _score_candidate(
            x=x,
            y_raw=y_raw,
            candidate_pipeline=new_pipe,
            lb=lb,
            ub=ub,
            model_fn=model_fn,
            n_starts=n_starts,
            modality=modality,
            axis_kind=axis_kind,
        )

        ranking_raw.append({
            "method": cand.method,
            "kwargs": dict(cand.kwargs),
            "rmse": score,
        })

    # Sort ascending by RMSE; keep inf entries at the end
    ranking = sorted(ranking_raw, key=lambda r: (not np.isfinite(r["rmse"]), r["rmse"]))

    winner_entry = ranking[0]
    winner = {"method": winner_entry["method"], **winner_entry["kwargs"]}

    # ------------------------------------------------------------------
    # Optional comparison figure
    # ------------------------------------------------------------------
    fig = None
    if plot:
        fig = _make_comparison_figure(
            ranking=ranking,
            x=x,
            y_raw=y_raw,
            axis_label=axis_label,
        )

    meta = {
        "n_candidates": len(candidates),
        "methods_scanned": methods if methods is not None else ["asls", "arpls", "airpls", "poly", "gaussian"],
        "seed_coord": list(seed_coord) if seed_coord is not None else None,
    }

    return BaselineAutotuneResult(
        ranking=ranking,
        winner=winner,
        seed_coord=seed_coord,
        figure=fig,
        meta=meta,
    )
