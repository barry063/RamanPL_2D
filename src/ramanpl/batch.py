"""
Batch utilities for loading and plotting multiple Raman/PL spectra.

Design goals
------------
- Minimal disruption: operates on the existing Spectrum container.
- Consistent axis inference and x-grid alignment with operation.ArithmeticSpectrum.
- Focused on comparative visualisation: overlay and waterfall plots.

Public API
----------
- load_spectra(...)
- align_spectra(...)
- plot_overlay(...)
- plot_waterfall(...)
- Batch.RamanBatch / Batch.PLBatch convenience wrappers (fit, export, summary_by_peak)
  including peak control via custom_peaks (replaces defaults) and remove_peaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np

from ramanpl.dataImporter import DataImporter
from ramanpl.operation import Spectrum, AxisType, AlignType, ArithmeticSpectrum
from ramanpl.exporter import write_table


LabelType = Union[str, Callable[[str], str]]
NormType = Literal["none", "max", "area", "peak"]
YSelectType = Literal["raw", "fit", "raw+fit"]


def load_spectra(
    files: Sequence[Union[str, Path]],
    *,
    axis: AxisType = "auto",
    x_range: Optional[Tuple[float, float]] = None,
    readlines=None,
    txt_delimiter: str = "\t",
    txt_skiprows: int = 1,
    label_from: Optional[LabelType] = None,
) -> List[Spectrum]:
    """
    Load multiple single-spectrum files into a list of Spectrum objects.
    """
    out: List[Spectrum] = []
    for f in files:
        p = str(f)
        y, x = DataImporter.data_import(
            filename=p,
            readlines=readlines,
            x_range=x_range,
            axis="auto",
            txt_delimiter=txt_delimiter,
            txt_skiprows=txt_skiprows,
        )
        y = np.asarray(y, dtype=float).ravel()
        x = np.asarray(x, dtype=float).ravel()

        # Reuse your existing axis inference (same behaviour as ArithmeticSpectrum.combine)
        ax = ArithmeticSpectrum._resolve_axis(x, axis=axis)

        if label_from is None:
            label = p
        elif isinstance(label_from, str):
            label = label_from
        else:
            label = str(label_from(p))

        out.append(Spectrum(y=y, x=x, axis=ax, source=label))
    return out


def align_spectra(
    spectra: Sequence[Spectrum],
    *,
    align: AlignType = "interp",
    reference: int = 0,
) -> List[Spectrum]:
    """
    Align a list of spectra to a common x-grid.

    Strategy
    --------
    - reference spectrum defines x-grid (default index 0)
    - align='interp': interpolate others onto reference.x
    - align='strict': require identical x arrays for all spectra
    """
    spectra = list(spectra)
    if len(spectra) == 0:
        return []

    if not (0 <= reference < len(spectra)):
        raise IndexError(f"reference index {reference} out of range for {len(spectra)} spectra")

    ref = spectra[reference]
    x_ref = np.asarray(ref.x, dtype=float).ravel()

    out: List[Spectrum] = []
    for i, s in enumerate(spectra):
        if s.axis != ref.axis:
            raise ValueError(
                f"Axis mismatch in batch: reference is '{ref.axis}' but item {i} is '{s.axis}'."
            )

        x = np.asarray(s.x, dtype=float).ravel()
        y = np.asarray(s.y, dtype=float).ravel()

        if align == "strict":
            if x.size != x_ref.size or not np.allclose(x, x_ref, rtol=0, atol=0):
                raise ValueError("align='strict' requires identical x arrays for all spectra.")
            y_aligned = y
        elif align == "interp":
            if i == reference:
                y_aligned = y
            else:
                y_aligned = ArithmeticSpectrum._interp_to(x_src=x, y_src=y, x_tgt=x_ref)
        else:
            raise ValueError("align must be 'interp' or 'strict'.")

        out.append(Spectrum(y=y_aligned, x=x_ref, axis=ref.axis, source=s.source))

    return out


def _normalise(y: np.ndarray, mode: NormType) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if mode == "none":
        return y

    if mode == "max":
        denom = float(np.nanmax(np.abs(y)))
        return y / denom if denom > 0 else y

    if mode == "area":
        area = float(np.trapz(np.abs(y)))
        return y / area if area > 0 else y

    if mode == "peak":
        denom = float(np.nanmax(y) - np.nanmin(y))
        return y / denom if denom > 0 else y

    raise ValueError("normalise must be one of: 'none', 'max', 'area', 'peak'.")


def plot_overlay(
    spectra: Sequence[Spectrum],
    *,
    fit_spectra: Optional[Sequence[Spectrum]] = None,
    align: AlignType = "interp",
    reference: int = 0,
    normalise: NormType = "none",
    y_select: YSelectType = "raw",
    xlim: Optional[Tuple[float, float]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    fit_linestyle: str = "--",
    fit_linewidth: float = 1.5,
):
    """
    Overlay plot for multiple spectra.

    Parameters
    ----------
    spectra : Sequence[Spectrum]
        Raw spectra.
    fit_spectra : Optional[Sequence[Spectrum]]
        Fitted spectra (same length and corresponding to `spectra`).
    y_select : {"raw", "fit", "raw+fit"}
        What to plot.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    if y_select in ("fit", "raw+fit") and fit_spectra is None:
        raise ValueError("fit_spectra must be provided when y_select is 'fit' or 'raw+fit'.")

    aligned_raw = align_spectra(spectra, align=align, reference=reference)

    aligned_fit = None
    if fit_spectra is not None:
        aligned_fit = align_spectra(fit_spectra, align=align, reference=reference)

        if len(aligned_fit) != len(aligned_raw):
            raise ValueError("fit_spectra must have the same length as spectra (raw).")
        # sanity check: axis should match
        if any(fr.axis != rr.axis for rr, fr in zip(aligned_raw, aligned_fit)):
            raise ValueError("Axis mismatch between raw spectra and fit spectra.")

    fig, ax = plt.subplots()

    # Plot raw
    if y_select in ("raw", "raw+fit"):
        for s in aligned_raw:
            y = _normalise(s.y, normalise)
            ax.plot(s.x, y, label=s.source)

    # Plot fit (same ordering)
    if y_select in ("fit", "raw+fit"):
        assert aligned_fit is not None
        for raw_s, fit_s in zip(aligned_raw, aligned_fit):
            y = _normalise(fit_s.y, normalise)

            # Labeling strategy:
            # - If raw already has legend entries, don't duplicate labels for fit curves
            # - Otherwise, label fit curves (useful for y_select="fit")
            label = (fit_s.source if y_select == "fit" else None)

            ax.plot(
                fit_s.x,
                y,
                linestyle=fit_linestyle,
                linewidth=fit_linewidth,
                label=label,
            )

    ax.set_xlabel("Energy (eV)" if aligned_raw[0].axis == "energy" else "Wavenumber (cm$^{-1}$)")
    if normalise != "none":
        ax.set_ylabel("Normalised intensity (a.u.)")
    else:
        ax.set_ylabel("Intensity (a.u.)")

    if x_range is not None:
        xlim = x_range
    if xlim is not None:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title)

    if show_legend:
        ax.legend(frameon=False)

    return fig, ax


def plot_waterfall(
    spectra: Sequence[Spectrum],
    *,
    fit_spectra: Optional[Sequence[Spectrum]] = None,
    align: AlignType = "interp",
    reference: int = 0,
    normalise: NormType = "none",
    offset: float = 1.0,
    y_select: YSelectType = "raw",
    xlim: Optional[Tuple[float, float]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    show_labels: bool = True,
    fit_linestyle: str = "--",
    fit_linewidth: float = 1.5,
):
    """
    Waterfall plot (stacked curves with a vertical offset).
    """
    import matplotlib.pyplot as plt

    if y_select in ("fit", "raw+fit") and fit_spectra is None:
        raise ValueError("fit_spectra must be provided when y_select is 'fit' or 'raw+fit'.")

    aligned_raw = align_spectra(spectra, align=align, reference=reference)

    aligned_fit = None
    if fit_spectra is not None:
        aligned_fit = align_spectra(fit_spectra, align=align, reference=reference)
        if len(aligned_fit) != len(aligned_raw):
            raise ValueError("fit_spectra must have the same length as spectra (raw).")
        if any(fr.axis != rr.axis for rr, fr in zip(aligned_raw, aligned_fit)):
            raise ValueError("Axis mismatch between raw spectra and fit spectra.")

    fig, ax = plt.subplots()

    for i, raw_s in enumerate(aligned_raw):
        y0 = i * offset

        if y_select in ("raw", "raw+fit"):
            y = _normalise(raw_s.y, normalise) + y0
            ax.plot(raw_s.x, y)
            if show_labels:
                # Place label slightly inside the left edge of the visible x-range.
                if xlim is not None:
                    x_text = float(xlim[0]) + 0.01 * (float(xlim[1]) - float(xlim[0]))
                else:
                    x_text = float(raw_s.x[0]) + 0.01 * (float(raw_s.x[-1]) - float(raw_s.x[0]))

                ax.text(
                    x_text,
                    float(y0),               # anchor at the baseline offset line
                    str(raw_s.source),
                    va="center",
                    ha="left",
                    fontsize=8,
                    clip_on=True,            # guarantees it cannot draw outside the axes
                )

        if y_select in ("fit", "raw+fit"):
            assert aligned_fit is not None
            fit_s = aligned_fit[i]
            y = _normalise(fit_s.y, normalise) + y0
            ax.plot(fit_s.x, y, linestyle=fit_linestyle, linewidth=fit_linewidth)

    ax.set_xlabel("Energy (eV)" if aligned_raw[0].axis == "energy" else "Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (a.u.) + offset")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", which="both", left=False)
    
    if x_range is not None:
        xlim = x_range
    if xlim is not None:
        ax.set_xlim(*xlim)
    if title:
        ax.set_title(title)

    return fig, ax

def _resolve_fitter_class(fitter_cls):
    for attr in ("RamanFit", "PLfit"):
        if hasattr(fitter_cls, attr):
            return getattr(fitter_cls, attr)
    return fitter_cls

def _instantiate_fitter(fitter_class, *, spectra, x, axis: str, fitter_kwargs: Optional[dict] = None):
    """
    Instantiate fitter_class robustly by selecting the correct x keyword
    based on fitter __init__ signature.

    axis: "wavenumber" or "energy"
    """
    import inspect

    fitter_kwargs = fitter_kwargs or {}
    sig = inspect.signature(fitter_class.__init__)
    params = sig.parameters

    kw = dict(fitter_kwargs)
    kw["spectra"] = spectra

    if axis == "wavenumber":
        if "wavenumber" not in params:
            raise TypeError(f"{fitter_class.__name__}.__init__ does not accept wavenumber=.")
        kw["wavenumber"] = x

    elif axis == "energy":
        # Prefer energy=, but allow future alternate names if PLfit changes later
        if "energy" in params:
            kw["energy"] = x
        elif "wavelength" in params:
            kw["wavelength"] = x
        else:
            raise TypeError(f"{fitter_class.__name__}.__init__ does not accept energy=.")
    else:
        raise ValueError(f"Unknown axis '{axis}'.")

    return fitter_class(**kw)

def fit_spectra_batch(
    spectra: Sequence[Spectrum],
    *,
    fitter_cls,
    fitter_kwargs: Optional[dict] = None,
    fit_spectrum_kwargs: Optional[dict] = None,
    return_fitters: bool = False,
):
    """
    Fit multiple spectra using PLfit or RamanFit.

    Parameters
    ----------
    spectra : Sequence[Spectrum]
        Batch of raw spectra to fit.
    fitter_cls
        Either the fitter class itself, or the imported module containing it.
    fitter_kwargs : dict, optional
        Keyword arguments passed directly to the fitter constructor.
        This includes preprocessing-related kwargs such as:
            - preprocessing
            - baseline_method
            - smoothing
            - peak_profile
            - custom_peaks
    fit_spectrum_kwargs : dict, optional
        Keyword arguments passed directly to fitter.fit_spectrum(...).
    return_fitters : bool, optional
        If True, also return the fitted fitter objects.

    Returns
    -------
    list
        - if return_fitters=False: list[(raw_spectrum, fitted_spectrum)]
        - if return_fitters=True : list[(raw_spectrum, fitted_spectrum, fitter_object)]
    """
    fitter_kwargs = fitter_kwargs or {}
    fit_spectrum_kwargs = fit_spectrum_kwargs or {}

    # Accept either a module (ramanpl.RamanFit) or the class (RamanFit.RamanFit)
    fitter_class = _resolve_fitter_class(fitter_cls)


    out = []

    for s in spectra:
        # Instantiate robustly by introspecting fitter signature
        fitter = _instantiate_fitter(
            fitter_class,
            spectra=s.y,
            x=s.x,
            axis=s.axis,
            fitter_kwargs=fitter_kwargs,
        )

        # Your fitters use fit_spectrum()
        fitter.fit_spectrum(**fit_spectrum_kwargs)

        # Build fitted spectrum FIRST
        x_fit, y_fit = fitter.get_fitted_spectrum()
        fit_spec = Spectrum(
            y=y_fit,
            x=x_fit,
            axis=s.axis,
            source=f"{s.source} (fit)",
        )

        # Append ONCE (after fit_spec exists)
        if return_fitters:
            out.append((s, fit_spec, fitter))
        else:
            out.append((s, fit_spec))

    return out


def collect_fit_parameters(
    fits: Sequence[tuple[Spectrum, Spectrum]],
    *,
    fitter_cls,
    fitter_kwargs: Optional[dict] = None,
):
    """
    Collect fitted parameters from a batch of spectra.

    Returns
    -------
    list[dict]
        Each dict: source, peak, position, fwhm, intensity
    """
    fitter_kwargs = dict(fitter_kwargs or {})

    # Accept either module (ramanpl.RamanFit / ramanpl.PLfit) or class
    if hasattr(fitter_cls, "RamanFit"):
        fitter_class = fitter_cls.RamanFit
        fitter_kind = "raman"
    elif hasattr(fitter_cls, "PLfit"):
        fitter_class = fitter_cls.PLfit
        fitter_kind = "pl"
    else:
        fitter_class = _resolve_fitter_class(fitter_cls)
        name = getattr(fitter_class, "__name__", "").lower()
        fitter_kind = "raman" if "raman" in name else ("pl" if "pl" in name else "unknown")

    rows = []

    for raw, _ in fits:
        # ---- axis/fitter compatibility + keyword dispatch ----
        if fitter_kind == "raman":
            if raw.axis != "wavenumber":
                raise ValueError(f"Raman fitter requires axis='wavenumber', got '{raw.axis}'.")
            fitter = fitter_class(
                spectra=raw.y,
                wavenumber=raw.x,
                **fitter_kwargs,
            )
        elif fitter_kind == "pl":
            if raw.axis != "energy":
                raise ValueError(f"PL fitter requires axis='energy', got '{raw.axis}'.")
            fitter = fitter_class(
                spectra=raw.y,
                energy=raw.x,
                **fitter_kwargs,
            )
        else:
            raise ValueError(
                "Could not infer fitter type. Pass ramanpl.RamanFit (module) or ramanpl.PLfit (module), "
                "or use a fitter class whose name includes 'Raman' or 'PL'."
            )

        fitter.fit_spectrum()
        params = fitter.get_fitted_parameters()

        for peak, d in params.items():
            rows.append(
                dict(
                    source=raw.source,
                    peak=peak,
                    position=d["position"],
                    fwhm=d["fwhm"],
                    peak_height=d["peak_height"],  # NEW: physical meaning (peak max)
                )
            )


    return rows


def collect_fit_parameters_from_fitters(
    fits_with_fitters: Sequence[tuple[Spectrum, Spectrum, object]],
):
    """
    Collect fitted parameters without re-fitting.

    Parameters
    ----------
    fits_with_fitters : list[(raw_spectrum, fitted_spectrum, fitter)]
        Output of fit_spectra_batch(..., return_fitters=True)

    Returns
    -------
    list[dict]
        Long-format rows: source, peak, position, fwhm, intensity
    """
    rows = []

    for raw, _fit, fitter in fits_with_fitters:
        params = fitter.get_fitted_parameters()
        for peak, d in params.items():
            if "peak_height" in d:
                peak_height = d["peak_height"]
            elif "intensity" in d:
                # legacy fallback (optional — you can delete this branch if you want it strict)
                peak_height = d["intensity"]
            else:
                raise KeyError(
                    f"Fitter returned params for peak '{peak}' without 'peak_height'. "
                    f"Available keys: {list(d.keys())}"
                )

            rows.append(
                dict(
                    source=raw.source,
                    peak=peak,
                    position=d["position"],
                    fwhm=d["fwhm"],
                    peak_height=peak_height,
                )
            )

    return rows


def _infer_metadata_from_fitters(
    fits_with_fitters: Sequence[tuple[Spectrum, Spectrum, object]],
) -> Dict[str, Any]:
    """
    Infer a metadata dict from fitter objects without re-fitting.
    """
    if not fits_with_fitters:
        return {}

    fitters = []
    for item in fits_with_fitters:
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            fitters.append(item[2])

    if not fitters:
        return {}

    # Keys mirror RamanFit/PLfit export metadata where possible.
    key_map: Dict[str, str] = {
        "spectrum_type": "spectrum_type",
        "x_quantity": "x_quantity",
        "x_unit": "x_unit",
        "materials": "materials",
        "substrate": "substrate",

        # legacy / behaviour-preserving fitter flags
        "background_remove": "background_remove",
        "baseline_method": "baseline_method",
        "poly_degree": "poly_degree",
        "gaussian_sigma": "gaussian_sigma",
        "smoothing": "smoothing",
        "smooth_window": "smooth_window",
        "smooth_order": "smooth_order",
        "normalize": "normalize",

        # peak-model metadata
        "peak_labels": "peak_labels",
        "peak_intensity": "peak_intensity",
        "peak_profile": "peak_profile",
        "params_per_peak": "params_per_peak",

        # preprocessing metadata (v0.3.4+)
        "preprocessing": "preprocessing",
        "preprocessing_recipe": "preprocessing_recipe",
    }

    def _norm_value(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()

        # Serialise preprocessing pipelines / similar objects cleanly into metadata
        if hasattr(v, "to_dict") and callable(getattr(v, "to_dict")):
            try:
                return v.to_dict()
            except Exception:
                return str(v)

        # Safeguard common containers
        if isinstance(v, tuple):
            return list(v)

        return v

    def _eq(a: Any, b: Any) -> bool:
        try:
            return a == b
        except Exception:
            return False

    def _merge(values: List[Any]) -> Any:
        cleaned = [_norm_value(v) for v in values if v is not None]
        if not cleaned:
            return None
        first = cleaned[0]
        if all(_eq(first, v) for v in cleaned[1:]):
            return first
        unique: List[Any] = []
        for v in cleaned:
            if not any(_eq(v, u) for u in unique):
                unique.append(v)
        return unique

    meta: Dict[str, Any] = {}

    for out_key, attr in key_map.items():
        vals = [getattr(f, attr, None) for f in fitters]
        merged = _merge(vals)
        if merged is not None:
            meta[out_key] = merged

    # Custom peaks flag (True/False/mixed)
    if any(hasattr(f, "custom_peaks") for f in fitters):
        vals = ["True" if getattr(f, "custom_peaks", None) is not None else "False" for f in fitters]
        merged = _merge(vals)
        if merged is not None:
            meta["custom_peaks"] = merged
    
    # Removed peaks (list / mixed)
    if any(hasattr(f, "remove_peaks_list") for f in fitters):
        vals = [getattr(f, "remove_peaks_list", None) for f in fitters]
        merged = _merge(vals)
        if merged is not None:
            meta["remove_peaks"] = merged

    # Fitter identity
    fitter_names = [type(f).__name__ for f in fitters]
    meta["fitter_class"] = _merge(fitter_names) or fitter_names[0]
    meta["n_spectra"] = len(fitters)

    return meta


def parameters_dataframe(rows):
    """
    Convert list-of-dicts rows (source, peak, position, fwhm, peak_height) into a DataFrame.
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    required = {"source", "peak", "position", "fwhm", "peak_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in rows: {sorted(missing)}")
    return df


def plot_fitted_parameters(
    df,
    *,
    metric: str = "position",
    peaks=None,
    x_from_source=None,
    sort_x: bool = True,
    title: str | None = None,
    ylabel: str | None = None,
    show_legend: bool = True,
):
    """
    Plot fitted parameter(s) across spectra.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format table with columns: source, peak, position, fwhm, peak_height
    metric : {"position","fwhm","peak_height"}
        Which fitted parameter to plot.
    peaks : list[str] | None
        Peak names to include. If None, include all peaks.
    x_from_source : callable | None
        Function mapping source string -> numeric x (e.g., temperature).
        If None, x-axis will be categorical (source order).
    sort_x : bool
        If x_from_source is provided, sort points by x.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if metric not in ("position", "fwhm", "peak_height"):
        raise ValueError("metric must be one of: 'position', 'fwhm', 'peak_height'.")

    dff = df.copy()

    if peaks is not None:
        dff = dff[dff["peak"].isin(list(peaks))]

    if dff.empty:
        raise ValueError("No data to plot after filtering. Check peak names and df content.")

    fig, ax = plt.subplots()  # create first

    if x_from_source is not None:
        dff["x"] = dff["source"].apply(x_from_source)
        if sort_x:
            dff = dff.sort_values("x")
        x_label = "Condition"
    else:
        sources = list(dict.fromkeys(dff["source"].tolist()))
        src_to_i = {s: i for i, s in enumerate(sources)}
        x_label = "Source"
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources, rotation=45, ha="right")

    for peak_name, sub in dff.groupby("peak", sort=False):
        if x_from_source is not None:
            xx = sub["x"].to_numpy(dtype=float)
        else:
            xx = sub["source"].map(src_to_i).to_numpy(dtype=float)

        ax.plot(
            xx,
            sub[metric].to_numpy(dtype=float),
            marker="o",
            label=str(peak_name),
        )

    ax.set_xlabel(x_label)

    if ylabel is None:
        if metric == "position":
            ylabel = "Position (cm$^{-1}$)" if "wavenumber" in str(df.get("axis", "")) else "Position (a.u.)"
        elif metric == "fwhm":
            ylabel = "FWHM (cm$^{-1}$)"
        else:
            ylabel = "Peak height (a.u.)"

    ax.set_ylabel(ylabel)

    if title is None:
        title = f"Fitted {metric} across batch"
    ax.set_title(title)

    if show_legend:
        ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax

def pivot_parameters(
    df,
    *,
    index_col: str = "source",
    peak_col: str = "peak",
    metrics: tuple[str, ...] = ("position", "fwhm", "peak_height"),
    sep: str = "_",
    aggfunc="first",
):
    """
    Pivot a long-format parameter DataFrame into wide format.

    Input long format columns (required):
        source | peak | position | fwhm | intensity

    Output wide format:
        index = source (by default)
        columns = <peak><sep><metric> e.g. E12g_position, A1g_fwhm, ...

    Parameters
    ----------
    aggfunc : str or callable
        If duplicates exist (same source+peak), how to aggregate (default 'first').
        Alternatives: 'mean', 'median', np.mean, etc.
    """
    import pandas as pd

    required = {index_col, peak_col, *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pivot_parameters: missing required columns: {sorted(missing)}")

    # Create a MultiIndex column: (peak, metric)
    wide = df.pivot_table(
        index=index_col,
        columns=peak_col,
        values=list(metrics),
        aggfunc=aggfunc,
    )

    # Flatten columns to "peak_metric" or "metric_peak" (choose one; here: peak_metric)
    # pivot_table returns columns as (metric, peak) by default when values is a list.
    # Convert to <peak>_<metric> for readability.
    wide.columns = [f"{peak}{sep}{metric}" for (metric, peak) in wide.columns]

    # Nice ordering: sort columns alphabetically (optional but helpful)
    wide = wide.reindex(sorted(wide.columns), axis=1)

    # Make index a normal column if you prefer:
    # wide = wide.reset_index()

    return wide

def export_dataframe(
    df,
    out_path: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    include_header: bool = True,
    headers: bool = True,
    meta_in_csv: bool = False,
    float_format: Optional[str] = None,
) -> str:
    """
    Export a pandas DataFrame using write_table(), with optional metadata header.

    float_format:
        If provided, applies to float values before writing (string formatting).
    """
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError("export_dataframe requires pandas to be installed.") from e

    if not hasattr(df, "to_dict"):
        raise TypeError("export_dataframe expects a pandas DataFrame.")

    # Optionally format floats
    if float_format is not None:
        def fmt(v):
            if isinstance(v, (float, int)) and v is not None:
                try:
                    return float_format % float(v)
                except Exception:
                    return v
            return v
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = df2[c].map(fmt)
        df = df2

    rows = df.reset_index().to_dict(orient="records")
    if len(rows) == 0:
        # still write headers if possible
        fieldnames = list(df.reset_index().columns)
    else:
        fieldnames = list(rows[0].keys())


    return write_table(
        rows=rows,
        out_path=out_path,
        fieldnames=fieldnames,
        include_header=include_header,
        meta=meta,
        headers=headers,
        meta_in_csv=meta_in_csv,
    )

def export_table(
    df,
    out_path: str,
    *,
    metadata: dict | None = None,
    fits_with_fitters=None,
    include_metadata: bool = True,
    meta_in_csv: bool = True,
    float_format: str | None = None,
):
    meta = {}
    if include_metadata:
        if fits_with_fitters is not None:
            meta.update(_infer_metadata_from_fitters(fits_with_fitters))
        if metadata:
            meta.update(metadata)

    return export_dataframe(
        df,
        out_path,
        meta=meta if (include_metadata and meta) else None,
        meta_in_csv=meta_in_csv,
        float_format=float_format,
    )


from dataclasses import dataclass
from typing import Any, Optional, Sequence, Callable, Literal

# ---- Convenience classes for batch workflows ----
# These classes are thin orchestration layers:
#   - reuse existing load_spectra, fit_spectra_batch, plotting, table/export helpers
#   - minimise intermediate variables in notebooks
#   - keep backward compatibility (existing functions remain usable)

AxisLiteral = Literal["wavenumber", "energy"]


@dataclass
class _BaseBatch:
    files: Sequence[str]
    axis: AxisLiteral
    fitter_module: Any  # ramanpl.RamanFit module or ramanpl.PLfit module
    fitter_kwargs: dict

    specs: Optional[list] = None                  # list[Spectrum]
    fits: Optional[list] = None                   # list[(raw, fit, fitter)] when return_fitters=True
    rows: Optional[list[dict]] = None             # long-format rows (dicts)
    _df_cache: Any = None                         # pandas DataFrame cache (optional)
    _wide_cache: Any = None                       # pandas DataFrame cache (optional)

    # reproducibility / future interop helpers
    _last_fit_spectrum_kwargs: Optional[dict] = None
    _last_fitter_kwargs: Optional[dict] = None

    def load(self):
        """Load spectra from files into self.specs."""
        self.specs = load_spectra(list(self.files), axis=self.axis)
        return self

    def fit(self, *, return_fitters: bool = True, fit_spectrum_kwargs: Optional[dict] = None):
        """
        Fit all spectra and cache:
          - self.fits (raw, fit, fitter)
          - self.rows (long-format fit parameter rows)
        """
        if self.specs is None:
            self.load()

        self.fits = fit_spectra_batch(
            self.specs,
            fitter_cls=self.fitter_module,
            fitter_kwargs=self.fitter_kwargs,
            fit_spectrum_kwargs=fit_spectrum_kwargs,
            return_fitters=return_fitters,
        )
        
        self._last_fit_spectrum_kwargs = dict(fit_spectrum_kwargs or {})
        self._last_fitter_kwargs = dict(self.fitter_kwargs or {})

        # Collect fitted parameter rows if we have fitters
        if return_fitters:
            self.rows = collect_fit_parameters_from_fitters(self.fits)
        else:
            # No fitter objects returned; cannot collect parameters without refitting.
            self.rows = None

        # Invalidate cached tables
        self._df_cache = None
        self._wide_cache = None
        return self

    # -----------------------
    # Plotting convenience
    # -----------------------
    def plot_overlay(
        self,
        *,
        normalise: NormType = "none",
        title: Optional[str] = None,
        raw_fit: bool = False,
        x_range: Optional[tuple[float, float]] = None,
        align: AlignType = "interp",
        reference: int = 0,
        show_legend: bool = True,
    ):
        """
        Overlay plot.

        raw_fit=False -> plot raw only
        raw_fit=True  -> plot raw + fit (requires .fit(return_fitters=True) called)
        """
        if self.specs is None:
            self.load()

        if not raw_fit:
            return plot_overlay(
                self.specs,
                normalise=normalise,
                title=title,
                x_range=x_range,
                align=align,
                reference=reference,
                show_legend=show_legend,
            )

        if self.fits is None:
            raise RuntimeError("No fits cached. Run .fit(return_fitters=True) first.")

        raw_specs = [r for r, f, _ in self.fits]
        fit_specs = [f for r, f, _ in self.fits]

        return plot_overlay(
            raw_specs,
            fit_spectra=fit_specs,
            y_select="raw+fit",
            normalise=normalise,
            title=title,
            x_range=x_range,
            align=align,
            reference=reference,
            show_legend=show_legend,
        )

    def plot_waterfall(
        self,
        *,
        normalise: NormType = "none",
        offset: float = 1.0,
        title: Optional[str] = None,
        raw_fit: bool = False,
        x_range: Optional[tuple[float, float]] = None,
        align: AlignType = "interp",
        reference: int = 0,
        show_labels: bool = True,
    ):
        """
        Waterfall plot.

        raw_fit=False -> plot raw only
        raw_fit=True  -> plot raw + fit (requires .fit(return_fitters=True) called)
        """
        if self.specs is None:
            self.load()

        if not raw_fit:
            return plot_waterfall(
                self.specs,
                normalise=normalise,
                offset=offset,
                title=title,
                x_range=x_range,
                align=align,
                reference=reference,
                show_labels=show_labels,
            )

        if self.fits is None:
            raise RuntimeError("No fits cached. Run .fit(return_fitters=True) first.")

        raw_specs = [r for r, f, _ in self.fits]
        fit_specs = [f for r, f, _ in self.fits]

        return plot_waterfall(
            raw_specs,
            fit_spectra=fit_specs,
            y_select="raw+fit",
            normalise=normalise,
            offset=offset,
            title=title,
            x_range=x_range,
            align=align,
            reference=reference,
            show_labels=show_labels,
        )

    # -----------------------
    # Results: rows / tables
    # -----------------------
    def to_dataframe(self):
        """
        Return a pandas DataFrame of long-format rows.
        This imports pandas internally so the user doesn't have to.
        """
        if self.rows is None:
            raise RuntimeError("No parameter rows cached. Run .fit(return_fitters=True) first.")

        if self._df_cache is not None:
            return self._df_cache

        df = parameters_dataframe(self.rows)
        self._df_cache = df
        return df

    def wide_table(
        self,
        *,
        add_condition: bool = False,
        condition_name: str = "condition",
        condition=None,  # Callable[[str], Any] | Sequence[Any] | None
        sort_by_condition: bool = False,
        ):
        """
        Return a wide-format table (pandas DataFrame):
        columns like <peak>_position, <peak>_fwhm, <peak>_intensity

        Optionally adds a user-defined condition column derived from the 'source' name.
        """
        df = self.to_dataframe()

        # build wide and cache
        if self._wide_cache is None:
            wide = pivot_parameters(df)
            # enforce first-appearance order, consistent with plot_fitted_parameters
            sources = list(dict.fromkeys(df["source"].tolist()))
            wide = wide.reindex(sources)
            self._wide_cache = wide
        else:
            wide = self._wide_cache

        if add_condition:
            if condition is None:
                raise ValueError("Provide condition (callable or sequence) when add_condition=True.")

            wide = wide.copy()
            sources = list(wide.index)

            if callable(condition):
                wide[condition_name] = [condition(s) for s in sources]
            else:
                cond_list = list(condition)
                if len(cond_list) != len(sources):
                    raise ValueError(f"Length of condition ({len(cond_list)}) must match number of sources ({len(sources)}).")
                wide[condition_name] = cond_list

            if sort_by_condition:
                wide = wide.sort_values(condition_name)

        return wide

    def summary(self, *, max_rows: int = 30):
        """
        Print a simple summary table without requiring pandas from the user.
        (If pandas is available, the user can still call .to_dataframe()).
        """
        if self.rows is None:
            raise RuntimeError("No parameter rows cached. Run .fit(return_fitters=True) first.")

        # Basic column formatting
        cols = ["source", "peak", "position", "fwhm", "peak_height"]
        print(" | ".join([f"{c:>12s}" for c in cols]))
        print("-" * (15 * len(cols)))

        for i, r in enumerate(self.rows[:max_rows]):
            print(
                f"{str(r.get('source','')):>12s} | "
                f"{str(r.get('peak','')):>12s} | "
                f"{float(r.get('position', float('nan'))):>12.6g} | "
                f"{float(r.get('fwhm', float('nan'))):>12.6g} | "
                f"{float(r.get('peak_height', float('nan'))):>12.6g}"
            )

        if len(self.rows) > max_rows:
            print(f"... ({len(self.rows) - max_rows} more rows)")

    # -----------------------
    # Export convenience
    # -----------------------
    def export(
        self,
        out_path: str,
        *,
        wide: bool = True,
        metadata: Optional[dict] = None,
        meta_in_csv: bool = True,
        float_format: Optional[str] = "%.6g",
        add_condition: bool = False,
        condition_name: str = "condition",
        condition: Optional[Callable[[str], Any]] = None,
        sort_by_condition: bool = False,
    ):
        if wide:
            table = self.wide_table(
                add_condition=add_condition,
                condition_name=condition_name,
                condition=condition,
                sort_by_condition=sort_by_condition,
            )
        else:
            table = self.to_dataframe()

        if self.fits is None:
            raise RuntimeError("No fits cached. Run .fit(return_fitters=True) first.")

        meta_user = dict(metadata or {})

        if self._last_fitter_kwargs is not None and "preprocessing" in self._last_fitter_kwargs:
            pp = self._last_fitter_kwargs["preprocessing"]
            if hasattr(pp, "to_dict") and callable(getattr(pp, "to_dict")):
                try:
                    meta_user.setdefault("preprocessing", pp.to_dict())
                except Exception:
                    meta_user.setdefault("preprocessing", str(pp))
            else:
                meta_user.setdefault("preprocessing", pp)

        if self._last_fit_spectrum_kwargs is not None:
            meta_user.setdefault("fit_spectrum_kwargs", dict(self._last_fit_spectrum_kwargs))

        return export_table(
            table,
            out_path=out_path,
            fits_with_fitters=self.fits,
            metadata=meta_user,
            meta_in_csv=meta_in_csv,
            float_format=float_format,
        )
    
    def plot_parameters(
        self,
        *,
        metric: str = "position",
        peaks=None,
        x=None,
        x_label: str = "Condition",
        sort_x: bool = True,
        title: str | None = None,
        ylabel: str | None = None,
        show_legend: bool = True,
    ):
        """
        Plot fitted parameters across the batch.

        Parameters
        ----------
        metric : {"position","fwhm","intensity"}
        peaks : list[str] | None
        x : None | Sequence[Any] | Callable[[str], Any]
            - None: x-axis is the source labels (default behaviour)
            - Sequence: user-supplied condition values aligned with batch sources
            - Callable: maps source string -> condition value
        x_label : str
            Label for x-axis when x is provided (Sequence/Callable).
        """
        df = self.to_dataframe()

        # Case A: x is None -> categorical by source (existing default)
        if x is None:
            return plot_fitted_parameters(
                df,
                metric=metric,
                peaks=peaks,
                x_from_source=None,
                sort_x=sort_x,
                title=title,
                ylabel=ylabel,
                show_legend=show_legend,
            )

        # Case B: x is a callable -> use it directly
        if callable(x):
            return plot_fitted_parameters(
                df,
                metric=metric,
                peaks=peaks,
                x_from_source=x,
                sort_x=sort_x,
                title=title,
                ylabel=ylabel,
                show_legend=show_legend,
            )

        # Case C: x is a sequence -> map sources to x values
        # We create a dict mapping source->value, then use a small wrapper
        x_list = list(x)
        sources = list(dict.fromkeys(df["source"].tolist()))
        if len(x_list) != len(sources):
            raise ValueError(
                f"Length of x ({len(x_list)}) must match number of sources ({len(sources)})."
            )
        mapping = {s: x_list[i] for i, s in enumerate(sources)}

        def _lookup(src):
            return mapping[src]

        fig, ax = plot_fitted_parameters(
            df,
            metric=metric,
            peaks=peaks,
            x_from_source=_lookup,
            sort_x=sort_x,
            title=title,
            ylabel=ylabel,
            show_legend=show_legend,
        )
        # Override the x-axis label to whatever the user wants
        ax.set_xlabel(x_label)
        return fig, ax

    def summary_by_peak(
        self,
        *,
        metrics: tuple[str, ...] = ("position", "fwhm", "peak_height"),
        percentiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
        dropna: bool = True,
        print_summary: bool = True,
        print_mode: str = "table",  # NEW: "table" or "blocks"
        interval: tuple[float, float] = (0.05, 0.95),  # NEW (for blocks)
        show_minmax_in_blocks: bool = False,           # NEW
        export_path: str | None = None,
        include_metadata: bool = True,
        meta_in_csv: bool = True,
        float_format: str | None = "%.6g",
    ):
        """
        Summarise fitted peak parameters across the batch (per-peak statistics).

        Requires:
            .fit(return_fitters=True) called first (so self.rows exists).

        Parameters
        ----------
        metrics
            Which columns to summarise (must exist in the long-format df).
            Default: ("position", "fwhm", "peak_height")
        percentiles
            Percentiles to include (0..1). Default: (5%,25%,50%,75%,95%)
        print_mode
            "table" (default) or "blocks". "blocks" shows each peak in a separate block with min/max values.
        interval
            For "blocks" mode, the interval to show in each block (e.g., (0.05, 0.95) for 5th to 95th percentiles).
        show_minmax_in_blocks
            If True, show min and max values in each block.
        dropna
            If True, ignore NaNs when computing stats (recommended).
        print_summary
            If True, print a readable summary to stdout.
        export_path
            If provided, export the summary table to .csv/.txt/.tsv.
            Uses export_table() to keep metadata behaviour consistent.
        include_metadata, meta_in_csv
            Passed through to export_table().
        float_format
            Numeric formatting for printing and export (export uses pandas float_format if supported).

        Returns
        -------
        stats_df : pandas.DataFrame
            Per-peak statistics in a wide, easy-to-read format.
        """
        import numpy as np
        import pandas as pd

        df = self.to_dataframe()

        # Validate requested metrics
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(f"summary_by_peak: metrics not found in dataframe: {missing}")

        # Preserve the "first appearance" peak order (matches your plotting/groupby(sort=False) flow)
        peak_order = list(dict.fromkeys(df["peak"].tolist()))

        # Ensure numeric types
        dff = df.copy()
        for m in metrics:
            dff[m] = pd.to_numeric(dff[m], errors="coerce")

        # Optionally drop rows where all requested metrics are NaN
        if dropna:
            dff = dff.dropna(subset=list(metrics), how="all")

        if dff.empty:
            raise ValueError("summary_by_peak: no numeric data available after filtering/NaN handling.")

        # Build per-peak stats
        groups = dff.groupby("peak", sort=False)

        # Core stats
        agg = {}
        for m in metrics:
            agg[m] = ["count", "mean", "std", "min", "median", "max"]

        core = groups.agg(agg)

        # Percentiles (quantiles)
        q = []
        for p in percentiles:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"summary_by_peak: invalid percentile {p}; must be within [0,1].")
            # groupby.quantile returns a Series with MultiIndex
            qq = groups[list(metrics)].quantile(p).copy()
            qq["quantile"] = p
            q.append(qq.reset_index())

        if q:
            qdf = pd.concat(q, ignore_index=True)
            # pivot quantiles to columns: (metric, qXX)
            qdf["q_label"] = qdf["quantile"].map(lambda x: f"q{int(round(100*x)):02d}")
            q_wide = (
                qdf.pivot_table(index="peak", columns="q_label", values=list(metrics), aggfunc="first")
                .sort_index()
            )
            # Combine core + quantiles
            stats = pd.concat([core, q_wide], axis=1)
        else:
            stats = core

        # Re-order peaks deterministically (first appearance)
        stats = stats.reindex(peak_order)

        # Flatten MultiIndex columns for export/readability: "<metric>_<stat>"
        stats.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col) for col in stats.columns
        ]
        stats.index.name = "peak"
        stats_df = stats.reset_index()

        # Pretty printing
        if print_summary:
            if print_mode.lower() == "blocks":
                # Block-style printing only
                self._print_peak_blocks(
                    stats_df,
                    metrics=metrics,
                    interval=interval,
                    float_format=float_format or "%.6g",
                    show_minmax=show_minmax_in_blocks,
                )

            else:
                # Table-style printing
                ordered_cols = ["peak"]
                for m in metrics:
                    for s in ("count", "mean", "std", "min", "median", "max"):
                        c = f"{m}_{s}"
                        if c in stats_df.columns:
                            ordered_cols.append(c)
                    for p in percentiles:
                        qlab = f"q{int(round(100*p)):02d}"
                        c = f"{m}_{qlab}"
                        if c in stats_df.columns:
                            ordered_cols.append(c)

                printable = stats_df.loc[
                    :, [c for c in ordered_cols if c in stats_df.columns]
                ].copy()

                with pd.option_context("display.max_columns", None, "display.width", 120):
                    if float_format is not None:
                        pd.set_option(
                            "display.float_format",
                            lambda x: float_format % x if np.isfinite(x) else "nan",
                        )
                    print("\nPer-peak summary statistics:")
                    print(printable.to_string(index=False))
                    if float_format is not None:
                        pd.reset_option("display.float_format")

        # Optional export
        if export_path is not None:
            if self.fits is None:
                raise RuntimeError("summary_by_peak export requires cached fitters. Run .fit(return_fitters=True) first.")

            # Use your existing export_table() helper for consistent metadata behaviour
            # (handles .csv vs .txt/.tsv, metadata headers, etc.)
            export_table(
                stats_df,
                out_path=export_path,
                metadata={"table_kind": "summary_by_peak"},
                fits_with_fitters=self.fits,
                include_metadata=include_metadata,
                meta_in_csv=meta_in_csv,
                float_format=float_format,
            )

        return stats_df

    def _print_peak_blocks(
        self,
        stats_df,
        *,
        metrics=("position", "fwhm", "intensity"),
        interval=(0.05, 0.95),
        float_format="%.6g",
        show_minmax=False,
    ) -> None:
        """
        Pretty-print per-peak summary statistics as blocks.

        Expects stats_df produced by summary_by_peak() (with flattened columns like
        'position_mean', 'position_std', 'position_median', 'position_q05', etc.).
        """
        import math

        def fmt(x):
            if x is None:
                return "—"
            try:
                if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                    return "—"
            except Exception:
                pass
            try:
                return float_format % float(x)
            except Exception:
                return str(x)

        qlo = f"q{int(round(100 * interval[0])):02d}"
        qhi = f"q{int(round(100 * interval[1])):02d}"

        for _, row in stats_df.iterrows():
            peak = row.get("peak", "UNKNOWN")
            print(f"\n{peak}")
            print("-" * len(str(peak)))

            for m in metrics:
                c_count = f"{m}_count"
                c_mean  = f"{m}_mean"
                c_std   = f"{m}_std"
                c_med   = f"{m}_median"
                c_qlo   = f"{m}_{qlo}"
                c_qhi   = f"{m}_{qhi}"
                c_min   = f"{m}_min"
                c_max   = f"{m}_max"

                if c_mean not in stats_df.columns:
                    continue  # metric absent

                count = row.get(c_count, None)
                mean  = row.get(c_mean, None)
                std   = row.get(c_std, None)
                med   = row.get(c_med, None)
                lo    = row.get(c_qlo, None)
                hi    = row.get(c_qhi, None)

                line = f"  {m:<10} n={int(count) if count is not None else '—'} | "
                line += f"mean±std {fmt(mean)} ± {fmt(std)} | median {fmt(med)}"

                # Interval (only if quantiles exist)
                if c_qlo in stats_df.columns and c_qhi in stats_df.columns:
                    line += f" | {qlo}–{qhi} {fmt(lo)}–{fmt(hi)}"

                # Optional min/max
                if show_minmax and c_min in stats_df.columns and c_max in stats_df.columns:
                    line += f" | min–max {fmt(row.get(c_min))}–{fmt(row.get(c_max))}"

                print(line)


# -------------------------
# Public classes: Raman / PL
# -------------------------
class RamanBatch(_BaseBatch):
    """
    Convenience class for Raman batch processing (multiple single spectra).

    This is a thin orchestration wrapper around:
      - load_spectra(...)
      - fit_spectra_batch(...)
      - collect_fit_parameters_from_fitters(...)
      - plotting helpers and export_table(...)

    Peak definition (recommended behaviour)
    --------------------------------------
    The fitter supports both library-driven defaults and user-defined peaks.

    Precedence is deterministic and matches single-spectrum fitting:
      1) If custom_peaks is provided (dict), it REPLACES any defaults derived from
         materials/substrate.
      2) remove_peaks (if provided) is applied last and always wins.

    Parameters (selected)
    ---------------------
    files : Sequence[str]
        Input file paths (e.g. .wdf or .txt depending on your importer).
    materials : list[str] | None
        Used only when custom_peaks is None. Determines default peak set.
    substrate : str | None
        Used only when custom_peaks is None. Adds substrate peaks if applicable.
    custom_peaks : dict | None
        User-defined peak bounds; replaces defaults when provided.

        Format:
            {peak_name: ([pos_lb, fwhm_lb, amp_lb], [pos_ub, fwhm_ub, amp_ub])}

        Example:
            custom_peaks = {
                "E2g": ([380, 1, 0], [392, 20, 50]),
                "A1g": ([402, 1, 0], [410, 20, 50]),
            }
    remove_peaks : list[str] | tuple[str, ...] | None
        Peak labels to remove after peak-definition is resolved. Useful for
        suppressing unwanted peaks (e.g. substrate peaks).
    peak_order : list[str] | None
        Optional deterministic ordering for peaks when applicable.
        If custom_peaks is provided and peak_order is None, insertion order is used.

    Other kwargs are passed directly to RamanFit.RamanFit, including:
    - baseline_method
    - smoothing / smooth_window / smooth_order
    - peak_profile
    - preprocessing
    - custom_peaks / remove_peaks / peak_order

    Examples
    --------
    (A) Default library peaks (materials/substrate) with removal
        >>> from ramanpl import Batch
        >>> b = Batch.RamanBatch(
        ...     files,
        ...     materials=["MoS2"],
        ...     substrate="Si",
        ...     remove_peaks=["Si"],
        ...     baseline_method="arPLS",
        ... )
        >>> b.fit()
        >>> b.plot_overlay(normalise="max", raw_fit=True)

    (B) Custom peaks (replaces defaults) + optional removal
        >>> custom_peaks = {
        ...     "E2g": ([380, 1, 0], [392, 20, 50]),
        ...     "A1g": ([402, 1, 0], [410, 20, 50]),
        ... }
        >>> b = Batch.RamanBatch(files, custom_peaks=custom_peaks, remove_peaks=["A1g"])
        >>> b.fit()
        >>> stats = b.summary_by_peak(print_mode="blocks")
        >>> b.export("raman_fit_wide.csv", wide=True)

    Notes
    -----
    - .fit(return_fitters=True) is required for parameter tables and summary_by_peak().
    - Export metadata will reflect custom_peaks/remove_peaks when supported by the fitter.
    """

    def __init__(self, files: Sequence[str], **fitter_kwargs):
        from ramanpl import RamanFit  # module
        super().__init__(
            files=files,
            axis="wavenumber",
            fitter_module=RamanFit,
            fitter_kwargs=fitter_kwargs,
        )


class PLBatch(_BaseBatch):
    """
    Convenience class for PL batch processing (multiple single spectra).

    Peak definition
    ---------------
    The fitter supports both default peak sets and user-defined peaks.

    Precedence is deterministic and matches single-spectrum fitting:
      1) If custom_peaks is provided (dict), it REPLACES defaults.
      2) remove_peaks (if provided) is applied last and always wins.

    Parameters (selected)
    ---------------------
    files : Sequence[str]
        Input file paths.
    custom_peaks : dict | None
        User-defined peak bounds; replaces defaults when provided.

        Format:
            {peak_name: ([pos_lb, gamma_lb, amp_lb], [pos_ub, gamma_ub, amp_ub])}

        Example:
            custom_peaks = {
                "Trion":   ([1.88, 0.01, 0], [1.96, 0.20, 10]),
                "Exciton": ([1.96, 0.01, 0], [2.05, 0.20, 10]),
            }
    remove_peaks : list[str] | tuple[str, ...] | None
        Peak labels to remove after peak-definition is resolved.
    peak_order : list[str] | None
        Optional deterministic ordering for peaks when applicable.

    Other kwargs are passed directly to PLFit.PLfit, including:
    - baseline_method
    - smoothing / smooth_window / smooth_order
    - peak_profile
    - preprocessing
    - custom_peaks / remove_peaks / peak_order

    Examples
    --------
    (A) Custom peaks (replaces defaults) + removal
        >>> from ramanpl import Batch
        >>> custom_peaks = {
        ...     "Trion":   ([1.88, 0.01, 0], [1.96, 0.20, 10]),
        ...     "Exciton": ([1.96, 0.01, 0], [2.05, 0.20, 10]),
        ... }
        >>> b = Batch.PLBatch(files, custom_peaks=custom_peaks, remove_peaks=["Trion"])
        >>> b.fit()
        >>> b.plot_waterfall(raw_fit=True)

    (B) Export wide table
        >>> b.export("pl_fit_wide.csv", wide=True)

    Notes
    -----
    - .fit(return_fitters=True) is required for parameter tables and summary_by_peak().
    """

    def __init__(self, files: Sequence[str], **fitter_kwargs):
        from ramanpl import PLfit  # module
        super().__init__(
            files=files,
            axis="energy",
            fitter_module=PLfit,
            fitter_kwargs=fitter_kwargs,
        )
