"""Internal batch plotting helpers for RamanPL v0.6.8."""

from __future__ import annotations

from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from ramanpl.operation import AlignType, ArithmeticSpectrum, Spectrum

NormType = Literal["none", "max", "area", "peak"]
YSelectType = Literal["raw", "fit", "raw+fit"]


def _align_spectra(
    spectra: Sequence[Spectrum],
    *,
    align: AlignType = "interp",
    reference: int = 0,
) -> list[Spectrum]:
    spectra = list(spectra)
    if len(spectra) == 0:
        return []

    if not (0 <= reference < len(spectra)):
        raise IndexError(f"reference index {reference} out of range for {len(spectra)} spectra")

    ref = spectra[reference]
    x_ref = np.asarray(ref.x, dtype=float).ravel()

    out = []
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

    aligned_raw = _align_spectra(spectra, align=align, reference=reference)

    aligned_fit = None
    if fit_spectra is not None:
        aligned_fit = _align_spectra(fit_spectra, align=align, reference=reference)

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

    aligned_raw = _align_spectra(spectra, align=align, reference=reference)

    aligned_fit = None
    if fit_spectra is not None:
        aligned_fit = _align_spectra(fit_spectra, align=align, reference=reference)
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

