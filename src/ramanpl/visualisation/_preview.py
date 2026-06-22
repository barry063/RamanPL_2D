"""Internal raw spectrum preview helpers for RamanPL v0.6.8."""

from __future__ import annotations

import numpy as np


def _plot_raw_spectrum(
    source,
    values=None,
    *,
    x=None,
    y=None,
    title=None,
    xlabel=None,
    ylabel="Intensity (a.u.)",
):
    """Plot a raw spectrum from arrays or a single mapping pixel."""
    import matplotlib.pyplot as plt

    if values is None and x is not None and y is not None and hasattr(source, "spectra"):
        spectra = np.asarray(source.spectra, dtype=float)
        if spectra.ndim != 3:
            raise ValueError("mapping spectra must be a 3D array with shape [Y, X, N].")

        x_pixel = int(x)
        y_pixel = int(y)
        height, width, _ = spectra.shape
        if not (0 <= x_pixel < width and 0 <= y_pixel < height):
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        axis = getattr(source, "wavenumber", None)
        if axis is None:
            axis = getattr(source, "xdata", None)
        if axis is None:
            axis = getattr(source, "energy", None)
        if axis is None:
            raise ValueError("mapping object must provide wavenumber, xdata, or energy axis data.")

        x_plot = np.asarray(axis, dtype=float).ravel()
        y_plot = spectra[y_pixel, x_pixel, :]
        if xlabel is None:
            xlabel = "Energy (eV)" if hasattr(source, "xdata") or hasattr(source, "energy") else "Wavenumber (cm$^{-1}$)"
        if title is None:
            title = f"Raw spectrum at (X={x_pixel}, Y={y_pixel})"
    else:
        x_plot = np.asarray(source, dtype=float).ravel()
        y_plot = np.asarray(values, dtype=float).ravel()
        if xlabel is None:
            xlabel = "X"

    if x_plot.ndim != 1 or y_plot.ndim != 1:
        raise ValueError("x and y data must be one-dimensional.")
    if x_plot.size != y_plot.size:
        raise ValueError("x and y data must have the same length.")
    if x_plot.size == 0:
        raise ValueError("x and y data must not be empty.")

    fig, ax = plt.subplots()
    ax.plot(x_plot, y_plot)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return fig, ax
