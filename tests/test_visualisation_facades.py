import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ramanpl.batch import (
    _BaseBatch,
    plot_fitted_parameters,
    plot_overlay,
    plot_waterfall,
)
from ramanpl.mapping._pl_mapping import PLMapping
from ramanpl.mapping._raman_mapping import RamanMapping
from ramanpl.operation import Spectrum
from ramanpl.single_fit.PLfit import PLfit
from ramanpl.single_fit.RamanFit import RamanFit
from ramanpl.visualisation._preview import _plot_raw_spectrum


def teardown_function():
    plt.close("all")


def _param_names(callable_obj):
    return list(inspect.signature(callable_obj).parameters)


def _defaults(callable_obj):
    return {
        name: param.default
        for name, param in inspect.signature(callable_obj).parameters.items()
        if param.default is not inspect._empty
    }


def test_public_plotting_facade_signatures_are_preserved():
    assert _param_names(RamanMapping.plot_spectrum_fit) == ["self", "x", "y"]
    assert _param_names(RamanMapping.plot_residual_distribution) == [
        "self", "filter_threshold", "robust", "p_low", "p_high", "hist_bins", "cmap",
    ]
    assert _defaults(RamanMapping.plot_residual_distribution) == {
        "filter_threshold": None, "robust": True, "p_low": 5,
        "p_high": 95, "hist_bins": 50, "cmap": "inferno",
    }
    assert _param_names(RamanMapping.plot_ratio_heatmap) == [
        "self", "ratio_type", "cmap", "filter_range", "x_range", "y_range",
    ]
    assert _param_names(RamanMapping.plot_heatmap) == [
        "self", "data_type", "cmap", "filter_range", "x_range",
        "y_range", "specific_wavenumber", "peak_name",
    ]

    assert _param_names(PLMapping.plot_spectrum_fit) == ["self", "x", "y"]
    assert _param_names(PLMapping.plot_residual_distribution) == [
        "self", "filter_threshold", "robust", "p_low", "p_high", "hist_bins", "cmap",
    ]
    assert _param_names(PLMapping.plot_heatmap) == [
        "self", "data_type", "cmap", "filter_range", "specific_xdata", "x_range", "y_range",
    ]

    assert _param_names(plot_overlay) == [
        "spectra", "fit_spectra", "align", "reference", "normalise",
        "y_select", "xlim", "x_range", "title", "show_legend",
        "fit_linestyle", "fit_linewidth",
    ]
    assert _param_names(plot_waterfall) == [
        "spectra", "fit_spectra", "align", "reference", "normalise",
        "offset", "y_select", "xlim", "x_range", "title", "show_labels",
        "fit_linestyle", "fit_linewidth",
    ]
    assert _param_names(plot_fitted_parameters) == [
        "df", "metric", "peaks", "x_from_source", "sort_x", "title", "ylabel", "show_legend",
    ]
    assert _param_names(_BaseBatch.plot_overlay) == [
        "self", "normalise", "title", "raw_fit", "x_range", "align", "reference", "show_legend",
    ]
    assert _param_names(_BaseBatch.plot_waterfall) == [
        "self", "normalise", "offset", "title", "raw_fit", "x_range", "align",
        "reference", "show_labels",
    ]
    assert _param_names(_BaseBatch.plot_parameters) == [
        "self", "metric", "peaks", "x", "x_label", "sort_x", "title", "ylabel", "show_legend",
    ]
    assert _param_names(RamanFit.plot_fit) == [
        "self", "params", "offset", "scale", "x_lim", "y_lim", "x_ticks",
    ]
    assert _param_names(PLfit.plot_fit) == ["self", "params", "offset", "scale", "x_lim"]


def _spectra(axis="wavenumber"):
    x = np.linspace(1.0, 5.0, 9)
    return [
        Spectrum(y=np.sin(x) + i, x=x, axis=axis, source=f"spec_{i}")
        for i in range(2)
    ]


def test_batch_free_functions_and_base_methods_return_fig_ax():
    raw = _spectra()
    fit = [Spectrum(y=s.y * 0.9, x=s.x, axis=s.axis, source=f"{s.source} fit") for s in raw]

    for fig, ax in [
        plot_overlay(raw),
        plot_overlay(raw, fit_spectra=fit, y_select="raw+fit"),
        plot_waterfall(raw),
        plot_fitted_parameters(
            __import__("pandas").DataFrame(
                [
                    {"source": "a", "peak": "p1", "position": 1.0, "fwhm": 2.0, "peak_height": 3.0},
                    {"source": "b", "peak": "p1", "position": 2.0, "fwhm": 3.0, "peak_height": 4.0},
                ]
            )
        ),
    ]:
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    batch = _BaseBatch(files=[], axis="wavenumber", fitter_module=None, fitter_kwargs={})
    batch.specs = raw
    batch.rows = [
        {"source": "a", "peak": "p1", "position": 1.0, "fwhm": 2.0, "peak_height": 3.0},
        {"source": "b", "peak": "p1", "position": 2.0, "fwhm": 3.0, "peak_height": 4.0},
    ]
    assert isinstance(batch.plot_overlay()[0], Figure)
    assert isinstance(batch.plot_waterfall()[0], Figure)
    assert isinstance(batch.plot_parameters()[0], Figure)


def _make_raman_mapping():
    m = RamanMapping.__new__(RamanMapping)
    x = np.linspace(350.0, 430.0, 81)
    cube = np.ones((2, 3, x.size), dtype=float)
    m.X, m.Y = 3, 2
    m.wavenumber = x
    m.spectra = cube
    m.data_range = (350.0, 430.0)
    m.smoothing = False
    m.background_remove = False
    m.normalize = False
    m.step_size = 0.5
    m.peak_profile = "lorentzian"
    m.params_per_peak = 3
    m.peak_params = ["E2g", "A1g"]
    m.fitted_params = np.zeros((m.Y, m.X, 6), dtype=float)
    m.fitted_params[:, :, :] = np.array([385.0, 3.0, 2.0, 405.0, 4.0, 3.0])
    m.norm_scale_map = np.ones((m.Y, m.X), dtype=float)
    m.residual_map = np.full((m.Y, m.X), 0.02, dtype=float)
    m.peak_positions = np.dstack([
        np.full((m.Y, m.X), 385.0),
        np.full((m.Y, m.X), 405.0),
    ])
    m.peak_intensities = np.dstack([
        np.full((m.Y, m.X), 2.0),
        np.full((m.Y, m.X), 3.0),
    ])
    m.Peaks_distance = np.full((m.Y, m.X), 20.0)
    m.ratio_A1g_E2g = np.full((m.Y, m.X), 1.5)
    m.ratio_E2g_A1g = np.full((m.Y, m.X), 2.0 / 3.0)
    return m


def _make_pl_mapping():
    m = PLMapping.__new__(PLMapping)
    x = np.linspace(1.8, 2.1, 81)
    cube = np.ones((2, 3, x.size), dtype=float)
    m.X, m.Y = 3, 2
    m.xdata = x
    m.spectra = cube
    m.data_range = (1.8, 2.1)
    m.smoothing = False
    m.background_remove = False
    m.normalize = False
    m.step_size = 0.5
    m.peak_profile = "lorentzian"
    m.params_per_peak = 3
    m.fitted_params = np.zeros((m.Y, m.X, 6), dtype=float)
    m.fitted_params[:, :, :] = np.array([1.9, 0.02, 1.0, 2.0, 0.03, 2.0])
    m.norm_scale_map = np.ones((m.Y, m.X), dtype=float)
    m.residual_map = np.full((m.Y, m.X), 0.02, dtype=float)
    m.peak_positions = np.dstack([
        np.full((m.Y, m.X), 1.9),
        np.full((m.Y, m.X), 2.0),
    ])
    m.peak_intensities = np.dstack([
        np.full((m.Y, m.X), 1.0),
        np.full((m.Y, m.X), 2.0),
    ])
    return m


def test_raman_mapping_facades_create_expected_figures_and_return_none():
    m = _make_raman_mapping()
    assert m.plot_spectrum_fit(1, 1) is None
    fig = plt.gcf()
    assert len(fig.axes[0].lines) >= 2
    assert fig.axes[0].get_xlabel() != ""

    assert m.plot_residual_distribution() is None
    assert len(plt.gcf().axes) == 3

    assert m.plot_ratio_heatmap() is None
    assert len(plt.gcf().axes) == 2

    assert m.plot_heatmap(data_type="position", peak_name="E2g") is None
    assert len(plt.gcf().axes[0].images) == 1

    with pytest.raises(ValueError):
        m.plot_spectrum_fit(9, 0)
    m.fitted_params[0, 0, :] = np.nan
    with pytest.raises(ValueError):
        m.plot_spectrum_fit(0, 0)


def test_pl_mapping_facades_create_expected_figures_and_return_none():
    m = _make_pl_mapping()
    assert m.plot_spectrum_fit(1, 1) is None
    assert len(plt.gcf().axes[0].lines) >= 2

    assert m.plot_residual_distribution() is None
    assert len(plt.gcf().axes) == 3

    assert m.plot_heatmap(data_type="exciton_position") is None
    assert len(plt.gcf().axes[0].images) == 1

    with pytest.raises(ValueError):
        m.plot_spectrum_fit(-1, 0)
    m.fitted_params[0, 0, :] = np.nan
    with pytest.raises(ValueError):
        m.plot_spectrum_fit(0, 0)


def _make_single_raman():
    obj = RamanFit.__new__(RamanFit)
    x = np.linspace(350.0, 430.0, 81)
    obj.wavenumber = x
    obj.raw_spectra = np.ones_like(x)
    obj.processed_spectra = np.ones_like(x)
    obj.intensity_normal = np.ones_like(x)
    obj.peak_intensity = 1.0
    obj.normalize = False
    obj.peak_profile = "lorentzian"
    obj.params_per_peak = 3
    obj.peak_labels = ["E2g", "A1g"]
    obj._smoothed_spectra = None
    obj._baseline = None
    obj._corrected_spectra = None
    return obj


def _make_single_pl():
    obj = PLfit.__new__(PLfit)
    x = np.linspace(1.8, 2.1, 81)
    obj.energy = x
    obj.processed_spectra = np.ones_like(x)
    obj.intensity_normal = np.ones_like(x)
    obj.peak_intensity = 1.0
    obj.normalize = False
    obj.peak_profile = "lorentzian"
    obj.params_per_peak = 3
    obj.peak_labels = ["Trion", "Exciton"]
    obj._smoothed_spectra = None
    obj._baseline = None
    obj._corrected_spectra = None
    return obj


def test_single_fit_facades_create_figures_and_return_none():
    raman = _make_single_raman()
    assert raman.plot_fit(np.array([385.0, 3.0, 2.0, 405.0, 4.0, 3.0])) is None
    assert len(plt.gcf().axes[0].lines) >= 4

    pl = _make_single_pl()
    assert pl.plot_fit(np.array([1.9, 0.02, 1.0, 2.0, 0.03, 2.0])) is None
    assert len(plt.gcf().axes[0].lines) >= 4


def test_raw_preview_helper_array_and_non_square_mapping_pixel():
    x = np.linspace(0.0, 1.0, 5)
    fig, ax = _plot_raw_spectrum(x, x**2)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert np.allclose(ax.lines[0].get_ydata(), x**2)

    class MappingLike:
        pass

    mapping = MappingLike()
    mapping.wavenumber = x
    mapping.spectra = np.zeros((2, 3, 5), dtype=float)
    mapping.spectra[1, 2, :] = np.arange(5.0)
    fig, ax = _plot_raw_spectrum(mapping, x=2, y=1)
    assert isinstance(fig, Figure)
    assert np.allclose(ax.lines[0].get_ydata(), np.arange(5.0))

    with pytest.raises(ValueError):
        _plot_raw_spectrum(mapping, x=3, y=1)
