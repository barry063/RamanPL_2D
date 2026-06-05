"""
test_api_stability.py
---------------------
Freeze regression tests for the public-surface contract (v0.5.5 through v0.6.6).

Enforces the additive-only column rule and frozen suffix/QA vocabulary
documented in docs/source/api-stability.md.

Reference column lists are derived analytically from:
  - build_feature_row() emission order (per-peak → separations → ratios → QA)
  - Fixture peak label order matching test_feature_table_mapping.py
    (Raman: E2g first, A1g second; PL: Trion first, Exciton second)
  - feature_table() prepends x, y as the first two columns
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ramanpl
from ramanpl.mapping import RamanMapping
from ramanpl.mapping._pl_mapping import PLMapping


# ---------------------------------------------------------------------------
# Frozen reference column lists
# ---------------------------------------------------------------------------

_QA_COLS = ["rmse", "ok", "n_starts", "n_params_at_bounds"]
_FROZEN_SUFFIXES = frozenset([
    "_position", "_fwhm", "_peak_height", "_peak_height_norm",
    "_component_area", "_component_area_norm", "_component_area_fraction",
    "_separation", "_ratio", "_area_ratio",
])

RAMAN_NO_PAIRS = [
    "x", "y",
    "E2g_position", "E2g_fwhm", "E2g_peak_height", "E2g_peak_height_norm",
    "E2g_component_area", "E2g_component_area_norm", "E2g_component_area_fraction",
    "A1g_position", "A1g_fwhm", "A1g_peak_height", "A1g_peak_height_norm",
    "A1g_component_area", "A1g_component_area_norm", "A1g_component_area_fraction",
] + _QA_COLS

RAMAN_WITH_PAIRS = [
    "x", "y",
    "E2g_position", "E2g_fwhm", "E2g_peak_height", "E2g_peak_height_norm",
    "E2g_component_area", "E2g_component_area_norm", "E2g_component_area_fraction",
    "A1g_position", "A1g_fwhm", "A1g_peak_height", "A1g_peak_height_norm",
    "A1g_component_area", "A1g_component_area_norm", "A1g_component_area_fraction",
    "A1g_E2g_separation",
    "A1g_E2g_ratio",
    "A1g_E2g_area_ratio",
] + _QA_COLS

PL_NO_PAIRS = [
    "x", "y",
    "Trion_position", "Trion_fwhm", "Trion_peak_height", "Trion_peak_height_norm",
    "Trion_component_area", "Trion_component_area_norm", "Trion_component_area_fraction",
    "Exciton_position", "Exciton_fwhm", "Exciton_peak_height", "Exciton_peak_height_norm",
    "Exciton_component_area", "Exciton_component_area_norm", "Exciton_component_area_fraction",
] + _QA_COLS

PL_WITH_PAIRS = [
    "x", "y",
    "Trion_position", "Trion_fwhm", "Trion_peak_height", "Trion_peak_height_norm",
    "Trion_component_area", "Trion_component_area_norm", "Trion_component_area_fraction",
    "Exciton_position", "Exciton_fwhm", "Exciton_peak_height", "Exciton_peak_height_norm",
    "Exciton_component_area", "Exciton_component_area_norm", "Exciton_component_area_fraction",
    "Exciton_Trion_separation",
    "Exciton_Trion_ratio",
    "Exciton_Trion_area_ratio",
] + _QA_COLS


# ---------------------------------------------------------------------------
# Fixture helpers (mirrors test_feature_table_mapping.py)
# ---------------------------------------------------------------------------

_N_PTS = 80
_X_RAMAN = np.linspace(300.0, 700.0, _N_PTS)
_X_PL = np.linspace(1.85, 2.10, _N_PTS)
_Y_MAP, _X_MAP = 3, 4

_CUSTOM_PEAKS_RAMAN = {
    "E2g": ([380.0, 1.0, 0.001], [392.0, 20.0, 5.0]),
    "A1g": ([402.0, 1.0, 0.001], [412.0, 20.0, 5.0]),
}
_CUSTOM_PEAKS_PL = {
    "Trion":   ([1.88, 0.005, 0.001], [1.96, 0.10, 5.0]),
    "Exciton": ([1.96, 0.005, 0.001], [2.05, 0.10, 5.0]),
}
_GOOD_PARAMS_RAMAN = np.array([386.0, 3.0, 2.0, 407.0, 4.0, 3.0], dtype=float)
_GOOD_PARAMS_PL    = np.array([1.91, 0.02, 1.5, 2.00, 0.03, 2.5], dtype=float)


def _make_raman_mapping():
    rng = np.random.default_rng(10)
    peak1 = np.exp(-0.5 * ((_X_RAMAN - 386.0) / 3.0) ** 2)
    peak2 = np.exp(-0.5 * ((_X_RAMAN - 407.0) / 4.0) ** 2)
    signal = peak1 + peak2
    cube = (signal[None, None, :] + rng.uniform(0.0, 0.02, (_Y_MAP, _X_MAP, _N_PTS))).astype(float)
    return RamanMapping.from_arrays(
        cube, _X_RAMAN, _X_MAP, _Y_MAP,
        custom_peaks=_CUSTOM_PEAKS_RAMAN,
        data_range=(300.0, 700.0),
        background_remove=False,
        smoothing=False,
        normalize=False,
    )


def _make_pl_mapping():
    rng = np.random.default_rng(20)
    peak1 = np.exp(-0.5 * ((_X_PL - 1.91) / 0.02) ** 2)
    peak2 = np.exp(-0.5 * ((_X_PL - 2.00) / 0.03) ** 2)
    signal = peak1 + peak2
    cube = (signal[None, None, :] + rng.uniform(0.0, 0.02, (_Y_MAP, _X_MAP, _N_PTS))).astype(float)
    return PLMapping.from_arrays(
        cube, _X_PL, _X_MAP, _Y_MAP,
        custom_peaks=_CUSTOM_PEAKS_PL,
        data_range=(1.85, 2.10),
        background_remove=False,
        smoothing=False,
        normalize=False,
    )


def _inject_fit(m, good_params):
    n_params = len(good_params)
    m.fitted_params = np.full((m.Y, m.X, n_params), float("nan"), dtype=float)
    m.residual_map = np.full((m.Y, m.X), float("nan"), dtype=float)
    m.norm_scale_map = np.full((m.Y, m.X), 1.0, dtype=float)
    m.fit_diagnostics_map = np.empty((m.Y, m.X), dtype=object)
    m.fit_diagnostics_map[:, :] = None
    n_peaks = len(list(m.peak_params))
    m.peak_intensities = np.full((m.Y, m.X, n_peaks), float("nan"), dtype=float)
    for j in range(m.Y):
        for i in range(m.X):
            m.fitted_params[j, i, :] = good_params
            m.residual_map[j, i] = 0.04
            m.norm_scale_map[j, i] = 5.0
            m.fit_diagnostics_map[j, i] = {
                "ok": True, "n_starts": 1,
                "n_params_at_lower_bounds": 0, "n_params_at_upper_bounds": 0,
            }
            for k in range(n_peaks):
                hwhm = float(good_params[m.params_per_peak * k + 1])
                amp  = float(good_params[m.params_per_peak * k + 2])
                h_norm = amp / (np.pi * hwhm) if hwhm != 0 else float("nan")
                m.peak_intensities[j, i, k] = h_norm * m.norm_scale_map[j, i]


def _is_ordered_subset(reference, actual_cols):
    """Return True if every item in reference appears in actual_cols in reference order."""
    actual_list = list(actual_cols)
    idx = 0
    for name in reference:
        try:
            pos = actual_list.index(name, idx)
            idx = pos + 1
        except ValueError:
            return False, name
    return True, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_raman_mapping_columns_subset_and_order_pinned():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()
    cols = list(df.columns)

    ok, missing = _is_ordered_subset(RAMAN_NO_PAIRS, cols)
    assert ok, f"Column '{missing}' missing or out of order in {cols}"

    assert cols[-len(_QA_COLS):] == _QA_COLS, (
        f"QA columns not a contiguous block at end: tail={cols[-len(_QA_COLS):]}"
    )


def test_raman_mapping_columns_with_pairs():
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table(
        ratios=[("A1g", "E2g")],
        separations=[("A1g", "E2g")],
        area_ratios=[("A1g", "E2g")],
    )
    cols = list(df.columns)

    ok, missing = _is_ordered_subset(RAMAN_WITH_PAIRS, cols)
    assert ok, f"Column '{missing}' missing or out of order in {cols}"

    assert cols[-len(_QA_COLS):] == _QA_COLS


def test_pl_mapping_columns_subset_and_order_pinned():
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)
    df = m.feature_table()
    cols = list(df.columns)

    ok, missing = _is_ordered_subset(PL_NO_PAIRS, cols)
    assert ok, f"Column '{missing}' missing or out of order in {cols}"

    assert cols[-len(_QA_COLS):] == _QA_COLS


def test_pl_mapping_columns_with_pairs():
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)
    df = m.feature_table(
        ratios=[("Exciton", "Trion")],
        separations=[("Exciton", "Trion")],
        area_ratios=[("Exciton", "Trion")],
    )
    cols = list(df.columns)

    ok, missing = _is_ordered_subset(PL_WITH_PAIRS, cols)
    assert ok, f"Column '{missing}' missing or out of order in {cols}"

    assert cols[-len(_QA_COLS):] == _QA_COLS


def test_suffix_vocabulary_is_frozen():
    """Every non-QA, non-spatial column must use one of the frozen suffixes."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table(
        ratios=[("A1g", "E2g")],
        separations=[("A1g", "E2g")],
        area_ratios=[("A1g", "E2g")],
    )

    skip = set(_QA_COLS) | {"x", "y"}
    for col in df.columns:
        if col in skip:
            continue
        assert any(col.endswith(suf) for suf in _FROZEN_SUFFIXES), (
            f"Column '{col}' does not end with a frozen suffix. "
            f"Frozen suffixes: {sorted(_FROZEN_SUFFIXES)}"
        )


def test_qa_column_names_are_frozen():
    """QA column set is a subset of every mapping feature table."""
    m_raman = _make_raman_mapping()
    _inject_fit(m_raman, _GOOD_PARAMS_RAMAN)
    m_pl = _make_pl_mapping()
    _inject_fit(m_pl, _GOOD_PARAMS_PL)

    for m, label in [(m_raman, "Raman"), (m_pl, "PL")]:
        df = m.feature_table()
        missing = set(_QA_COLS) - set(df.columns)
        assert not missing, f"{label} feature_table() missing QA cols: {missing}"


def test_ramanpl_ml_public_surface_is_frozen():
    """pca_reduce and kmeans_cluster must be in ramanpl.ml.__all__."""
    import ramanpl.ml as ml
    assert hasattr(ml, "__all__"), "ramanpl.ml has no __all__"
    ml_all = set(ml.__all__)
    required = {"pca_reduce", "kmeans_cluster"}
    missing = required - ml_all
    assert not missing, f"Missing from ramanpl.ml.__all__: {missing}"


def test_ramanpl_descriptors_public_surface_is_frozen():
    """build_feature_row and validate_peak_pairs must be reachable via ramanpl.descriptors."""
    from ramanpl import descriptors
    assert callable(descriptors.build_feature_row), (
        "ramanpl.descriptors.build_feature_row is not callable"
    )
    assert callable(descriptors.validate_peak_pairs), (
        "ramanpl.descriptors.validate_peak_pairs is not callable"
    )


def test_cluster_seeds_keyword_present_on_both_mapping_classes():
    """cluster_seeds=False keyword must exist on both fit_spectra methods (v0.6.5+)."""
    import inspect
    for cls, label in [(RamanMapping, "RamanMapping"), (PLMapping, "PLMapping")]:
        sig = inspect.signature(cls.fit_spectra)
        assert "cluster_seeds" in sig.parameters, (
            f"{label}.fit_spectra is missing the cluster_seeds parameter"
        )
        default = sig.parameters["cluster_seeds"].default
        assert default is False, (
            f"{label}.fit_spectra cluster_seeds default should be False, got {default!r}"
        )


# ---------------------------------------------------------------------------
# v0.6.1 — show_progress contract
# ---------------------------------------------------------------------------

def test_show_progress_keyword_on_mapping_fit_spectra():
    """show_progress=True must be the frozen default on both mapping fit_spectra (v0.6.1+)."""
    import inspect
    for cls, label in [(RamanMapping, "RamanMapping"), (PLMapping, "PLMapping")]:
        sig = inspect.signature(cls.fit_spectra)
        assert "show_progress" in sig.parameters, (
            f"{label}.fit_spectra is missing the show_progress parameter"
        )
        default = sig.parameters["show_progress"].default
        assert default is True, (
            f"{label}.fit_spectra show_progress default should be True, got {default!r}"
        )


def test_show_progress_keyword_on_batch_fit():
    """show_progress=True must be the frozen default on the public batch fit entry point (v0.6.1+)."""
    import inspect
    from ramanpl.batch import fit_spectra_batch
    sig = inspect.signature(fit_spectra_batch)
    assert "show_progress" in sig.parameters, (
        "fit_spectra_batch is missing the show_progress parameter"
    )
    default = sig.parameters["show_progress"].default
    assert default is True, (
        f"fit_spectra_batch show_progress default should be True, got {default!r}"
    )


# ---------------------------------------------------------------------------
# v0.6.4 — n_jobs contract
# ---------------------------------------------------------------------------

def test_n_jobs_keyword_on_mapping_fit_spectra():
    """n_jobs=1 must be the frozen default on both mapping fit_spectra methods (v0.6.4+)."""
    import inspect
    for cls, label in [(RamanMapping, "RamanMapping"), (PLMapping, "PLMapping")]:
        sig = inspect.signature(cls.fit_spectra)
        assert "n_jobs" in sig.parameters, (
            f"{label}.fit_spectra is missing the n_jobs parameter"
        )
        default = sig.parameters["n_jobs"].default
        assert default == 1, (
            f"{label}.fit_spectra n_jobs default should be 1, got {default!r}"
        )


# ---------------------------------------------------------------------------
# v0.6.2 — autotune contract
# ---------------------------------------------------------------------------

def test_autotune_methods_exist_on_mapping_and_single_fit():
    """autotune_baseline and apply_choice must exist on all four public fit objects (v0.6.2+)."""
    from ramanpl.single_fit.RamanFit import RamanFit
    from ramanpl.single_fit.PLfit import PLfit
    targets = [
        (RamanMapping, "RamanMapping"),
        (PLMapping, "PLMapping"),
        (RamanFit, "RamanFit"),
        (PLfit, "PLfit"),
    ]
    for cls, label in targets:
        assert callable(getattr(cls, "autotune_baseline", None)), (
            f"{label} is missing callable autotune_baseline"
        )
        assert callable(getattr(cls, "apply_choice", None)), (
            f"{label} is missing callable apply_choice"
        )


# ---------------------------------------------------------------------------
# v0.6.3 — method_grids contract
# ---------------------------------------------------------------------------

def test_method_grids_is_supported_autotune_kwarg():
    """method_grids must be an accepted keyword on autotune_baseline (v0.6.3+)."""
    import inspect
    from ramanpl.single_fit.RamanFit import RamanFit
    sig = inspect.signature(RamanFit.autotune_baseline)
    assert "method_grids" in sig.parameters, (
        "RamanFit.autotune_baseline is missing the method_grids parameter"
    )


def test_deprecated_autotune_kwargs_absent():
    """Removed v0.6.2-era kwargs 'methods' and 'lam_grid' must not appear in autotune_baseline (v0.6.3+)."""
    import inspect
    from ramanpl.single_fit.RamanFit import RamanFit
    from ramanpl.single_fit.PLfit import PLfit
    for cls, label in [(RamanFit, "RamanFit"), (PLfit, "PLfit")]:
        sig = inspect.signature(cls.autotune_baseline)
        for removed_kwarg in ("methods", "lam_grid"):
            assert removed_kwarg not in sig.parameters, (
                f"{label}.autotune_baseline should not accept removed kwarg {removed_kwarg!r}"
            )


# ---------------------------------------------------------------------------
# v0.6.7 — component-area contract
# ---------------------------------------------------------------------------

def test_component_area_columns_present_raman_mapping():
    """RamanMapping.feature_table() emits all three component-area columns for each peak (v0.6.7+)."""
    m = _make_raman_mapping()
    _inject_fit(m, _GOOD_PARAMS_RAMAN)
    df = m.feature_table()
    for peak in ("E2g", "A1g"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, (
                f"RamanMapping feature_table() missing column '{peak}{suffix}'"
            )


def test_component_area_columns_present_pl_mapping():
    """PLMapping.feature_table() emits all three component-area columns for each peak (v0.6.7+)."""
    m = _make_pl_mapping()
    _inject_fit(m, _GOOD_PARAMS_PL)
    df = m.feature_table()
    for peak in ("Trion", "Exciton"):
        for suffix in ("_component_area", "_component_area_norm", "_component_area_fraction"):
            assert f"{peak}{suffix}" in df.columns, (
                f"PLMapping feature_table() missing column '{peak}{suffix}'"
            )


def test_area_ratios_keyword_on_all_feature_table_entry_points():
    """area_ratios= keyword must be accepted on all five feature_table() entry points (v0.6.7+)."""
    import inspect
    from ramanpl.single_fit.RamanFit import RamanFit
    from ramanpl.single_fit.PLfit import PLfit
    from ramanpl.batch import RamanBatch
    from ramanpl.mapping._preprocess import _MappingPreprocessMixin

    for cls, method_name, label in [
        (RamanFit, "feature_table", "RamanFit"),
        (PLfit, "feature_table", "PLfit"),
        (RamanBatch, "feature_table", "RamanBatch"),
        (_MappingPreprocessMixin, "feature_table", "_MappingPreprocessMixin"),
    ]:
        sig = inspect.signature(getattr(cls, method_name))
        assert "area_ratios" in sig.parameters, (
            f"{label}.{method_name} is missing the area_ratios parameter"
        )
