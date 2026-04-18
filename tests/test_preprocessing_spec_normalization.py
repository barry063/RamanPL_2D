"""Tests for canonical preprocessing helper functions in preprocessing.py."""
import pytest

from ramanpl.preprocessing import (
    canonicalize_baseline_spec,
    canonicalize_pipeline_steps,
    normalize_backend_request,
    BaselineSubtract,
    SmoothSavGol,
    CropByRange,
    PreprocessStep,
)


# ---------------------------------------------------------------------------
# canonicalize_baseline_spec
# ---------------------------------------------------------------------------

def test_canonicalize_baseline_spec_valid_poly():
    spec = canonicalize_baseline_spec({"method": "poly", "poly_order": 3})
    assert spec["method"] == "poly"
    assert spec["poly_order"] == 3


def test_canonicalize_baseline_spec_valid_gaussian():
    spec = canonicalize_baseline_spec({"method": "gaussian", "gaussian_sigma": 10.0})
    assert spec["method"] == "gaussian"


def test_canonicalize_baseline_spec_valid_asls():
    spec = canonicalize_baseline_spec({"method": "asls", "lam": 1e5, "p": 0.01})
    assert spec["method"] == "asls"


def test_canonicalize_baseline_spec_wrong_type_raises():
    """Non-string, non-dict, non-tuple inputs should raise."""
    with pytest.raises(TypeError):
        canonicalize_baseline_spec(42)


def test_canonicalize_baseline_spec_dict_with_poly_degree_migrates_to_poly_order():
    """poly_degree in dict is silently migrated to poly_order (schema compatibility)."""
    spec = canonicalize_baseline_spec({"method": "poly", "poly_degree": 3})
    assert spec["method"] == "poly"
    assert spec.get("poly_order") == 3
    assert "poly_degree" not in spec


# ---------------------------------------------------------------------------
# canonicalize_pipeline_steps
# ---------------------------------------------------------------------------

def test_canonicalize_pipeline_steps_valid():
    steps = [CropByRange(data_range=(100.0, 200.0)), SmoothSavGol(window_length=5, polyorder=2)]
    result = canonicalize_pipeline_steps(steps)
    assert len(result) == 2
    assert all(isinstance(s, PreprocessStep) for s in result)


def test_canonicalize_pipeline_steps_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        canonicalize_pipeline_steps([])


def test_canonicalize_pipeline_steps_not_list_raises():
    with pytest.raises((ValueError, TypeError)):
        canonicalize_pipeline_steps("not_a_list")


def test_canonicalize_pipeline_steps_wrong_element_type_raises():
    with pytest.raises(TypeError):
        canonicalize_pipeline_steps([{"step": "crop"}])


def test_canonicalize_pipeline_steps_tuple_is_accepted():
    steps = (CropByRange(data_range=(100.0, 200.0)),)
    result = canonicalize_pipeline_steps(steps)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# normalize_backend_request
# ---------------------------------------------------------------------------

def test_normalize_backend_request_auto():
    assert normalize_backend_request("auto") == "auto"


def test_normalize_backend_request_native():
    assert normalize_backend_request("native") == "native"


def test_normalize_backend_request_ramanspy():
    assert normalize_backend_request("ramanspy") == "ramanspy"


def test_normalize_backend_request_unknown_raises():
    with pytest.raises(ValueError):
        normalize_backend_request("bogus_backend")


def test_normalize_backend_request_case_insensitive():
    result = normalize_backend_request("AUTO")
    assert result == "auto"


# ---------------------------------------------------------------------------
# BaselineSubtract normalises in __post_init__
# ---------------------------------------------------------------------------

def test_baseline_subtract_normalises_on_construction():
    step = BaselineSubtract(baseline_spec={"method": "poly", "poly_order": 2})
    spec = step.baseline_spec
    assert isinstance(spec, dict)
    assert spec["method"] == "poly"


def test_baseline_subtract_apply_does_not_re_normalise(monkeypatch):
    import numpy as np
    from ramanpl import preprocessing as pre
    from ramanpl.preprocessing import SpectralDataset

    calls = []
    original = pre.normalise_baseline_spec

    def counting_normalise(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(pre, "normalise_baseline_spec", counting_normalise)

    step = BaselineSubtract(baseline_spec={"method": "poly", "poly_order": 1})
    calls.clear()

    x = np.linspace(100, 200, 50)
    y = np.ones(50)
    ds = SpectralDataset(x=x, y=y, meta={}, modality="Raman", axis_kind="raman_shift_cm-1")
    step.apply(ds)

    assert len(calls) == 0, "apply() should not re-normalise the already-normalised spec"
