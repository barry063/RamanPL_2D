from __future__ import annotations

from typing import Any, Dict, Literal, Optional

Modality = Literal["Raman", "PL"]
AxisKind = Literal["raman_shift_cm-1", "energy_eV", "wavelength_nm"]
PeakProfile = Literal["lorentzian", "pvoigt"]
CoordMode = Literal["pixel", "real"]


def normalise_modality(value: str) -> Modality:
    s = str(value).strip().lower()
    if s == "raman":
        return "Raman"
    if s in {"pl", "photoluminescence"}:
        return "PL"
    raise ValueError(f"Unsupported modality '{value}'.")


def normalise_axis_kind(value: str) -> AxisKind:
    s = str(value).strip().lower()
    mapping = {
        "raman_shift_cm-1": "raman_shift_cm-1",
        "raman_shift": "raman_shift_cm-1",
        "wavenumber": "raman_shift_cm-1",
        "energy_ev": "energy_eV",
        "energy": "energy_eV",
        "wavelength_nm": "wavelength_nm",
        "wavelength": "wavelength_nm",
    }
    if s not in mapping:
        raise ValueError(f"Unsupported axis_kind '{value}'.")
    return mapping[s]


def normalise_peak_profile(value: str) -> PeakProfile:
    s = str(value).strip().lower()
    if s not in {"lorentzian", "pvoigt"}:
        raise ValueError("peak_profile must be 'lorentzian' or 'pvoigt'")
    return s  # type: ignore[return-value]


def normalise_coord_mode(value: str) -> CoordMode:
    s = str(value).strip().lower()
    if s not in {"pixel", "real"}:
        raise ValueError("coord_mode must be 'pixel' or 'real'")
    return s  # type: ignore[return-value]


def normalise_baseline_spec(
    baseline_method: Any,
    *,
    poly_degree: Optional[int] = None,
    gaussian_sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Canonical user-facing baseline spec for v0.3.8.

    Preferred form:
        {"method": "poly", "poly_order": 3}
        {"method": "gaussian", "gaussian_sigma": 10}
        {"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6}

    Legacy inputs are still accepted and normalised here.
    """
    if isinstance(baseline_method, dict):
        spec = dict(baseline_method)

    elif isinstance(baseline_method, str):
        spec = {"method": baseline_method}

    elif isinstance(baseline_method, (tuple, list)):
        if len(baseline_method) == 0:
            raise ValueError("baseline_method tuple/list cannot be empty.")

        method = baseline_method[0]
        spec = {"method": method}

        if len(baseline_method) == 2:
            payload = baseline_method[1]
            if isinstance(payload, dict):
                spec.update(payload)
            else:
                method_l = str(method).strip().lower()
                if method_l == "poly":
                    spec["poly_order"] = int(payload)
                elif method_l == "gaussian":
                    spec["gaussian_sigma"] = float(payload)
                else:
                    raise ValueError(
                        f"Shorthand tuple baseline_method is not supported for method='{method_l}'."
                    )
        elif len(baseline_method) > 2:
            raise ValueError("baseline_method tuple/list must be length 1 or 2.")

    else:
        raise TypeError(
            "baseline_method must be str, tuple/list, or dict."
        )

    if "method" not in spec:
        raise ValueError("baseline_method dict must include key 'method'.")

    spec["method"] = str(spec["method"]).strip().lower()

    # Legacy alias -> canonical key
    if "poly_degree" in spec and "poly_order" not in spec:
        spec["poly_order"] = int(spec.pop("poly_degree"))

    if spec["method"] == "poly":
        if poly_degree is not None and "poly_order" not in spec:
            spec["poly_order"] = int(poly_degree)
        spec.setdefault("poly_order", 3)

    if spec["method"] == "gaussian":
        if gaussian_sigma is not None and "gaussian_sigma" not in spec:
            spec["gaussian_sigma"] = float(gaussian_sigma)

    return spec


def baseline_spec_to_runtime(spec: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Convert canonical baseline spec into the kwargs currently consumed by BaselineAPI.subtract().
    """
    d = dict(spec)
    method = str(d.pop("method")).strip().lower()

    # Runtime layer still accepts poly_degree
    if "poly_order" in d and "poly_degree" not in d:
        d["poly_degree"] = int(d["poly_order"])

    return method, d