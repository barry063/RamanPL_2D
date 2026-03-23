# src/ramanpl/exporter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import csv
import math
import os


Number = Union[int, float]


@dataclass(frozen=True)
class FitExportRow:
    peak: str
    centre: float
    fwhm: float
    scale: float
    amp: float
    peak_height_norm: float
    peak_height: float


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def _normalise_meta_value(v: Any) -> Any:
    """
    Convert metadata values into export-safe serialisable forms.

    Rules
    -----
    - numpy arrays -> lists
    - tuples -> lists
    - objects with to_dict() -> dict
    - dict / list -> recursively normalised
    - everything else -> returned as-is

    This keeps metadata headers readable and stable across single, batch,
    and mapping exports.
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None and isinstance(v, np.ndarray):
        return v.tolist()

    if isinstance(v, tuple):
        return [_normalise_meta_value(x) for x in v]

    if isinstance(v, list):
        return [_normalise_meta_value(x) for x in v]

    if isinstance(v, dict):
        return {str(k): _normalise_meta_value(val) for k, val in v.items()}

    if hasattr(v, "to_dict") and callable(getattr(v, "to_dict")):
        try:
            return _normalise_meta_value(v.to_dict())
        except Exception:
            return str(v)

    return v


def normalise_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return a defensively normalised metadata dict suitable for export.
    """
    if not meta:
        return {}
    return {str(k): _normalise_meta_value(v) for k, v in dict(meta).items()}


def _meta_value_to_text(v: Any) -> str:
    """
    Convert a normalised metadata value into a stable text representation
    for header blocks.
    """
    import json

    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def build_export_meta(
    *,
    export_kind: str,
    user_meta: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build a consistent export metadata dict.

    Parameters
    ----------
    export_kind
        Example values:
        - "single_fit"
        - "batch_fit"
        - "mapping_fit"
        - "mapping_table"
    user_meta
        Optional user-supplied metadata to merge in last.
    **kwargs
        Additional metadata fields.

    Returns
    -------
    dict
        Normalised metadata dict.
    """
    meta = {"export_kind": export_kind}
    for k, v in kwargs.items():
        if v is not None:
            meta[str(k)] = v

    if user_meta:
        meta.update(dict(user_meta))

    return normalise_meta(meta)
    
def params_to_rows(
    *,
    peak_labels: Sequence[str],
    params: Sequence[Number],
    intensity_scale: float = 1.0,
) -> List[FitExportRow]:
    """
    Convert a concatenated [centre, scale, amp]... parameter vector into per-peak rows.

    Your Lorentzian form in RamanFit/PLfit is:
        L(x) = (scale / ((x-centre)^2 + scale^2)) * amp / pi
    so peak height at x=centre is:
        height = amp / (pi * scale)

    intensity_scale:
        Multiply height_norm by this to report values in original units (e.g. counts),
        if fitting was performed in normalised intensity space.
    """
    if len(params) != 3 * len(peak_labels):
        raise ValueError(
            f"Parameter length mismatch: got {len(params)} values, "
            f"expected {3 * len(peak_labels)} for {len(peak_labels)} peaks."
        )

    rows: List[FitExportRow] = []
    for k, name in enumerate(peak_labels):
        i = 3 * k
        centre = _safe_float(params[i + 0])
        scale = _safe_float(params[i + 1])
        amp = _safe_float(params[i + 2])

        fwhm = 2.0 * scale

        # Avoid division by zero / negative scale
        if not math.isfinite(scale) or scale <= 0:
            peak_height_norm = float("nan")
        else:
            peak_height_norm = amp / (math.pi * scale)

        peak_height = peak_height_norm * float(intensity_scale)

        rows.append(
            FitExportRow(
                peak=str(name),
                centre=centre,
                fwhm=fwhm,
                scale=scale,
                amp=amp,
                peak_height_norm=peak_height_norm,
                peak_height=peak_height,
            )
        )
    return rows

def params_to_rows_profiled(
    *,
    peak_labels: Sequence[str],
    params: Sequence[Number],
    profile: str = "lorentzian",
    intensity_scale: float = 1.0,
    xaxis: Optional[Sequence[Number]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert fitted parameter vectors into export rows for either Lorentzian
    or pseudo-Voigt models.

    Returns rows as plain dicts so optional fields such as eta can be included
    without changing the original FitExportRow dataclass.

    Parameters
    ----------
    peak_labels
        Ordered peak labels.
    params
        Concatenated fitted parameters.
    profile
        "lorentzian" or "pvoigt".
    intensity_scale
        Scale factor to convert normalised peak heights back into original units.
    xaxis
        Required for pVoigt peak_height_norm estimation.
    """
    profile = str(profile).lower().strip()

    if profile == "lorentzian":
        return [
            {
                "peak": r.peak,
                "centre": r.centre,
                "fwhm": r.fwhm,
                "scale": r.scale,
                "amp": r.amp,
                "peak_height_norm": r.peak_height_norm,
                "peak_height": r.peak_height,
            }
            for r in params_to_rows(
                peak_labels=peak_labels,
                params=params,
                intensity_scale=intensity_scale,
            )
        ]

    if profile != "pvoigt":
        raise ValueError("profile must be 'lorentzian' or 'pvoigt'")

    if xaxis is None:
        raise ValueError("xaxis is required for pvoigt export.")

    try:
        import numpy as np
        try:
            from .peak_models import single_peak
        except Exception:  # pragma: no cover
            from peak_models import single_peak
    except Exception as exc:
        raise ImportError("Could not import peak_models.single_peak for pVoigt export.") from exc

    p = np.asarray(params, dtype=float).ravel()
    x = np.asarray(xaxis, dtype=float).ravel()

    expected = 4 * len(peak_labels)
    if len(p) != expected:
        raise ValueError(
            f"Parameter length mismatch: got {len(p)} values, expected {expected} "
            f"for {len(peak_labels)} pVoigt peaks."
        )

    rows: List[Dict[str, Any]] = []
    for k, name in enumerate(peak_labels):
        i = 4 * k
        centre = _safe_float(p[i + 0])
        fwhm = _safe_float(p[i + 1])
        amp = _safe_float(p[i + 2])
        eta = _safe_float(p[i + 3])

        block = np.asarray([centre, fwhm, amp, eta], dtype=float)
        y_norm = single_peak(x, block, profile="pvoigt")
        peak_height_norm = float(np.nanmax(y_norm))
        peak_height = float(peak_height_norm * float(intensity_scale))

        rows.append(
            {
                "peak": str(name),
                "centre": centre,
                "fwhm": fwhm,
                "scale": fwhm,
                "amp": amp,
                "eta": eta,
                "peak_height_norm": peak_height_norm,
                "peak_height": peak_height,
            }
        )

    return rows

def write_rows(
    rows: Iterable[FitExportRow],
    out_path: str,
    *,
    delimiter: Optional[str] = None,
    include_header: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    headers: bool = True,
) -> str:
    """
    Write rows to CSV (default) or delimiter-separated TXT/TSV.

    headers:
        Controls whether a metadata header block is written (TXT/TSV only).
        Defaults to True.

    include_header:
        Controls whether the column header row is written (CSV/TXT/TSV).
    """
    out_path = os.fspath(out_path)
    ext = os.path.splitext(out_path)[1].lower()

    if delimiter is None:
        delimiter = "\t" if ext in (".tsv", ".txt") else ","

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fieldnames = [
        "Peak",
        "Centre",
        "FWHM",
        "Scale",
        "Amp",
        "PeakHeight_norm",
        "PeakHeight",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # 1) Optional metadata header block for TXT/TSV only
        meta_norm = normalise_meta(meta)

        if headers and meta_norm and ext in (".txt", ".tsv"):
            # comment-style lines, safe for spreadsheet import
            f.write("# RamanPL_2D fit export\n")
            for k, v in meta_norm.items():
                f.write(f"# {k}: {_meta_value_to_text(v)}\n")
            f.write("#\n")

        # 2) Column headers + data table
        w = csv.writer(f, delimiter=delimiter)
        if include_header:
            w.writerow(fieldnames)

        for r in rows:
            w.writerow([
                r.peak,
                r.centre,
                r.fwhm,
                r.scale,
                r.amp,
                r.peak_height_norm,
                r.peak_height,
            ])

    return out_path

def write_table(
    rows: Iterable[Dict[str, Any]],
    out_path: str,
    *,
    fieldnames: Sequence[str],
    delimiter: Optional[str] = None,
    include_header: bool = True,
    meta: Optional[Dict[str, Any]] = None,
    headers: bool = True,
    meta_in_csv: bool = False,
    meta_prefix: str = "# ",
) -> str:
    """
    Generic table writer for wide-format exports (e.g., mapping results).

    rows:
        Iterable of dict-like rows. Missing keys will be written as empty.

    fieldnames:
        Column order (explicit to guarantee stable reload).

    Metadata header behaviour matches write_rows():
    - TXT/TSV when headers=True
    - CSV as well when meta_in_csv=True

    Metadata values are normalised into export-safe text forms so nested preprocessing
    recipes and pipeline metadata remain readable.
    """
    out_path = os.fspath(out_path)
    ext = os.path.splitext(out_path)[1].lower()

    if delimiter is None:
        delimiter = "\t" if ext in (".tsv", ".txt") else ","

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        meta_norm = normalise_meta(meta)
        write_meta = headers and meta_norm and (ext in (".txt", ".tsv") or (meta_in_csv and ext == ".csv"))
        if write_meta:
            f.write(f"{meta_prefix}RamanPL_2D export\n")
            for k, v in meta_norm.items():
                f.write(f"{meta_prefix}{k}: {_meta_value_to_text(v)}\n")
            f.write(f"{meta_prefix}\n")

        w = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter=delimiter, extrasaction="ignore")
        if include_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_path
