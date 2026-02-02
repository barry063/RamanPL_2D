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
    height_norm: float
    height_scaled: float


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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
            height_norm = float("nan")
        else:
            height_norm = amp / (math.pi * scale)

        height_scaled = height_norm * float(intensity_scale)

        rows.append(
            FitExportRow(
                peak=str(name),
                centre=centre,
                fwhm=fwhm,
                scale=scale,
                amp=amp,
                height_norm=height_norm,
                height_scaled=height_scaled,
            )
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
        "Height_norm",
        "Height_scaled",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # 1) Optional metadata header block for TXT/TSV only
        if headers and meta and ext in (".txt", ".tsv"):
            # comment-style lines, safe for spreadsheet import (they will appear as first rows)
            f.write("# RamanPL_2D fit export\n")
            for k, v in meta.items():
                f.write(f"# {k}: {v}\n")
            f.write("#\n")  # blank comment separator

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
                r.height_norm,
                r.height_scaled,
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

    Metadata header behaviour matches write_rows(): TXT/TSV only when headers=True.
    """
    out_path = os.fspath(out_path)
    ext = os.path.splitext(out_path)[1].lower()

    if delimiter is None:
        delimiter = "\t" if ext in (".tsv", ".txt") else ","

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        write_meta = headers and meta and (ext in (".txt", ".tsv") or (meta_in_csv and ext == ".csv"))
        if write_meta:
            f.write(f"{meta_prefix}RamanPL_2D export\n")
            for k, v in meta.items():
                f.write(f"{meta_prefix}{k}: {v}\n")
            f.write(f"{meta_prefix}\n")

        w = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter=delimiter, extrasaction="ignore")
        if include_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_path
