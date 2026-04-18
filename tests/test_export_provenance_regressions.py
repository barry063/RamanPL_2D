import numpy as np

from ramanpl.exporter import build_export_meta, write_table
from ramanpl.Mapping import RamanMapping, PLMapping


def test_build_export_meta_normalises_nested_values():
    meta = build_export_meta(
        export_kind="mapping_fit",
        peak_labels=("A1g", "E2g"),
        user_meta={
            "roi": (0, 1, 2, 3),
            "array_value": np.array([1.0, 2.0, 3.0]),
            "nested": {
                "window": (5, 7),
                "labels": np.array(["a", "b"], dtype=object),
            },
        },
    )

    assert meta["schema_version"] == "0.3.8"
    assert meta["export_kind"] == "mapping_fit"
    assert meta["peak_labels"] == ["A1g", "E2g"]

    # nested normalisation
    assert meta["roi"] == [0, 1, 2, 3]
    assert meta["array_value"] == [1.0, 2.0, 3.0]
    assert meta["nested"]["window"] == [5, 7]
    assert meta["nested"]["labels"] == ["a", "b"]


def test_write_table_txt_writes_metadata_header_block(tmp_path):
    out_path = tmp_path / "export_table.txt"

    rows = [
        {"x": 0, "y": 0, "value": 1.23},
        {"x": 1, "y": 0, "value": 4.56},
    ]
    fieldnames = ["x", "y", "value"]
    meta = {
        "coord_mode": "real",
        "peak_labels": ["A1g", "E2g"],
        "preprocessing_backend_resolved": "ramanspy",
    }

    write_table(
        rows,
        str(out_path),
        fieldnames=fieldnames,
        delimiter="\t",
        include_header=True,
        meta=meta,
        headers=True,
    )

    text = out_path.read_text(encoding="utf-8")

    assert text.startswith("# RamanPL_2D export\n")
    assert '# coord_mode: "real"\n' in text
    assert '# peak_labels: ["A1g", "E2g"]\n' in text
    assert '# preprocessing_backend_resolved: "ramanspy"\n' in text
    assert "x\ty\tvalue\n" in text


def test_raman_mapping_fit_export_meta_includes_backend_provenance():
    x = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    spectra = np.ones((1, 2, x.size), dtype=float)

    custom_peaks = {
        "test_peak": ([110.0, 1.0, 0.01], [130.0, 10.0, 100.0]),
    }

    obj = RamanMapping.from_arrays(
        spectra=spectra,
        xdata=x,
        X=2,
        Y=1,
        custom_peaks=custom_peaks,
        data_range=(100.0, 140.0),
        step_size=0.5,
        normalize=False,
        background_remove=False,
        smoothing=False,
        preprocessing_backend="auto",
    )

    obj._preprocess_meta = {
        "preprocessing_backend_info": {
            "requested_backend": "auto",
            "resolved_backend": "ramanspy",
            "fallback_used": False,
        }
    }
    obj.preprocessing_recipe = [
        {"step": "crop", "range": [105.0, 135.0]},
        {"step": "smooth", "window_length": 5, "polyorder": 2},
    ]

    meta = obj._build_mapping_fit_export_meta(
        coord_mode="real",
        scaled=False,
        peak_labels=obj.peak_params,
    )

    assert meta["export_kind"] == "mapping_fit"
    assert meta["map_kind"] == "fit_params"
    assert meta["spectrum_type"] == "Raman"
    assert meta["x_quantity"] == "Raman shift"
    assert meta["x_unit"] == "cm^-1"
    assert meta["coord_mode"] == "real"
    assert meta["step_size"] == 0.5
    assert meta["step_unit"] == "um"

    assert meta["peak_labels"] == ["test_peak"]
    assert meta["peak_profile"] == "lorentzian"
    assert meta["params_per_peak"] == 3
    assert meta["scaled"] is False

    assert meta["preprocessing_backend_requested"] == "auto"
    assert meta["preprocessing_backend_resolved"] == "ramanspy"
    assert meta["preprocessing_recipe"] == [
        {"step": "crop", "range": [105.0, 135.0]},
        {"step": "smooth", "window_length": 5, "polyorder": 2},
    ]


def test_pl_mapping_fit_export_meta_keeps_pl_identity():
    x = np.array([1.80, 1.85, 1.90, 1.95, 2.00])
    spectra = np.ones((1, 1, x.size), dtype=float)

    custom_peaks = {
        "exciton": ([1.84, 0.01, 0.01], [1.96, 0.10, 100.0]),
    }

    obj = PLMapping.from_arrays(
        spectra=spectra,
        xdata=x,
        X=1,
        Y=1,
        custom_peaks=custom_peaks,
        data_range=(1.80, 2.00),
        step_size=0.25,
        normalize=True,
        background_remove=False,
        smoothing=False,
        preprocessing_backend="native",
    )

    obj._preprocess_meta = {
        "preprocessing_backend_info": {
            "requested_backend": "native",
            "resolved_backend": "native",
            "fallback_used": False,
        }
    }
    obj.preprocessing_recipe = [{"step": "crop", "range": [1.82, 1.98]}]

    meta = obj._build_mapping_fit_export_meta(
        coord_mode="pixel",
        scaled=True,
        peak_labels=obj.peak_params,
    )

    assert meta["export_kind"] == "mapping_fit"
    assert meta["map_kind"] == "fit_params"
    assert meta["spectrum_type"] == "Photoluminescence"
    assert meta["x_quantity"] == "Photon energy"
    assert meta["x_unit"] == "eV"
    assert meta["coord_mode"] == "pixel"
    assert meta["step_size"] == 0.25
    assert meta["step_unit"] == "um"

    assert meta["peak_labels"] == ["exciton"]
    assert meta["peak_profile"] == "lorentzian"
    assert meta["params_per_peak"] == 3
    assert meta["scaled"] is True

    assert meta["preprocessing_backend_requested"] == "native"
    assert meta["preprocessing_backend_resolved"] == "native"
    assert meta["preprocessing_recipe"] == [{"step": "crop", "range": [1.82, 1.98]}]


# ---------------------------------------------------------------------------
# Batch export provenance tests (v0.4.4)
# ---------------------------------------------------------------------------

def _write_raman_txt(path, x, y):
    import numpy as np
    data = np.column_stack([x, y])
    np.savetxt(path, data, delimiter="\t", header="wavenumber\tintensity", comments="")


def _fit_batch(tmp_path, n=2, **kwargs):
    """Create and fit a minimal RamanBatch for export tests."""
    from ramanpl.batch import RamanBatch

    x = np.linspace(350.0, 430.0, 81)
    rng = np.random.default_rng(7)
    files = []
    for i in range(n):
        y = 10.0 + 80.0 * np.exp(-0.5 * ((x - 385.0) / 4.0) ** 2) + rng.normal(0, 0.3, x.size)
        p = tmp_path / f"spec_{i}.txt"
        _write_raman_txt(str(p), x, y)
        files.append(str(p))

    custom_peaks = {"test_peak": ([370.0, 1.0, 0.01], [400.0, 15.0, 300.0])}
    batch = RamanBatch(
        files,
        custom_peaks=custom_peaks,
        normalize=False,
        smoothing=False,
        background_remove=False,
        **kwargs,
    )
    batch.fit()
    return batch


def test_raman_batch_export_meta_includes_backend_provenance(tmp_path):
    """Batch TXT export header must contain preprocessing_backend_requested and _resolved."""
    batch = _fit_batch(tmp_path, preprocessing_backend="native")
    out = tmp_path / "batch_out.txt"
    batch.export(str(out), wide=True)

    text = out.read_text(encoding="utf-8")

    assert "preprocessing_backend_requested" in text, (
        "TXT export header is missing 'preprocessing_backend_requested'"
    )
    assert "preprocessing_backend_resolved" in text, (
        "TXT export header is missing 'preprocessing_backend_resolved'"
    )


def test_raman_batch_export_meta_uniform_backend_is_scalar(tmp_path):
    """When all spectra use the same backend, the metadata value must be a scalar string."""
    import json

    batch = _fit_batch(tmp_path, n=3, preprocessing_backend="native")
    out = tmp_path / "batch_uniform.txt"
    batch.export(str(out), wide=True)

    # Extract the header lines (lines starting with #) and parse the value
    header_lines = [
        ln[2:].strip()  # strip leading "# "
        for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.startswith("# ")
    ]

    resolved_line = next(
        (ln for ln in header_lines if ln.startswith("preprocessing_backend_resolved")),
        None,
    )
    assert resolved_line is not None, "Header missing 'preprocessing_backend_resolved'"

    # Value is the JSON after the colon
    _, raw_value = resolved_line.split(":", 1)
    value = json.loads(raw_value.strip())

    # Uniform batch → scalar string, not a list
    assert isinstance(value, str), (
        f"Uniform backend should serialise as a string, got {type(value).__name__}: {value!r}"
    )
    assert value == "native"


def test_raman_batch_txt_export_writes_export_kind(tmp_path):
    """Batch TXT export header must contain export_kind: 'batch_fit'."""
    import json

    batch = _fit_batch(tmp_path, preprocessing_backend="native")
    out = tmp_path / "batch_kind.txt"
    batch.export(str(out), wide=True)

    header_lines = [
        ln[2:].strip()
        for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.startswith("# ")
    ]

    kind_line = next(
        (ln for ln in header_lines if ln.startswith("export_kind")),
        None,
    )
    assert kind_line is not None, "Header missing 'export_kind'"
    _, raw_value = kind_line.split(":", 1)
    assert json.loads(raw_value.strip()) == "batch_fit"