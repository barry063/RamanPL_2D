# RamanPL_2D

**RamanPL_2D** is a Python toolkit for the analysis and visualisation of **Raman** and **photoluminescence (PL)** spectra in two-dimensional materials. It extracts peak positions, intensities, and FWHM values and supports single-spectrum fitting, batch processing, and spectral mapping. The package is designed for reproducible spectroscopy workflows and exports provenance metadata alongside fitted results. An optional [RamanSPy](https://github.com/baettigph/ramanspy) backend is supported for Raman preprocessing pipelines.

---

## Key features

* Single-spectrum and batch fitting (Lorentzian, pseudo-Voigt)
* Spectral mapping with heatmaps and derived maps (peak separation, intensity ratios)
* Modular preprocessing pipelines with `native`, `auto`, and `ramanspy` backend modes
* Import from `.wdf` (Renishaw WiRE) and `.txt`
* Export to `.csv` / `.txt` with full provenance metadata
* CI-validated packaging, documented public API, and canonical example notebooks

---

## Installation

Base install (no optional dependencies):

```
pip install -e .
```

With optional RamanSPy preprocessing backend:

```
pip install -e ".[ramanspy]"
```

With optional ML utilities (PCA, k-means clustering on feature tables):

```
pip install -e ".[ml]"
```

Requires Python ≥ 3.9.

---

## Quickstart

```
from ramanpl import RamanFit

raman_fit = RamanFit.RamanFit(
    spectra=spectra,
    wavenumber=wavenumber,
    materials=["MoS2"],
    smoothing=True,
    background_remove=True,
    baseline_method={"method": "poly", "poly_order": 3},
    preprocessing_backend="auto",
)
raman_fit.fit()
raman_fit.export("results.txt")
```

See [`example-usage/`](example-usage) for full notebooks.

---

## Documentation

For local documentation build:

```
pip install -r docs/requirements.txt
pip install -e .
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser.

Key pages:

* [Installation](docs/source/installation.md)
* [Quickstart](docs/source/quickstart.md)
* [Backend behaviour](docs/source/user-guide/backend-behaviour.md)
* [API reference](docs/source/api/index.rst)
* [Canonical notebook examples](docs/source/examples/canonical-notebooks.md)

---

## Backend support summary

| Workflow | `native` | `auto` | `ramanspy` |
| --- | --- | --- | --- |
| Raman + supported pipeline | native | ramanspy (if installed) | ramanspy |
| Raman + unsupported step | native | native (fallback) | raises error |
| PL (any pipeline) | native | native | native |

Supported pipeline steps for RamanSPy: `CropByRange`, `SmoothSavGol`, `BaselineSubtract` with `poly`, `asls`, `airpls`, `arpls`. Gaussian baseline and all PL workflows remain native-only.

See [backend behaviour docs](docs/source/user-guide/backend-behaviour.md) for full details.

---

## Peak proposal for failed-fit recovery

| Function / option | Role | Typical call site |
|---|---|---|
| `propose_peaks(spectrum, wavenumber, n_peaks, ...)` | Detect candidate centres and FWHM from a 1-D preprocessed spectrum via `scipy.signal.find_peaks` | Diagnostic; standalone inspection |
| `p0_from_proposals(proposals, peak_profile, current_p0, bounds)` | Convert proposals to a revised p0; bounds-checked, falls back to `current_p0` for any missing or out-of-bounds proposal | Same as above |
| `fit_spectrum_kwargs=dict(use_peak_proposals=True)` | Enable the automatic fallback (v0.5.3 default) — invoked only when all existing retries have failed | Normal mapping runs — no action needed |
| `fit_spectrum_kwargs=dict(use_peak_proposals=False)` | Disable the fallback — reproduces v0.5.2 behaviour exactly | Regression comparisons; debugging |

> **Scientific note:** The proposal fallback does not change the fitting model or the
> optimisation algorithm. It provides a better starting point (`p0`) for
> `scipy.optimize.curve_fit` on pixels that would otherwise be marked failed.
> All reported peak parameters remain the output of deterministic Lorentzian or
> pseudo-Voigt least-squares fitting — traceable to the same physical model as all
> prior builds.

More details and examples in the [peak proposal demo notebook](example-usage/Mapping/Peak_Proposal_Demo.ipynb).

## Development and validation

Release validation commands and pre-tag checklist are in [`RELEASE.md`](RELEASE.md).

CI runs on GitHub Actions (`.github/workflows/ci.yml`) and GitLab (`.gitlab-ci.yml`):

- base tests (no RamanSPy required)
- RamanSPy extras tests
- package build and clean-install smoke
- notebook smoke (with RamanSPy)
- benchmark smoke
- documentation build

---

### v0.6.x — User experience, mapping performance, and visualisation modularisation roadmap

Post-v0.6.0 development honours the v0.5.5–v0.6.0 API freeze and continues to treat deterministic Lorentzian / pseudo-Voigt least-squares fitting as the final authority for reported peak parameters. The v0.6.x series targets three concrete pain points raised after real mapping use: lack of progress feedback on long fits, manually-tuned preprocessing parameters, and single-threaded mapping runtime on multi-core machines. v0.6.6 is a consolidation pause mirroring the v0.5.5 pattern, freezing the new additive surface; v0.6.7 then adds integrated component area to the feature table as a further additive change, before v0.6.8 begins separating plotting from fitting classes. Full deprecation of the embedded plotting methods, and promotion of `ramanpl.visualisation` to a stable public API, is deferred to v0.7.x.

| Version | Scope | Details |
|---|---|---|
| v0.6.1 ✓ | Progress indicators for mapping and batch fits | Shipped 2026-05-19. `tqdm>=4.66` added as a hard dependency. `show_progress=True` keyword added to `RamanMapping.fit_spectra`, `PLMapping.fit_spectra`, and `_BaseBatch.fit` / `fit_spectra_batch`. Progress bars throttled with `mininterval=0.5`. tqdm packaging smoke test added. No changes to fitting algorithms, fitted values, or export schemas. |
| v0.6.2 ✓ | Seed-pixel baseline auto-tuning with visual review | Shipped 2026-05-20. `autotune_baseline()` and `apply_choice()` added to `RamanMapping`, `PLMapping`, `RamanFit`, and `PLfit`. 24-candidate grid over asls/arpls/airpls/poly/gaussian; scored by post-fit RMSE; ranked result returned with optional comparison figure. Provenance block recorded in export metadata. No changes to baseline algorithms, fit_spectra, or fit_spectrum signatures. |
| v0.6.3 ✓ | Autotune API refinement, real-data notebooks, API docs | Shipped 2026-05-22. `method_grids` dict API replaces `methods`/`lam_grid`; Cartesian-product sweep over per-method parameter grids. `methods`/`lam_grid` kept as `DeprecationWarning` shims (removal in v0.6.4). `Baseline_Autotune_Demo.ipynb` extended with real bilayer-graphene data section. Autotune blocks added to `Raman_background-remove.ipynb`. Sphinx autodoc page for `autotune_baseline`, `apply_choice`, `BaselineAutotuneResult`. All 19 test call sites converted to `method_grids=` syntax. No changes to fitting algorithms or export schemas. |
| v0.6.4 ✓ | Parallel mapping fit with row-band warm-start | Shipped 2026-05-24. `n_jobs` keyword added to `RamanMapping.fit_spectra` and `PLMapping.fit_spectra` (default `1` = serial, byte-parity with v0.6.3). When `n_jobs > 1`, row loop distributed via `joblib.Parallel` with loky backend; row-band workers in `_parallel.py`. Unsafe mode (`n_jobs > 1` + `warm_start=True` + `row_reset=False`) raises `ValueError`. `joblib>=1.3` added as hard dependency. Benchmark extended with `n_jobs` axis (2.42× speedup at `n_jobs=4`, `extended_15x15`). `methods`/`lam_grid` deprecation shim removed; `tol` added to `_ALLOWED_PARAMS`. No changes to per-pixel fitting algorithms or scientific output values. |
| v0.6.5 ✓ | Similarity-based seed selection for warm-start | Shipped 2026-05-28. `cluster_seeds` keyword added to `RamanMapping.fit_spectra` and `PLMapping.fit_spectra` (default off, byte-parity with v0.6.4). When enabled, a cheap PCA + k-means partition of the preprocessed cube fits one representative pixel per cluster first; remaining cluster members receive its fitted result as their warm-start `p0`. Serial-only — `cluster_seeds=True` with `n_jobs>1` or `seed_coord=...` raises `ValueError`. Requires the `[ml]` extra; falls back gracefully when absent. Final parameters remain the output of per-pixel least-squares fitting. No changes to baseline algorithms or export schemas. |
| v0.6.6 ✓ | Consolidation pause: API freeze, validation, docs | Shipped 2026-06-02. Freeze contracts for v0.6.1–v0.6.5 additive surface written to `docs/source/api-stability.md` (§9–§13) and enforced by 6 new regression tests in `tests/test_api_stability.py`. Reproducible validation harness `benchmarks/validation_v0.6.6_vs_v0.6.0.py` added; all 7 gates passed (fit-output parity, export-schema stability, call-count sanity, parallel safety, cluster-seed boundary, autotune non-mutation, batch-progress default). Citable report at `docs/source/validation/v0.6.6.md`. Documentation pass across `mapping.md`, `batch.md`, `low_snr_advisory.md`, `baseline-autotune.md`. `show_progress=True` default corrected in checklist. No new algorithms; no changes to fitting, preprocessing, or export schemas. |
| v0.6.7 | Integrated component area in the feature table | - Add per-peak `{peak}_component_area` (processed units) and `{peak}_component_area_norm` (normalised) columns, derived analytically from the fitted area-like amplitude (`amp`) — no numerical integration in production.<br>- Add per-peak `{peak}_component_area_fraction` and a new `area_ratios=` keyword producing `{P1}_{P2}_area_ratio`, mirroring the existing `ratios=` height-ratio mechanism.<br>- Available identically across single-fit, batch, and mapping through the shared `build_feature_row`; single-fit and batch per-peak dicts forward `amp`.<br>- Additive-only: existing columns byte-identical; the QA block remains the final column group. Frozen suffix vocabulary extended (`api-stability.md` §14).<br>- No changes to fitting, preprocessing, or baseline algorithms, or to export schemas; `export()` and plotting unchanged. |
| v0.6.8 | Visualisation module extraction | - Create `src/ramanpl/visualisation/` and move the implementation of `plot_heatmap`, `plot_ratio_heatmap`, `plot_residual_distribution`, `plot_spectrum`, `plot_spectrum_fit`, `plot_waterfall`, `plot_overlay`, and `plot_parameters` into it.<br>- Keep the existing class methods on `RamanMapping`, `PLMapping`, `RamanFit`, `PLfit`, `RamanBatch`, `PLBatch` as thin delegating façades — public method signatures unchanged. `ramanpl.visualisation` itself remains internal in this build.<br>- Add a small standalone preview helper (`ramanpl.visualisation._plot_raw_spectrum`) for quick pre-fit inspection of a single pixel or file.<br>- Acceptance criterion: notebook-smoke output figures unchanged versus v0.6.7 on the same inputs.<br>- Promotion of `ramanpl.visualisation` to a stable public surface, and deprecation of the embedded class methods, is deferred to v0.7.0. |

---

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

---

## Licence

BSD-3-Clause. See [`LICENSE`](LICENSE) for details.

---

## Contact

Hao Yu — <hy377@cam.ac.uk>  
Issues and pull requests: <https://github.com/barry063/RamanPL_2D/issues>
