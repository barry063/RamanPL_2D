# v0.6.6 Closeout — API Freeze, Validation, Docs

**Build type:** Consolidation pause — no new algorithms, no schema changes.  
**Branch:** `v0.6.6-dev`  
**Started:** 2026-06-02  
**Completed:** 2026-06-02  
**Status:** COMPLETE

---

## §0 Decision Gate Record

| Gate | Resolved value |
|---|---|
| **0.1 Pre-edit ref** | HEAD `952b25d` — `doc(v0.6.5): CHANGELOG and closeout_v0.6.5.md updated`. Working tree clean at build start. |
| **0.2 `show_progress` default** | `True` on `RamanMapping.fit_spectra` (line 403), `PLMapping.fit_spectra` (line 368), and `fit_spectra_batch` in `src/ramanpl/batch.py` (line 372). README v0.6.1 row also states `True` — source and README agree. |
| **0.3 `n_jobs` / `cluster_seeds`** | `n_jobs=1`, `cluster_seeds=False` on both mapping classes — consistent with `closeout_v0.6.5.md`. |
| **0.4 v0.6.0 reference** | `benchmarks/results/v0.6.0_validation.csv` (4663 B, dated 2026-05-20) and `benchmarks/results/v0.6.0_validation_summary.json` exist. No regeneration required. |
| **0.5 gitignore** | `.gitignore` line 43 blanket-ignores `benchmarks/results/*.csv`. `v0.6.0_validation.csv` was committed by force-add. `v0.6.6_validation.csv` and `v0.6.6_validation_summary.json` must be committed the same way (force-add or `!` exception). |
| **0.6 Frozen vocabulary** | Suffix set: `{"_position", "_fwhm", "_peak_height", "_peak_height_norm", "_separation", "_ratio"}`. QA cols: `["rmse", "ok", "n_starts", "n_params_at_bounds"]`. Both derived from `tests/test_api_stability.py`. |
| **0.7 File existence** | `docs/source/user-guide/low_snr_advisory.md` exists (update, not create). Autotune test files: `test_autotune_baseline_mapping.py`, `test_autotune_baseline_single_fit.py`. All required test files present. |

**Discrepancy corrected:** `checklist.md` §2 originally stated `show_progress=False`; corrected to `True` per Gate 0.2 before any doc or test was written.

---

## Known Constraints Carried from v0.6.5

- `cluster_seeds=True` remains serial-only; raises with `n_jobs > 1`.
- Synthetic benchmark suite did not demonstrate call-count reduction for cluster seeding. This limitation is documented in validation report and docs; no speedup claim is made.

---

## Files Changed

### New

- `benchmarks/validation_v0.6.6_vs_v0.6.0.py`
- `benchmarks/results/v0.6.6_validation.csv`
- `benchmarks/results/v0.6.6_validation_summary.json`
- `docs/source/validation/v0.6.6.md`
- `closeout_v0.6.6.md` (this file)

### Modified

- `checklist.md` — corrected show_progress default typo
- `tests/test_api_stability.py`
- `docs/source/api-stability.md`
- `docs/source/user-guide/mapping.md`
- `docs/source/user-guide/batch.md`
- `docs/source/user-guide/low_snr_advisory.md`
- `docs/source/user-guide/baseline-autotune.md`
- `docs/source/user-guide/parallel-fitting.md` (no content change needed — already complete)
- `docs/source/changelog.md`
- `CHANGELOG`
- `README.md` — v0.6.6 row marked shipped 2026-06-02

---

## Verification Commands

```bash
pytest tests/test_api_stability.py -q
pytest tests/test_parallel_fit_mapping.py -q
pytest tests/test_cluster_seed_helpers.py -q
pytest tests/test_cluster_seed_fit_mapping.py -q
pytest tests/test_autotune_baseline_mapping.py tests/test_autotune_baseline_single_fit.py -q
pytest tests/test_release_benchmark_smoke.py -q
python benchmarks/validation_v0.6.6_vs_v0.6.0.py
python -c "import ramanpl; import ramanpl.mapping"
```

## Verification Results

97 tests passed, 1 warning (pre-existing overflow in exp in baselineAPI.py, not a failure).
Validation harness: all 7 gates passed (2026-06-02).

---

## Known Limitations

- `cluster_seeds=True` is serial-only; no parallel dispatch shipped.
- Synthetic call-count reduction for cluster seeding not demonstrated on homogeneous cubes.
- No fitting algorithms, preprocessing algorithms, peak models, or export schemas changed in this build.
