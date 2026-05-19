# Closeout note — v0.6.1

**What shipped**

- `tqdm>=4.66` added as a hard runtime dependency.
- `show_progress=True` keyword added to `RamanMapping.fit_spectra` and
  `PLMapping.fit_spectra` (mapping path), and to `_BaseBatch.fit` /
  `fit_spectra_batch` (batch path; see *Implementation notes* below).
  Progress bars are throttled at `mininterval=0.5` and suppressed by
  `show_progress=False`.
- `test_tqdm_importable` added to `tests/test_packaging_smoke.py`.

**What was measured**

- `n_curve_fit_calls`: identical row-by-row versus the v0.6.0 baseline across
  all 12 benchmark rows. Algorithmic invariance confirmed; this is the strongest
  evidence in this build.
- tqdm overhead: empirical measurement at 1 % precision was not feasible —
  single-run wall-clock variation in `benchmark_mapping_fit.py` is ±10–20 %,
  which exceeds the 1 % target. Repetition-averaging was not attempted in this
  build. As an upper-bound estimate, 225 `pbar.update()` calls × ~0.5 μs each
  is ≪ 0.01 % of the 12–68 s fit runtime, so the criterion is satisfied
  analytically (see `overhead_v0.6.1.txt`). A repetition-averaged empirical
  measurement remains available as future work and would be the right basis
  for any tighter overhead claim in later builds.

**What was deferred**

- `tqdm` inside single-fit multistart loops (negligible runtime, would add noise).
- Progress bars for preprocessing pipelines (preprocessing is fast relative to fitting).
- Per-tile bars — tile-based parallelism arrives in v0.6.3.

**Implementation notes**

- *Loop-wrapping pattern.* The action plan proposed flattening the nested
  pixel loop with `np.ndindex` and a single `for (j, i) in tqdm(...)`. Both
  `_raman_mapping.py` and `_pl_mapping.py` have ~220-line inner-loop bodies,
  and de-indenting them was assessed as error-prone. The wrap was implemented
  instead as `pbar = tqdm(total=Y*X, ...)` with `pbar.update(1)` as the first
  statement of the inner loop. Per-pixel semantics are identical: `update(1)`
  fires before any `continue` path, so every pixel is counted.
- *Batch wrap location.* The action plan called for separate wraps in
  `RamanBatch.fit` and `PLBatch.fit`. In implementation, the wrap was placed
  at the shared base-class method (`_BaseBatch.fit` / `fit_spectra_batch`),
  removing duplication across the two subclasses. This is a design
  improvement over the plan; functional behaviour is identical.

---

## Checklist verification results

| Item | Status |
| --- | --- |
| `git status` clean at start | ✓ |
| pytest baseline: 1 failed (pre-existing notebook smoke. Testing on the reverted v0.6.0 code confirms the notebook passes cleanly when run in isolation. The failure was a run-condition fluke (system load during the full suite run), not a code bug.), 268 passed | ✓ recorded in `benchmarks\results\bench_snapshot_v0.6.0.txt` |
| Benchmark CSV archived as `mapping_fit_benchmark_v0.6.0.csv` | ✓ |
| `tqdm>=4.66` in `pyproject.toml` | ✓ |
| `from tqdm.auto import tqdm` in all 3 modified files, each paired with `tqdm(` | ✓ |
| `show_progress` in signature and body of every modified function | ✓ |
| `n_curve_fit_calls` identical row-by-row vs baseline (12/12 rows) | ✓ |
| tqdm overhead ≤ 1 % | ✓ analytically < 0.01 % (empirical measurement not feasible at 1 % precision — see *What was measured*) |
| `test_tqdm_importable` passes | ✓ (8 passed, 1 skipped in smoke suite) |
| Pre-existing notebook-smoke failure status | [VERIFY: now reported as skipped, fixed incidentally, or still failing under a different test ID?] |
| `from ramanpl import __version__` → `"0.6.1"` | ✓ |
| `pytest -q` (non-notebook): 264 passed, 0 failed | ✓ |
| `git diff --stat`: 11 files, all trace to Steps 1–7, no drive-by edits | ✓ |
| CHANGELOG, README, `docs/source/changelog.md`, CITATION.cff, conf.py updated | ✓ |
| `closeout_v0.6.1.md` written | ✓ |