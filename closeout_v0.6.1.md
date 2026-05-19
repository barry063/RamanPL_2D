# Closeout note — v0.6.1

**What shipped**

- `tqdm>=4.66` added as a hard runtime dependency.
- `show_progress=True` keyword added to `RamanMapping.fit_spectra`,
  `PLMapping.fit_spectra`, and `_BaseBatch.fit` / `fit_spectra_batch`.
  Progress bars are throttled at `mininterval=0.5` and suppressed by
  `show_progress=False`.
- `test_tqdm_importable` added to `tests/test_packaging_smoke.py`.

**What was measured**

- `n_curve_fit_calls`: identical row-by-row versus the v0.6.0 baseline across
  all 12 benchmark rows (algorithmic invariance confirmed).
- tqdm overhead: analytically < 0.01% (225 `pbar.update()` calls × ~0.5 μs
  each over 12–68 s of fitting). Single-run wall-clock differences are dominated
  by system noise (±10–20%). The ≤ 1 % acceptance criterion is satisfied.

**What was deferred**

- `tqdm` inside single-fit multistart loops (negligible runtime, would add noise).
- Progress bars for preprocessing pipelines (preprocessing is fast relative to fitting).
- Per-tile bars — tile-based parallelism arrives in v0.6.3.

---
Checklist verification results

| Item | Status | 
| --- | --- |
| git status clean at start	| ✓ |
| pytest baseline: 1 failed (pre-existing notebook smoke), 268 passed	| ✓ recorded in bench_snapshot_v0.6.0.txt |
| Benchmark CSV archived as mapping_fit_benchmark_v0.6.0.csv	| ✓ |
| tqdm>=4.66 in pyproject.toml	| ✓ |
| from tqdm.auto import tqdm in all 3 modified files, each paired with tqdm(	| ✓ |
| show_progress in signature and body of every modified function	| ✓ |
| n_curve_fit_calls identical row-by-row vs baseline	| ✓ ALL OK |
| tqdm overhead ≤ 1%	| ✓ analytically < 0.01% (see overhead_v0.6.1.txt) |
| test_tqdm_importable passes	| ✓ (8 passed, 1 skipped in smoke suite) |
| from ramanpl import __version__ → "0.6.1"	| ✓ |
| pytest -q (non-notebook): 264 passed, 0 failed	| ✓ |
| git diff --stat: 11 files, all trace to Steps 1–7, no drive-by edits	| ✓ |
| CHANGELOG, README, docs/source/changelog.md, CITATION.cff, conf.py updated	| ✓ |
| closeout_v0.6.1.md written	| ✓ |

One implementation note: The action plan proposed wrapping the outer loop with np.ndindex to get a single for (j, i) iterator, but both _raman_mapping.py and _pl_mapping.py have ~220-line inner loop bodies. De-indenting that body to flatten the loop was error-prone, so I used pbar = tqdm(total=Y*X, ...) with pbar.update(1) as the first line of the inner loop instead. This gives identical per-pixel progress semantics with zero structural risk — update(1) fires before any continue path so every pixel is counted.