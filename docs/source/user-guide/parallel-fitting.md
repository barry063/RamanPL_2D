# Parallel fitting with `n_jobs` (v0.6.4+)

Both `RamanMapping.fit_spectra` and `PLMapping.fit_spectra` accept an `n_jobs`
keyword that distributes the pixel row loop across multiple worker processes via
[joblib](https://joblib.readthedocs.io).

## Basic usage

```python
# Default: serial (byte-parity with v0.6.3)
mapping.fit_spectra()

# Parallel — 4 workers, no warm-start
mapping.fit_spectra(n_jobs=4)

# Parallel — warm-start requires row_reset=True
mapping.fit_spectra(
    warm_start=True,
    seed_coord=(10, 10),
    row_reset=True,    # required when n_jobs > 1
    n_jobs=4,
)
```

## Hard constraints

| Condition | Behaviour |
|---|---|
| `n_jobs=1` (default) | Serial path, identical output to v0.6.3 |
| `n_jobs > self.Y` | Clamped to `self.Y` with `UserWarning` |
| `n_jobs > 1`, `warm_start=True`, `row_reset=False` | `ValueError` — unsafe combination |
| Non-integer or `n_jobs < 1` | `ValueError` |
| `cluster_seeds=True`, `n_jobs > 1` | `ValueError` — cluster seeding is serial-only in v0.6.5 |

## `cluster_seeds` is serial-only (v0.6.5)

`cluster_seeds=True` requires `n_jobs=1`. Passing `n_jobs > 1` with `cluster_seeds=True` raises `ValueError` with a message naming both resolutions: use `n_jobs=1`, or set `cluster_seeds=False`.

Two-phase parallel cluster dispatch is deferred to a later release.

## Why row-band parallelism?

The cube is split into contiguous row bands — one band per worker.  Each band
is fitted independently.  Fitted arrays from all bands are merged back into the
parent object after `Parallel` returns.

When `warm_start=False` (default) every pixel starts from the same `p0_base`
midpoint, so bands are truly independent and results are byte-identical to the
serial path regardless of `n_jobs`.

When `warm_start=True` + `row_reset=True`, warm-start propagation is
**intra-band only**: the last pixel of one band does not seed the first pixel of
the next.  This is the safe parallel warm-start mode.

## Performance

Benchmark on `extended_15x15` (225 pixels, 150 spectral points), Windows 11,
loky backend:

| n_starts | n_jobs=1 | n_jobs=2 | n_jobs=4 | speedup (4×) |
|---|---|---|---|---|
| 1 | 15.3 s | 8.5 s | 6.3 s | 2.42× |
| 4 | 47.3 s | 30.4 s | 18.4 s | 2.58× |

Sub-linear scaling is expected: loky process creation overhead dominates for
small cubes.  Larger cubes (e.g. 50×50 pixels) benefit more.

## Backend

Workers use `joblib` with `backend="loky"` (multiprocessing, avoids GIL).
The worker functions (`_raman_fit_band`, `_pl_fit_band`) are module-level
callables in `src/ramanpl/mapping/_parallel.py` — necessary for loky pickling.
