# Baseline auto-tuning

The `autotune_baseline` / `apply_choice` workflow is an opt-in diagnostic
(v0.6.2+) available on all four fit classes: `RamanMapping`, `PLMapping`,
`RamanFit`, and `PLfit`.

It scores a configurable grid of baseline candidates on a **representative
spectrum** (a seed pixel for mapping; the single spectrum for single-fit),
ranks them by post-fit RMSE, and lets you commit the winner. The workflow
never runs silently during `fit_spectra` / `fit_spectrum`.

## Quick start

### Mapping

```python
from ramanpl import RamanMapping

mapping = RamanMapping.from_arrays(...)

# Step 1 — score the grid on pixel (row=5, col=3)
result = mapping.autotune_baseline(seed_coord=(5, 3), plot=True)

# Step 2 — inspect the ranking
for entry in result.ranking[:5]:
    print(f"{entry['method']:8s}  {entry['kwargs']}  RMSE={entry['rmse']:.4f}")

# Step 3 — commit the winner and re-run fits
mapping.apply_choice(result.winner)
mapping.fit_spectra()
```

### Single-fit

```python
from ramanpl import RamanFit

fit = RamanFit(spectra=y, wavenumber=x, materials=["WS2"],
               background_remove=True)

result = fit.autotune_baseline(methods=["poly", "airpls"], plot=False)
fit.apply_choice(result.winner)
fit.fit_spectrum()
```

## API reference

### `autotune_baseline` (mapping classes)

```python
mapping.autotune_baseline(
    *,
    seed_coord: tuple[int, int],   # (row, col) — row-major
    methods: list[str] | None = None,
    lam_grid: list[float] | None = None,
    plot: bool = True,
    fit_spectrum_kwargs: dict | None = None,
) -> BaselineAutotuneResult
```

**Does NOT modify the object.** Returns a `BaselineAutotuneResult`.

### `autotune_baseline` (single-fit classes)

Same signature but without `seed_coord` (the object holds one spectrum).

### `apply_choice`

```python
obj.apply_choice(choice: dict) -> None
```

Commits a baseline spec dict (e.g. `{"method": "airpls", "lam": 1e5,
"niter": 50}`) to `obj.preprocessing`. For mapping objects, invalidates
the preprocessed-cube cache so the next `fit_spectra()` uses the new
baseline. For single-fit objects, re-applies the full pipeline from the
pristine raw spectrum.

Raises `ValueError` if the pipeline has zero or more than one
`BaselineSubtract` steps.

### `BaselineAutotuneResult`

| Field | Type | Description |
|---|---|---|
| `ranking` | `list[dict]` | Candidates sorted ascending by RMSE. Each dict has `method`, `kwargs`, `rmse`. |
| `winner` | `dict` | Baseline spec dict of the top-ranked candidate. Pass to `apply_choice()`. |
| `seed_coord` | `tuple[int,int] \| None` | Pixel that was scored (mapping only). |
| `figure` | `Figure \| None` | Comparison figure (when `plot=True`). |
| `meta` | `dict` | Provenance (n_candidates, methods_scanned, seed_coord). |

## Default baseline grid (24 candidates)

| Method | Parameter sweep | Candidates |
|---|---|---|
| `asls` | `lam ∈ {1e3, 1e4, 1e5, 1e6, 1e7}`, `p=0.001`, `niter=20` | 5 |
| `arpls` | `lam ∈ {1e3, 1e4, 1e5, 1e6, 1e7}`, `niter=50` | 5 |
| `airpls` | `lam ∈ {1e3, 1e4, 1e5, 1e6}`, `niter=50` | 4 |
| `poly` | `poly_order ∈ {1, 2, 3, 4, 5}` | 5 |
| `gaussian` | `gaussian_sigma ∈ {5, 10, 20, 50, 100}` | 5 |
| **Total** | | **24** |

Restrict the scan with `methods=["poly", "gaussian"]` or override the lam
sweep with `lam_grid=[1e4, 1e5, 1e6]`.

## Provenance in exports

When `apply_choice` has been called, a `baseline_autotune` block is
included in TXT/CSV export metadata:

```
baseline_autotune:
  methods: [asls, arpls, airpls, poly, gaussian]
  n_candidates: 24
  seed_coord: [5, 3]
  winner: {method: airpls, lam: 100000.0, niter: 50}
  winner_rmse: 0.0234
  ranking_top5: [...]
```

When `apply_choice` has *not* been called, the block is absent (no
spurious key in default exports).

## Constraints

- `autotune_baseline()` is **read-only** — it never modifies the object.
- Only `apply_choice()` writes to `self`.
- `apply_choice()` raises `ValueError` if the pipeline contains zero or
  more than one `BaselineSubtract` steps.
- No new keywords on `fit_spectra` or `fit_spectrum`.
- No new runtime dependencies.

## See also

- {doc}`preprocessing` — pipeline architecture
- {doc}`low_snr_advisory` — guidance for low-SNR spectra
- {doc}`mapping` — full mapping workflow
