# Advice for lower-SNR or more challenging datasets

The demo above used a high-quality MoS₂/WS₂ map where the existing adaptive multistart already achieves 100% success, so the proposal fallback is never triggered. If your data has lower SNR, overlapping peaks, weak Raman signals, or a varying background, the steps below will systematically improve fitting before and after enabling proposals.

Work through the steps in order — each one reduces the problem before the next:

1. Find a high-SNR seed pixel and inspect it first
2. Validate bounds on a single spectrum before committing to the full map
3. Trial preprocessing settings (smoothing, baseline method)
4. Tighten peak bounds based on single-spectrum results
5. Choose between Lorentzian and pseudo-Voigt profiles deliberately
6. Increase multistart attempts for genuinely hard pixels
7. Tune `prominence_rel` if the proposal fallback still misses peaks

---

## Step 1 — Identify your best pixel before fitting anything

The most common cause of widespread map failures is a poor `seed_coord` for `warm_start`. The seed pixel is fitted first and its result is propagated to neighbours; if the seed fails or produces a poor fit, the warm-start chain degrades quickly across the map.

Use a raw intensity heatmap at the expected peak wavenumber to find a bright, representative region **before** calling `fit_spectra()`:

```python
# Plot raw intensity at the expected position of your strongest peak
raman_map.plot_heatmap(
    data_type='specific_intensity',
    specific_wavenumber=383.0,   # change to your strongest expected peak
    filter_range=[0, 1],
)

# Programmatic alternative: find the pixel with maximum integrated intensity
import numpy as np
integrated = raman_map.spectra.sum(axis=2)   # shape (n_y, n_x)
iy, ix = np.unravel_index(integrated.argmax(), integrated.shape)
print(f'Brightest pixel: x={ix}, y={iy}  → use as seed_coord')
```

Avoid edge pixels, substrate-only regions, and pixels near obvious contamination — these are unrepresentative of the material you are mapping.

---

## Step 2 — Validate bounds on a single spectrum with `RamanFit` before the full map

Running the full map only to find that bounds are misspecified wastes significant time. Use `RamanFit` on the seed pixel with the same `custom_peaks` dict first. `plot_fit()` will show immediately whether the fitted curves land on the correct peaks.

```python
from ramanpl import RamanFit

single_fit = RamanFit.RamanFit(
    spectra           = raman_map.spectra[iy, ix, :],
    wavenumber        = raman_map.wavenumber,
    materials         = list(custom_peaks.keys()),
    custom_peaks      = custom_peaks,
    normalize         = True,
    background_remove = True,
    baseline_method   = {"method": "arpls", "lam": 1e4, "niter": 250, "tol": 1e-6},
    smoothing         = False,
    preprocessing_backend = 'auto',
)
single_fit.fit()
single_fit.plot_fit()

# Check bound-sticking: n_params_at_bounds > 0 means at least one
# parameter has hit a bound wall — a reliable sign of misspecified bounds.
ft = single_fit.feature_table()
print(ft[[c for c in ft.columns if c.endswith('_position')]].to_string(index=False))
print(f"Parameters at bounds: {ft['n_params_at_bounds'].values[0]}")
```

Common signs of bounds problems to look for in `plot_fit()`:

| Symptom | Likely cause |
|---------|-------------|
| Fitted centre at the edge of the bounds window | Window does not contain the peak; shift the bounds |
| Fitted amplitude at zero | Peak absent at this pixel, or bounds too wide |
| FWHM at the upper bound | A broad background feature is being fitted as a peak |

---

## Step 3 — Trial preprocessing settings

Preprocessing has a large effect on fitting success for low-SNR data. The two most impactful controls are smoothing and the baseline correction method.

**Smoothing (`smoothing=True`, `smooth_window`):** Savitzky-Golay smoothing reduces high-frequency noise without significantly broadening Raman peaks, which typically have FWHM of 3–20 cm⁻¹. A window of 5–7 points is a reasonable starting point for 532 nm excitation at 1800 gr/mm grating resolution. Check that the smoothed spectrum still shows the expected peak shape — over-smoothing will artificially broaden peaks and bias reported FWHM values upward.

**Baseline method:**

| Method | Best suited for | Key parameter |
|--------|----------------|---------------|
| `arpls` | Smoothly varying fluorescence background | `lam` (smoothness, 10³–10⁶); `niter` |
| `asls` | Similar to `arpls`; slightly more aggressive | `lam`, `p` (asymmetry, 0.001–0.1) |
| `airpls` | Steep or rapidly varying backgrounds | `lam` |
| `poly` | Flat or slowly-varying background; fast | `poly_order` (2–5) |

For MoS₂/WS₂ on Si/SiO₂, `arpls` with `lam=1e4` is a robust default. If the background dominates (e.g. bulk Si substrate), increase `lam` toward 10⁶. If the baseline dips below zero between peaks after correction, decrease `lam`.

A quick visual comparison on the seed pixel, without running the full map, helps select these settings efficiently:

```python
# Construct temporary mapping objects (no full fit) to compare preprocessed spectra.
# fit_spectra with n_starts=1 is used purely to populate _preprocessed_cube_cache.
for label, kwargs in [
    ('arpls lam=1e4, no smoothing',
     dict(smoothing=False,
          baseline_method={"method": "arpls", "lam": 1e4, "niter": 250, "tol": 1e-6})),
    ('arpls lam=1e4, smoothing window=5',
     dict(smoothing=True, smooth_window=5,
          baseline_method={"method": "arpls", "lam": 1e4, "niter": 250, "tol": 1e-6})),
    ('arpls lam=1e5, smoothing window=5',
     dict(smoothing=True, smooth_window=5,
          baseline_method={"method": "arpls", "lam": 1e5, "niter": 250, "tol": 1e-6})),
]:
    m = Mapping.RamanMapping('Mapping Raman Sample.wdf', custom_peaks,
                             data_range=(300, 460), normalize=True,
                             background_remove=True, step_size=0.5, **kwargs)
    m.fit_spectra(warm_start=False, fit_spectrum_kwargs=dict(n_starts=1))
    spec = m._preprocessed_cube_cache[iy, ix, :]
    plt.plot(m.wavenumber, spec, label=label)

plt.axhline(0, color='grey', lw=0.5, ls='--')
plt.xlabel('Raman shift (cm⁻¹)')
plt.legend(fontsize=7)
plt.title('Preprocessing comparison — seed pixel')
plt.tight_layout()
plt.show()
```

Criteria for a good preprocessed spectrum: baseline sits near zero between peaks (not dipping below), peaks are clearly resolved above the noise floor, and peak shapes are not visibly broadened.

---

## Step 4 — Tighten peak bounds based on single-spectrum results

Overly wide bounds are one of the most common causes of multistart failures: the optimiser has a larger search space and is more likely to converge on a local minimum that does not correspond to the correct peak. Use the single-spectrum result from Step 2 to set tighter windows before the full map run.

A practical rule: set the centre bounds to `fitted_centre ± 3 × FWHM`. This is wide enough to accommodate realistic spatial variation across the map but narrow enough to exclude neighbouring peaks or background features.

```python
ft = single_fit.feature_table()
for name in custom_peaks:
    c    = ft[f'{name}_position'].values[0]
    fwhm = ft[f'{name}_fwhm'].values[0]
    margin = 3 * fwhm
    lb_old, ub_old = custom_peaks[name]
    print(f'{name}: centre window  [{lb_old[0]}, {ub_old[0]}]'
          f'  →  suggested [{c - margin:.1f}, {c + margin:.1f}]')
```

> **Caution:** If the map spans regions with genuinely different peak positions (e.g. a strain gradient, a layer-number boundary, or a heterostructure interface), tightening bounds based on a single seed pixel will cause failures precisely where the spectral variation is scientifically interesting. In these cases, keep the bounds wider and rely on `n_starts` and the proposal fallback rather than tight windows.

---

## Step 5 — Choose between Lorentzian and pseudo-Voigt deliberately

The pseudo-Voigt profile adds a mixing parameter η (0 = pure Gaussian, 1 = pure Lorentzian), which is physically motivated when phonon lifetime broadening and inhomogeneous broadening contribute comparably to the lineshape — common in disordered, strained, or defective 2D materials.

However, for low-SNR data the extra parameter is a liability:

| Scenario | Recommended profile |
|----------|---------------------|
| High SNR, lineshape asymmetry physically meaningful | pseudo-Voigt |
| Low SNR, peaks barely above noise floor | Lorentzian (3 params, more robust) |
| Peaks strongly asymmetric (e.g. 2LA(M) in MoS₂) | pseudo-Voigt |
| Quick mapping survey; peak position is the primary output | Lorentzian |

After fitting with pseudo-Voigt, check `n_params_at_bounds` — if η is consistently hitting 0 or 1 across the map, the data cannot constrain the mixing and a Lorentzian will be both faster and more reliable.

---

## Step 6 — Increase multistart attempts for genuinely hard pixels

`n_starts` controls how many random starting points are tried per pixel before the fitter gives up. Increasing it comes at a direct runtime cost, so pair it with tight bounds (Step 4) to keep the search tractable.

`width_penalty` adds a regularisation term that penalises unphysically narrow fits, preventing the optimiser from collapsing onto a single noise spike — a common failure mode at low SNR. A value of 0.01–0.05 is a reasonable starting range.

`p0_strategy='jitter'` randomises the starting point around the current best guess at each retry, rather than drawing fully at random from within the bounds. This is better suited to tight bounds windows and low-SNR data because retries stay in a physically plausible region of parameter space.

```python
raman_map.fit_spectra(
    warm_start = True,
    seed_coord = (ix, iy),       # brightest pixel from Step 1
    fit_spectrum_kwargs = dict(
        n_starts      = 8,       # increase to 16 for very hard maps
        p0_strategy   = 'jitter',
        random_state  = 42,
        width_penalty = 0.02,
        use_peak_proposals = True,
    )
)
raman_map.fit_summary()
raman_map.plot_residual_distribution(filter_threshold=0.05)
```

---

## Step 7 — Tune `prominence_rel` for low-SNR spectra

If the proposal fallback is still missing peaks, `prominence_rel` may need to be lowered. The default of 0.05 (5% of spectrum range) is safe for high-SNR data but may miss real peaks that do not stand clearly above the noise floor after baseline correction.

Lowering `prominence_rel` increases sensitivity but also increases false detections. The `width_min_pts` parameter provides a complementary guard: noise spikes typically span only 1–2 data points, whereas a real Raman peak at typical spectral resolution spans at least 3–5 points. A combination of `prominence_rel=0.02–0.03` and `width_min_pts=3–4` is usually the right balance for noisy data.

Sweep the parameter on the seed pixel before committing to a full map run:

```python
from ramanpl.single_fit.initialisation import propose_peaks

spectrum   = raman_map._preprocessed_cube_cache[iy, ix, :]
wavenumber = raman_map.wavenumber

for prom in [0.10, 0.05, 0.03, 0.02]:
    props = propose_peaks(spectrum, wavenumber,
                          n_peaks=len(custom_peaks),
                          prominence_rel=prom, width_min_pts=3)
    centres = [f"{p['centre']:.1f}" for p in props]
    print(f'prominence_rel={prom:.2f}  →  {len(props)} proposal(s): {centres}')
```

Decision rule: choose the lowest `prominence_rel` that detects all expected peaks without returning more proposals than `n_peaks`. If no value achieves this on the seed pixel, revisit the preprocessing settings in Step 3 — proposals operate on the preprocessed spectrum and their quality is directly limited by baseline correction quality.

---

## Quick-reference checklist

| Step | Action | Key parameter | First-try value |
|------|--------|---------------|-----------------|
| 1 | Find seed pixel | `seed_coord` | `spectra.sum(axis=2).argmax()` |
| 2 | Single-spectrum validation | `RamanFit.plot_fit()` | Check `n_params_at_bounds` |
| 3 | Enable smoothing | `smooth_window` | 5 points |
| 3 | Adjust baseline | `lam` | Increase if baseline dips below zero |
| 4 | Tighten bounds | Centre window | fitted_centre ± 3 × FWHM |
| 5 | Profile choice | Lorentzian vs pseudo-Voigt | Lorentzian for low SNR |
| 6 | More retries | `n_starts` | 8 |
| 6 | Penalise noise spikes | `width_penalty` | 0.02 |
| 6 | Smarter retry | `p0_strategy` | `'jitter'` with tight bounds |
| 7 | Proposal sensitivity | `prominence_rel` | 0.03 for noisy data |

> **Scientific caution:** Steps 3–7 all involve trade-offs. Smoothing reduces noise but will bias FWHM values if applied aggressively. Tightening bounds based on a single seed pixel will cause failures in regions where the peak genuinely shifts across the map. Always verify the final fit quality with `plot_residual_distribution()` and spot-check suspect pixels with `plot_spectrum_fit()` before interpreting mapped results physically.
