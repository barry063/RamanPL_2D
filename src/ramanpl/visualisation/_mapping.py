"""Internal fitted-mapping plotting helpers for RamanPL v0.6.8."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ramanpl.dataImporter import DataImporter


def plot_raman_spectrum_fit(self, x, y):
    """Plot raw data and fitting results for a single map point.

    Display logic (PL-equivalent):
        - Fitting is always done in normalised space.
        - normalize=True  -> show normalised, background-removed spectrum + fit
        - normalize=False -> show raw spectrum + fit (+ background overlay if enabled)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import savgol_filter

    if x < 0 or x >= self.X or y < 0 or y >= self.Y:
        raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

    # --- Extract full spectrum
    x_full = np.asarray(self.wavenumber, dtype=float)
    y_full = np.asarray(self.spectra[y, x, :], dtype=float)

    # --- Mask by wavenumber range (cm^-1)
    mask = DataImporter.mask_by_xrange(x_full, self.data_range)
    xdata = x_full[mask]
    raw_intensity = y_full[mask]


    # --- Preprocessing consistent with fitting (except final normalisation)
    proc = raw_intensity.copy()

    if self.smoothing:
        proc = savgol_filter(proc, self.smooth_window, self.smooth_poly)

    if self.background_remove:
        bg_removed = self.remove_background(xdata, proc)
        background = proc - bg_removed
    else:
        bg_removed = proc
        background = None

    # --- Scale used for fitting normalisation
    # Prefer stored scale if available (ensures exact match to fit_spectra)
    scale = None
    if hasattr(self, "norm_scale_map"):
        scale = self.norm_scale_map[y, x]

    if scale is None or not np.isfinite(scale) or scale <= 0:
        scale = np.max(bg_removed)

    if scale <= 0:
        raise ValueError(f"No positive signal at (X={x}, Y={y}); cannot scale fitted curve.")

    # --- Load fitted parameters (normalised space)
    params = np.asarray(self.fitted_params[y, x, :], dtype=float)
    if np.any(np.isnan(params)):
        raise ValueError(f"Fit parameters are NaN at (X={x}, Y={y}). Fit may have failed.")

    model_fn = self._model_dispatch()
    fitted_norm = model_fn(xdata, *params)
    fitted_raw = fitted_norm * scale

    # --- Plot
    plt.figure(figsize=(10, 6))

    if self.normalize:
        # Normalised display (background-removed)
        spectrum_norm = bg_removed / scale
        plt.plot(xdata, spectrum_norm, "k-", label="Background-removed (normalised)")
        plt.plot(xdata, fitted_norm, "g--", linewidth=2, label="Fitted Curve")
        plt.ylabel("Normalised Intensity (a.u.)")
    else:
        # Raw display
        plt.plot(xdata, raw_intensity, "k-", label="Raw Spectrum")

        if self.background_remove:
            plt.plot(xdata, background, "r--", label="Estimated Background")
            plt.plot(xdata, bg_removed, "b-", alpha=0.8, label="Background Removed (smoothed)")

            # Peak-only fit (raw units)
            plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve (peak only)")

            # Overlay vs raw spectrum: (peak fit + estimated background)
            fitted_plus_bg = fitted_raw + background
            plt.plot(xdata, fitted_plus_bg, "-", linewidth=2, label="Fit + Estimated Background")
        else:
            # No background removal → show fit only once
            plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve")

        plt.ylabel("Intensity (a.u.)")

    plt.xlabel("Wavenumber (cm⁻¹)")
    title = f"Spectrum Fit at (X={x}, Y={y})"
    if hasattr(self, "residual_map") and np.isfinite(self.residual_map[y, x]):
        title += f" | RMSE(norm)={self.residual_map[y, x]:.4g}"
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_raman_residual_distribution(
    self,
    filter_threshold=None,
    robust=True,
    p_low=5,
    p_high=95,
    hist_bins=50,
    cmap="inferno"
                                ):
    """
    Visualise spatial distribution of fitting residuals and their histogram.

    Behaviour:
    - If filter_threshold is None:
        show full residual heatmap (optionally robust-scaled) + histogram.
    - If filter_threshold is set:
        ONLY show pixels with residual >= filter_threshold (others masked out) + histogram.

    Notes:
    - residual_map is RMSE computed in the *normalised fit space* (dimensionless).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    residuals = np.asarray(self.residual_map, dtype=float)
    valid = ~np.isnan(residuals)
    residuals_flat = residuals[valid]

    if residuals_flat.size == 0:
        raise ValueError("Residual map contains no valid values to plot.")

    # Determine colour scaling
    if robust:
        vmin = np.percentile(residuals_flat, p_low)
        vmax = np.percentile(residuals_flat, p_high)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    # If thresholding, focus colour scale on the thresholded region for contrast
    if filter_threshold is not None:
        above = residuals_flat[residuals_flat >= filter_threshold]
        if above.size == 0:
            raise ValueError(
                f"No pixels found with residual >= {filter_threshold:g}. "
                "Try lowering filter_threshold."
            )
        vmin = filter_threshold
        vmax_thr = np.percentile(above, 99) if above.size > 5 else np.max(above)
        vmax = vmax_thr if (np.isfinite(vmax_thr) and vmax_thr > vmin) else np.max(above)

    # Layout
    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # ---- Map panel ----
    if filter_threshold is None:
        data_masked = np.ma.masked_invalid(residuals)
        title = "Residual Distribution (higher = worse fit)"
    else:
        keep = (residuals >= filter_threshold) & valid
        data_masked = np.ma.masked_where(~keep, residuals)
        title = f"Residual Distribution (≥ {filter_threshold:g})"

    im = ax_map.imshow(
        data_masked,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("Residual Error (RMSE, normalised)")

    ax_map.set_title(title)
    ax_map.set_xlabel("X Position")
    ax_map.set_ylabel("Y Position")

    # ---- Histogram panel ----
    ax_hist.hist(
        residuals_flat,
        bins=hist_bins,
        orientation="horizontal",
        color="darkred",
        edgecolor="black"
    )
    ax_hist.set_xlabel("Count")
    ax_hist.set_ylabel("Residual RMSE (normalised)")
    ax_hist.set_title("Residual Histogram")

    if filter_threshold is not None:
        ax_hist.axhline(filter_threshold, linestyle="--", linewidth=1)
        upper = residuals_flat[residuals_flat >= filter_threshold]
        y_lo = max(filter_threshold * 0.98, np.min(upper))
        y_hi = np.max(upper)
        if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
            ax_hist.set_ylim(y_lo, y_hi)
    else:
        if vmin is not None and vmax is not None:
            ax_hist.set_ylim(vmin, vmax)

    plt.tight_layout()
    plt.show()


def plot_raman_ratio_heatmap(self, ratio_type='A1g/E2g', cmap='viridis', filter_range=None, x_range=None, y_range=None):
    """Visualize 2D map of peak intensity ratios.
    
    Args:
        ratio_type: 'A1g/E2g' or 'E2g/A1g'
        cmap: Matplotlib colormap name
        filter_range: Data display range [min, max]
        x_range: X display range [start, end]
        y_range: Y display range [start, end]
        
    Raises:
        ValueError: For invalid ratio types or missing peaks
    """
    # Ensure derived maps exist (in case user calls this before fit_spectra)
    if not hasattr(self, "ratio_A1g_E2g") or not hasattr(self, "ratio_E2g_A1g"):
        raise ValueError("Ratio maps not initialised. Run fit_spectra() first.")

    # Choose ratio map
    if ratio_type == 'A1g/E2g':
        data = self.ratio_A1g_E2g
        label = 'A1g/E2g Intensity Ratio'
    elif ratio_type == 'E2g/A1g':
        data = self.ratio_E2g_A1g
        label = 'E2g/A1g Intensity Ratio'
    else:
        raise ValueError("Invalid ratio_type. Choose from 'A1g/E2g' or 'E2g/A1g'.")

    # Filter range: clip outliers only if requested
    if filter_range is not None:
        data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, np.nan)

    # Crop
    if x_range is not None and y_range is not None:
        x_start, x_end = x_range
        y_start, y_end = y_range
        data = data[y_start:y_end+1, x_start:x_end+1]
        x_length = (x_end - x_start + 1) * self.step_size
        y_length = (y_end - y_start + 1) * self.step_size
    else:
        x_length = self.X * self.step_size
        y_length = self.Y * self.step_size

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('gray')

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        cmap=cm,
        vmin=filter_range[0] if filter_range else None,
        vmax=filter_range[1] if filter_range else None,
        extent=[0, x_length, y_length, 0]
    )
    plt.colorbar(im, label=label)
    plt.xlabel("X Position (μm)")
    plt.ylabel("Y Position (μm)")
    plt.title(f"Heatmap of {label}")
    plt.tight_layout()
    plt.show()


def plot_raman_heatmap(self, data_type='position', cmap='viridis', filter_range=None, 
                x_range=None, y_range=None, specific_wavenumber=None, peak_name=None):
    """Visualize 2D map of spectral features.
    
    Args:
        data_type: Plot type ('position', 'intensity', 'specific_intensity', 'distance')
        cmap: Matplotlib colormap name
        filter_range: Data display range [min, max]
        specific_wavenumber: Wavenumber for 'specific_intensity' plots
        peak_name: Peak name for position/intensity plots
        x_range: X display range [start, end]
        y_range: Y display range [start, end]
        
    Raises:
        ValueError: For invalid data types or missing parameters
    """
    # Handle input validation dynamically
    if data_type in ['position', 'intensity']:
        if peak_name is None or peak_name not in self.peak_params:
            raise ValueError(f"Must provide valid peak_name for {data_type} plots")
    elif data_type == 'specific_intensity':
        if specific_wavenumber is None:
            raise ValueError("Must provide specific_wavenumber for intensity at spectra")
    elif data_type == 'distance':
        pass  # No peak_name needed
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    ### Updated in v0.2.4 ###
    # Generate data based on data_type
    if data_type == 'specific_intensity':
        data = np.full((self.Y, self.X), np.nan, dtype=float)

        for j in range(self.Y):
            for i in range(self.X):
                params = self.fitted_params[j, i, :]
                if np.any(np.isnan(params)):
                    continue  # fit failed / not available

                model_fn = self._model_dispatch()
                y_norm = model_fn(np.asarray([specific_wavenumber], dtype=float), *params)[0]

                if self.normalize:
                    # display normalised model intensity
                    data[j, i] = y_norm
                else:
                    # display raw model intensity using stored scale
                    if not hasattr(self, "norm_scale_map") or np.isnan(self.norm_scale_map[j, i]):
                        continue
                    data[j, i] = y_norm * self.norm_scale_map[j, i]

        label = (f'Normalised intensity at {specific_wavenumber} cm⁻¹'
                if self.normalize else
                f'Intensity at {specific_wavenumber} cm⁻¹')
    ### End UPDATED METHOD ###

    elif data_type == 'distance':
        data = self.Peaks_distance
        label = 'A1g - E2g Distance (cm⁻¹)'
    else:
        peak_idx = self.peak_params.index(peak_name)
        data = (self.peak_positions[:, :, peak_idx] if data_type == 'position'
                else self.peak_intensities[:, :, peak_idx])
        label = f'{peak_name} {data_type.capitalize()}'

    # Filter data range
    if filter_range is not None:
        # Replace outliers with filter_range[0] instead of NaN
        data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])
    # If x_range and y_range are specified, only plot data within the specified region
    if x_range is not None and y_range is not None:
        x_start, x_end = x_range
        y_start, y_end = y_range
        data = data[y_start:y_end+1, x_start:x_end+1]
        # Calculate actual length range
        x_length = (x_end - x_start + 1) * self.step_size
        y_length = (y_end - y_start + 1) * self.step_size
    else:
        # Calculate actual length range
        x_length = self.X * self.step_size
        y_length = self.Y * self.step_size

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        cmap=cmap,
        vmin=filter_range[0] if filter_range else None,  # Anchor color scale
        vmax=filter_range[1] if filter_range else None,  # to filter range
        extent=[0, x_length, y_length, 0])
    cbar = plt.colorbar(im, label=label)
    plt.xlabel("X Position (μm)")
    plt.ylabel("Y Position (μm)")
    plt.title(f"Heatmap of {label}")
    plt.show()

def plot_pl_spectrum_fit(self, x, y):
    """Plot raw data and fitting results for a single map point.

    Args:
        x (int): X coordinate (0-indexed)
        y (int): Y coordinate (0-indexed)

    Display logic:
        - Fitting is always done in normalised space.
        - normalize=True  -> show normalised, background-removed spectrum + fit
        - normalize=False -> show raw spectrum + fit (+ background overlay if enabled)
    """
    if x < 0 or x >= self.X or y < 0 or y >= self.Y:
        raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

    # --- Extract spectrum
    x_full = np.asarray(self.xdata, dtype=float)
    y_full = np.asarray(self.spectra[y, x, :], dtype=float)

    # --- Mask by energy range (eV)
    mask = DataImporter.mask_by_xrange(self.xdata, self.data_range)
    xdata = x_full[mask]
    raw_intensity = y_full[mask]

    # --- Preprocessing consistent with fitting (except final normalisation)
    proc = raw_intensity.copy()

    if self.smoothing:
        proc = savgol_filter(proc, self.smooth_window, self.smooth_poly)

    if self.background_remove:
        bg_removed = self.remove_background(xdata, proc)
        background = proc - bg_removed
    else:
        bg_removed = proc
        background = None

    # --- Scale used for fitting normalisation
    scale = None
    if hasattr(self, "norm_scale_map"):
        scale = self.norm_scale_map[y, x]

    if scale is None or not np.isfinite(scale) or scale <= 0:
        scale = np.max(bg_removed)

    if scale <= 0:
        raise ValueError(f"No positive signal at (X={x}, Y={y}); cannot scale fitted curve.")

    # --- Load fitted parameters (normalised space)
    params = np.asarray(self.fitted_params[y, x, :], dtype=float)
    if np.any(np.isnan(params)):
        raise ValueError(f"Fit parameters are NaN at (X={x}, Y={y}). Fit may have failed.")

    model_fn = self._model_dispatch()
    fitted_norm = model_fn(xdata, *params)
    fitted_raw = fitted_norm * scale

    # --- Plot
    plt.figure(figsize=(10, 6))

    if self.normalize:
        # Normalised display (background-removed)
        spectrum_norm = bg_removed / scale
        plt.plot(xdata, spectrum_norm, "k-", label="Background-removed (normalised)")
        plt.plot(xdata, fitted_norm, "g--", linewidth=2, label="Fitted Curve")
        plt.ylabel("Normalised Intensity (a.u.)")

    else:
        # Raw display
        plt.plot(xdata, raw_intensity, "k-", label="Raw Spectrum")

        if self.background_remove:
            # Show background components
            plt.plot(xdata, background, "r--", label="Estimated Background")
            plt.plot(xdata, bg_removed, "b-", alpha=0.8, label="Background Removed (smoothed)")

            # Peak-only fit
            plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve (peak only)")

            # Best overlay vs raw spectrum
            fitted_plus_bg = fitted_raw + background
            plt.plot(xdata, fitted_plus_bg, "-", linewidth=2, label="Fit + Estimated Background")

        else:
            # No background removal → show fit only once
            plt.plot(xdata, fitted_raw, "g--", linewidth=2, label="Fitted Curve")

        plt.ylabel("Intensity (a.u.)")

    plt.xlabel("Energy (eV)")
    plt.title(f"Spectrum Fit at (X={x}, Y={y})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pl_residual_distribution(
    self,
    filter_threshold=None,
    robust=True,
    p_low=5,
    p_high=95,
    hist_bins=50,
    cmap="inferno"
):
    """
    Visualise spatial distribution of fitting residuals and their histogram.

    Behaviour:
    - If filter_threshold is None:
        show full residual heatmap (optionally robust-scaled) + histogram.
    - If filter_threshold is set:
        ONLY show pixels with residual >= filter_threshold (others masked out) + histogram.
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    residuals = np.asarray(self.residual_map, dtype=float)
    valid = ~np.isnan(residuals)
    residuals_flat = residuals[valid]

    if residuals_flat.size == 0:
        raise ValueError("Residual map contains no valid values to plot.")

    # Determine colour scaling
    if robust:
        vmin = np.percentile(residuals_flat, p_low)
        vmax = np.percentile(residuals_flat, p_high)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    # If thresholding, focus colour scale on the thresholded region for contrast
    if filter_threshold is not None:
        above = residuals_flat[residuals_flat >= filter_threshold]
        if above.size == 0:
            raise ValueError(
                f"No pixels found with residual >= {filter_threshold:g}. "
                "Try lowering filter_threshold."
            )
        # Make the colour scale meaningful for the highlighted pixels
        vmin = filter_threshold
        vmax_thr = np.percentile(above, 99) if above.size > 5 else np.max(above)
        vmax = vmax_thr if (np.isfinite(vmax_thr) and vmax_thr > vmin) else np.max(above)

    # Layout
    fig, (ax_map, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # ---- Map panel ----
    if filter_threshold is None:
        # Normal view (like your 2nd image)
        data_masked = np.ma.masked_invalid(residuals)
        title = "Residual Distribution (higher = worse fit)"
    else:
        # Threshold view: only show pixels >= threshold
        keep = (residuals >= filter_threshold) & valid
        data_masked = np.ma.masked_where(~keep, residuals)
        title = f"Residual Distribution (≥ {filter_threshold:g})"

    im = ax_map.imshow(
        data_masked,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax_map)
    cbar.set_label("Residual Error (RMSE)")

    ax_map.set_title(title)
    ax_map.set_xlabel("X Position")
    ax_map.set_ylabel("Y Position")

    # ---- Histogram panel ----
    ax_hist.hist(
        residuals_flat,
        bins=hist_bins,
        orientation="horizontal",
        color="darkred",
        edgecolor="black"
    )
    ax_hist.set_xlabel("Count")
    ax_hist.set_ylabel("Residual RMSE")
    ax_hist.set_title("Residual Histogram")

    # Add a threshold line to the histogram for clarity
    if filter_threshold is not None:
        ax_hist.axhline(filter_threshold, linestyle="--", linewidth=1)
        # If thresholding, it can help to zoom histogram y-range to the upper tail
        upper = residuals_flat[residuals_flat >= filter_threshold]
        y_lo = max(filter_threshold * 0.98, np.min(upper))
        y_hi = np.max(upper)
        if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
            ax_hist.set_ylim(y_lo, y_hi)
    else:
        # Match histogram y-range to map scaling if robust scaling is enabled
        if vmin is not None and vmax is not None:
            ax_hist.set_ylim(vmin, vmax)

    plt.tight_layout()
    plt.show()   


def plot_pl_heatmap(self, data_type='exciton_position', cmap='viridis',
                filter_range=None, specific_xdata=None,
                x_range=None, y_range=None):
    """Visualize 2D map of spectral features."""
    import numpy as np
    import matplotlib.pyplot as plt

    if data_type == 'specific_intensity':
        if specific_xdata is None:
            raise ValueError("For 'specific_intensity' data type, 'specific_xdata' must be provided (in eV).")

        data = np.full((self.Y, self.X), np.nan, dtype=float)

        model_fn = self._model_dispatch()
        for j in range(self.Y):
            for i in range(self.X):
                params = self.fitted_params[j, i, :]
                if np.any(np.isnan(params)):
                    continue  # fit failed

                y_norm = model_fn(np.asarray([specific_xdata], dtype=float), *params)[0]

                if self.normalize:
                    # display normalised model intensity (dimensionless)
                    data[j, i] = y_norm
                else:
                    # display raw model intensity using stored per-pixel scale
                    if (not hasattr(self, "norm_scale_map")) or np.isnan(self.norm_scale_map[j, i]):
                        continue
                    data[j, i] = y_norm * self.norm_scale_map[j, i]

        label = (f'Normalised intensity at {specific_xdata} eV (a.u.)'
                if self.normalize else
                f'Intensity at {specific_xdata} eV (a.u.)')

    elif data_type == 'exciton_position':
        data = self.peak_positions[:, :, 0]
        label = 'Exciton Position (eV)'

    elif data_type == 'trion_position':
        if self.peak_positions.shape[2] > 1:
            data = self.peak_positions[:, :, 1]
            label = 'Trion Position (eV)'
        else:
            raise ValueError("Trion data not available.")

    elif data_type == 'exciton_intensity':
        data = self.peak_intensities[:, :, 0]
        label = 'Exciton Intensity (a.u.)'  # already scaled in fit_spectra when normalize=False

    elif data_type == 'trion_intensity':
        if self.peak_intensities.shape[2] > 1:
            data = self.peak_intensities[:, :, 1]
            label = 'Trion Intensity (a.u.)'
        else:
            raise ValueError("Trion data not available.")

    else:
        raise ValueError("Invalid data_type. Choose from "
                        "'exciton_position', 'trion_position', "
                        "'exciton_intensity', 'trion_intensity', 'specific_intensity'.")

    # Apply optional range filter (your current behaviour: clip outliers to lower bound)
    if filter_range is not None:
        data = np.where((data >= filter_range[0]) & (data <= filter_range[1]), data, filter_range[0])

    # Apply optional cropping
    if x_range is not None and y_range is not None:
        x_start, x_end = x_range
        y_start, y_end = y_range
        data = data[y_start:y_end + 1, x_start:x_end + 1]
        Xp = (x_end - x_start + 1)
        Yp = (y_end - y_start + 1)
    else:
        Xp, Yp = self.X, self.Y

    x_length = Xp * self.step_size
    y_length = Yp * self.step_size

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('gray')

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        data,
        cmap=cm,
        vmin=filter_range[0] if filter_range else None,
        vmax=filter_range[1] if filter_range else None,
        extent=[0, x_length, y_length, 0]
    )
    plt.colorbar(im, label=label)
    plt.xlabel("X Position (μm)")
    plt.ylabel("Y Position (μm)")
    plt.title(f"Heatmap of {label}")
    plt.tight_layout()
    plt.show()

