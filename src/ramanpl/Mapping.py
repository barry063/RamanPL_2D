"""Module for spectroscopic mapping data analysis and visualization.

Provides classes for loading, processing, and visualizing photoluminescence (PL) 
and Raman mapping data from .wdf and .txt files. Includes peak fitting, spectral 
integration, and 2D heatmap visualization capabilities.

Classes:
    MappingFileLoader: Loads spectroscopic mapping data from files
    MappingImage: Displays optical images from .wdf files
    PLMapping: Analyzes PL data through Lorentzian peak fitting
    PL_Integration: Analyzes PL data through spectral integration
    RamanMapping: Analyzes Raman data through Lorentzian peak fitting
    Raman_Integration: Analyzes Raman data through spectral integration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from renishawWiRE import WDFReader
from ramanpl import BaselineAPI
from ramanpl import DataImporter
from typing import Optional, Tuple
from ramanpl.exporter import params_to_rows, write_table

#########################################################################################################################


class MappingFileLoader:
    """Loader for spectroscopic mapping data from .wdf and .txt files.

    After refactor:
    - delegates file IO to ramanpl.dataImporter.DataImporter.map_import
    - optionally trims the spectral axis using x_range to reduce memory usage

    Attributes:
        filename (str): Path to input file
        data_format (str): File format ('txt' or 'wdf')
        X (int): Number of points in X-direction
        Y (int): Number of points in Y-direction
        xdata (ndarray): Spectral axis values (eV for PL, cm^-1 for Raman)
        spectra (ndarray): 3D array of spectra [Y, X, spectral_points]
    """

    def __init__(self, filename, x_range=None, axis="auto", txt_skiprows=1):
        self.filename = filename
        self.axis = axis
        self.data_format = "wdf" if filename.lower().endswith(".wdf") else "txt" if filename.lower().endswith(".txt") else "unknown"

        if self.data_format == "unknown":
            raise RuntimeError("Unsupported mapping file format. Supported formats: .wdf, .txt")

        spectra_cube, xdata, X, Y = DataImporter.map_import(
            filename=filename,
            x_range=x_range,
            axis=axis,
            txt_skiprows=txt_skiprows,
        )

        self.X = X
        self.Y = Y
        self.xdata = xdata
        self.spectra = spectra_cube


class MappingImage:
    """Displays optical images from .wdf files with mapping region overlay.
    
    Attributes:
        reader (WDFReader): Renishaw file reader object
    """
    def __init__(self, filename):
        """Initialize image viewer for .wdf files.
        
        Args:
            filename (str): Path to .wdf file
            
        Raises:
            ValueError: If non-.wdf file is provided
        """
        if not filename.endswith(".wdf"):
            raise ValueError("MappingImage can only be used with .wdf files.")
        self.reader = WDFReader(filename)

    def show_optical_image(self):
        """Display optical image with mapping area rectangle overlay.
        
        Uses PIL for image handling and matplotlib for visualization.
        """
        from PIL import Image
        import matplotlib.patches as patches

        image = Image.open(self.reader.img)
        cb = self.reader.img_cropbox
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        rect = patches.Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.title("Optical Image with Mapping Area")
        plt.show()


#########################################################################################################################
#################################################### PL Mapping #########################################################
#########################################################################################################################

class PLMapping:
    """Photoluminescence mapping analysis through Lorentzian peak fitting.
    
    Attributes:
        filename (str): Path to .wdf file
        custom_peaks (dict): Peak parameters for fitting
        data_range (tuple): Spectral analysis range (min, max) in eV
        step_size (float): Physical step size in micrometers
        poly_degree (int): Polynomial degree for background removal
        normalize (bool): Enable spectrum normalization
        background_remove (bool): Enable background subtraction
        baseline_method (str): Background method ('poly' or 'gaussian')
        smoothing (bool): Enable spectral smoothing
        smooth_window (int): Savitzky-Golay window size
        smooth_poly (int): Savitzky-Golay polynomial order
        gaussian_sigma (int): Gaussian filter width
        peak_params (list): Peak names from custom_peaks
        X (int): Map width in pixels
        Y (int): Map height in pixels
        xdata (ndarray): Spectral axis in eV
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        peak_positions (ndarray): Fitted peak centers [Y, X, peaks]
        peak_intensities (ndarray): Fitted peak amplitudes [Y, X, peaks]
        fitted_params (ndarray): Full fitting parameters [Y, X, 3*peaks]
        residual_map (ndarray): Fitting residuals [Y, X]
    """

    def __init__(self, filename, custom_peaks, data_range=None, step_size=0.3,
                 poly_degree=3, normalize=True, background_remove=True,
                 baseline_method='poly', smoothing=True, smooth_window=11,
                 smooth_poly=3, gaussian_sigma=10):
        """Initialize PL mapping analyzer.
        
        Args:
            filename: Path to .wdf PL mapping file
            custom_peaks: Peak definitions with bounds {name: (min_params, max_params)}
            data_range: Spectral range (min, max) in eV (default: full spectrum)
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            normalize (bool): Controls display/output scaling only. Fitting is always performed on peak-normalised spectra.
            background_remove: Enable background subtraction
            baseline_method: 'poly' or 'gaussian' background
            smoothing: Enable spectral smoothing
            smooth_window: Savitzky-Golay window size
            smooth_poly: Savitzky-Golay polynomial order
            gaussian_sigma: Gaussian filter width
        """

        self.filename = filename
        self.custom_peaks = custom_peaks
        self.data_range = data_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.normalize = normalize
        self.background_remove = background_remove
        self.baseline_method = baseline_method
        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma
        self.peak_params = list(custom_peaks.keys())

        # --- identity metadata for exports (added in v0.2.8) ---
        self.spectrum_type = "Photoluminescence"
        self.x_quantity = "Photon energy"
        self.x_unit = "eV"
        self.step_unit = "um"  # keep consistent with your plotting labels "μm"


        # New in v0.2.5: Baseline configuration (single source of truth)
        self._baseline_method, self._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma
        )

        # If user supplied a data_range, trim at load-time to reduce memory and speed downstream work.
        if self.data_range is not None:
            loader = MappingFileLoader(filename, x_range=self.data_range, axis="energy")
            self._x_trimmed_on_load = True
        else:
            loader = MappingFileLoader(filename, axis="energy")
            self._x_trimmed_on_load = False

        self.X = loader.X
        self.Y = loader.Y
        self.xdata = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None

        # If not provided, default to the full (possibly already trimmed) x-range
        if self.data_range is None:
            self.data_range = (float(np.min(self.xdata)), float(np.max(self.xdata)))

        num_peaks = len(self.custom_peaks)
        self.peak_positions = np.zeros((self.Y, self.X, num_peaks))
        self.peak_intensities = np.zeros((self.Y, self.X, num_peaks))
        self.fitted_params = np.zeros((self.Y, self.X, num_peaks * 3))
        
        ### UPDATED INITIALIZATION in v0.2.2 ###
        self.residual_map = np.full((self.Y, self.X), np.nan)
        self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        ### END UPDATED INITIALIZATION ###

    ### NEW METHOD IN v0.2.7 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        custom_peaks,
        data_range=None,
        step_size=0.3,
        poly_degree=3,
        normalize=True,
        background_remove=True,
        baseline_method='poly',
        smoothing=True,
        smooth_window=11,
        smooth_poly=3,
        gaussian_sigma=10,
    ):
        """
        Create a PLMapping instance from in-memory mapping arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Spectral axis with shape [N] (energy in eV)
        X, Y : int
            Map dimensions
        """

        obj = cls.__new__(cls)

        # ---- mirror __init__ fields ----
        obj.filename = None
        obj.custom_peaks = custom_peaks
        obj.data_range = data_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.normalize = normalize
        obj.background_remove = background_remove
        obj.baseline_method = baseline_method
        obj.smoothing = smoothing
        obj.smooth_window = smooth_window
        obj.smooth_poly = smooth_poly
        obj.gaussian_sigma = gaussian_sigma
        obj.peak_params = list(custom_peaks.keys())

        # Baseline config (same as __init__)
        obj._baseline_method, obj._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma
        )

        # ---- assign data ----
        obj.X = int(X)
        obj.Y = int(Y)
        obj.xdata = np.asarray(xdata, dtype=float).ravel()
        obj.spectra = np.asarray(spectra, dtype=float)

        # Validate shapes
        if obj.spectra.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if obj.spectra.shape[0] != obj.Y or obj.spectra.shape[1] != obj.X:
            raise ValueError(f"spectra shape {obj.spectra.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if obj.spectra.shape[2] != obj.xdata.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        # No optical image when constructed from arrays
        obj.image_viewer = None

        # Load-time trimming flag: False, because we did not trim during import here
        obj._x_trimmed_on_load = False

        # Default range: full axis if not supplied
        if obj.data_range is None:
            obj.data_range = (float(np.min(obj.xdata)), float(np.max(obj.xdata)))

        # Allocate output arrays (same as __init__)
        num_peaks = len(obj.custom_peaks)
        obj.peak_positions = np.zeros((obj.Y, obj.X, num_peaks))
        obj.peak_intensities = np.zeros((obj.Y, obj.X, num_peaks))
        obj.fitted_params = np.zeros((obj.Y, obj.X, num_peaks * 3))
        obj.residual_map = np.full((obj.Y, obj.X), np.nan)
        obj.norm_scale_map = np.full((obj.Y, obj.X), np.nan)

        return obj

    ### New in v0.2.7 ###
    def get_reference_spectrum(self, *, x: int, y: int, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Return a reference spectrum from this already-loaded mapping object.

        Parameters
        ----------
        x, y : int
            Pixel coordinate (0-indexed)
        roi : (x0, x1, y0, y1) inclusive, optional
            If provided, returns the mean spectrum over the ROI.

        Returns
        -------
        (y_ref, xdata)
        """
        if roi is not None:
            x0, x1, y0, y1 = roi
            if not (0 <= x0 <= x1 < self.X and 0 <= y0 <= y1 < self.Y):
                raise ValueError("ROI out of bounds.")
            y_ref = np.nanmean(self.spectra[y0:y1+1, x0:x1+1, :], axis=(0, 1))
        else:
            if not (0 <= x < self.X and 0 <= y < self.Y):
                raise ValueError("Pixel out of bounds.")
            y_ref = self.spectra[y, x, :]

        return np.asarray(y_ref, dtype=float).ravel(), np.asarray(self.xdata, dtype=float).ravel()

    ## NEW METHOD ADDED in v0.2.3 ##
    def _preprocess_single_spectrum(self, xdata, spec):
        """
        Preprocessing for fitting (always normalised):
        1) optional smoothing
        2) optional background removal
        3) normalisation by peak intensity

        Returns
        -------
        y_norm : ndarray
            Peak-normalised spectrum for fitting
        scale : float
            Peak intensity used for scaling back (raw units)
        """
        y = np.asarray(spec, dtype=float)

        # 1) smoothing
        if self.smoothing:
            y = savgol_filter(y, self.smooth_window, self.smooth_poly)

        # 2) background removal
        if self.background_remove:
            y = self.remove_background(xdata, y)

        # 3) peak normalisation (always)
        scale = np.max(y)
        if scale <= 0:
            return None, None

        y_norm = y / scale
        return y_norm, scale
    ## END NEW METHOD ###

    def show_optical_image(self):
        """Display the optical image."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def lorentzian(self, x, *params):
        """Multi-Lorentzian function for curve fitting.
        
        Args:
            x: Spectral axis values
            *params: Fitting parameters (loc, scale, amp) for each peak
            
        Returns:
            Sum of Lorentzian components
        """
        result = np.zeros_like(x)
        for i in range(0, len(params), 3):
            loc = params[i]
            scale = params[i+1]
            amp = params[i+2]
            result += (scale / ((x - loc)**2 + scale**2)) * amp / np.pi
        return result

    ### UPDATED METHOD in v0.2.5 ##
    def remove_background(self, xdata, intensity):
        """Remove spectral background via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=xdata,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected



    ## UPDATED METHOD in v0.2.3 ##
    def fit_spectra(self, initial_p0=None, warm_start=False, reset_on_fail=True,
                    maxfev=6400, warm_start_rmse_gate=0.06):
        """
        Fit all map spectra using self.custom_peaks as bounds.

        Parameters
        ----------
        initial_p0 : array-like or dict or None
            Optional initial guess vector (e.g., from a single-point PLfit result),
            or dict package {"p0": <vector>, "peak_order": <list>}.
            Must match parameter ordering implied by self.custom_peaks.
        warm_start : bool
            If True, use previous successful fit parameters as p0 for next pixel.
        reset_on_fail : bool
            If True, on fit failure reset p0 to baseline (midpoint/initial_p0).
        maxfev : int
            curve_fit maximum function evaluations.
        warm_start_rmse_gate : float
            RMSE threshold (normalised space) for accepting warm-start propagation.

        Returns
        -------
        params_map : ndarray
            Fitted parameter cube with shape [Y, X, n_params].
            In notebooks, assign the return value or use `_ = fit_spectra(...)`
            to avoid auto-display.

        """
        import numpy as np
        from scipy import optimize

        if not hasattr(self, "custom_peaks") or not isinstance(self.custom_peaks, dict) or len(self.custom_peaks) == 0:
            raise ValueError("custom_peaks is not set or empty. Provide custom_peaks when initialising PLMapping.")

        # --- Build spectral mask from energy range (eV)
        # If trimmed at load, no need to remask.
        if getattr(self, "_x_trimmed_on_load", False):
            xdata = self.xdata
            mask = None
        else:
            mask = DataImporter.mask_by_xrange(self.xdata, self.data_range)
            xdata = self.xdata[mask]

        # --- Build bounds from custom_peaks (in insertion order)
        lower_bound, upper_bound = [], []
        for params_range in self.custom_peaks.values():
            lower_bound.extend(params_range[0])
            upper_bound.extend(params_range[1])

        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
        n_params = lower_bound.size

        # Default p0: midpoint of bounds
        p0_base = (lower_bound + upper_bound) / 2.0

        # Optional: seed from single-point PLfit
        if initial_p0 is not None:

            if isinstance(initial_p0, dict):
                peak_order_pkg = initial_p0.get("peak_order", None)
                p0_vec = initial_p0.get("p0", None)

                if p0_vec is None:
                    raise ValueError("initial_p0 dict must contain key 'p0' with a numeric vector.")

                if peak_order_pkg is not None:
                    if [p.lower() for p in peak_order_pkg] != [p.lower() for p in self.peak_params]:
                        raise ValueError(
                            "peak_order mismatch between PLfit and PLMapping.\n"
                            f"PLfit: {list(peak_order_pkg)}\n"
                            f"PLMapping: {list(self.peak_params)}\n"
                            "Ensure both use the same custom_peaks ordering (or pass peak_order explicitly)."
                        )

                initial_p0 = p0_vec

            initial_p0 = np.asarray(initial_p0, dtype=float)
            if initial_p0.shape != p0_base.shape:
                raise ValueError(f"initial_p0 shape {initial_p0.shape} does not match expected {p0_base.shape}")

            p0_base = np.clip(initial_p0, lower_bound, upper_bound)

        p0_current = p0_base.copy()

        # Output arrays
        fitted_params = np.full((self.Y, self.X, n_params), np.nan)

        # Ensure these exist and are float arrays
        if not hasattr(self, "norm_scale_map"):
            self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        if not hasattr(self, "residual_map"):
            self.residual_map = np.full((self.Y, self.X), np.nan)

        for j in range(self.Y):
            for i in range(self.X):
                raw_spec = self.spectra[j, i, :] if mask is None else self.spectra[j, i, :][mask]

                spec_norm, scale = self._preprocess_single_spectrum(xdata, raw_spec)

                if spec_norm is None:
                    self.norm_scale_map[j, i] = np.nan
                    self.residual_map[j, i] = np.nan
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                # valid spectrum
                self.norm_scale_map[j, i] = scale

                try:
                    params, _ = optimize.curve_fit(
                        self.lorentzian,
                        xdata,
                        spec_norm,
                        p0=p0_current,
                        bounds=(lower_bound, upper_bound),
                        maxfev=maxfev
                    )

                    # Residual (fit space)
                    model_norm = self.lorentzian(xdata, *params)
                    residual_norm = spec_norm - model_norm
                    rmse_norm = np.sqrt(np.mean(residual_norm ** 2))
                    self.residual_map[j, i] = rmse_norm

                    fitted_params[j, i, :] = params

                    # Store peak positions and intensities
                    for k, _ in enumerate(self.peak_params):
                        idx = 3 * k
                        center, width, amp = params[idx:idx + 3]
                        self.peak_positions[j, i, k] = center

                        peak_height_norm = amp / (np.pi * width)
                        if self.normalize:
                            self.peak_intensities[j, i, k] = peak_height_norm
                        else:
                            self.peak_intensities[j, i, k] = peak_height_norm * scale

                    # Gated warm-start
                    if warm_start:
                        if rmse_norm <= warm_start_rmse_gate:
                            p0_current = params
                        else:
                            if reset_on_fail:
                                p0_current = p0_base.copy()

                except RuntimeError:
                    self.residual_map[j, i] = np.nan
                    # keep the scale (it was valid), but reset p0 if requested
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

        n_fit = np.sum(~np.isnan(self.residual_map))
        print(f"Successful fits: {n_fit} / {self.X * self.Y}")

        self.fitted_params = fitted_params
        return fitted_params
               
    def plot_spectrum_fit(self, x, y):
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
        scale = np.max(bg_removed)
        if scale <= 0:
            raise ValueError(f"No positive signal at (X={x}, Y={y}); cannot scale fitted curve.")

        # --- Load fitted parameters (normalised space)
        params = np.asarray(self.fitted_params[y, x, :], dtype=float)
        if np.any(np.isnan(params)):
            raise ValueError(f"Fit parameters are NaN at (X={x}, Y={y}). Fit may have failed.")

        fitted_norm = self.lorentzian(xdata, *params)
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

    def plot_residual_distribution(
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

    def plot_heatmap(self, data_type='exciton_position', cmap='viridis',
                    filter_range=None, specific_xdata=None,
                    x_range=None, y_range=None):
        """Visualize 2D map of spectral features."""
        import numpy as np
        import matplotlib.pyplot as plt

        if data_type == 'specific_intensity':
            if specific_xdata is None:
                raise ValueError("For 'specific_intensity' data type, 'specific_xdata' must be provided (in eV).")

            data = np.full((self.Y, self.X), np.nan, dtype=float)

            for j in range(self.Y):
                for i in range(self.X):
                    params = self.fitted_params[j, i, :]
                    if np.any(np.isnan(params)):
                        continue  # fit failed

                    y_norm = self.lorentzian(specific_xdata, *params)

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

    ### Added in v0.2.8
    def _iter_coords(self, coord_mode: str = "pixel"):
        """
        Yield (x, y, j, i) for every pixel.

        coord_mode:
        - "pixel": x,y are integer pixel indices
        - "real":  x,y are physical coordinates using step_size
        """
        step = float(self.step_size)
        for j in range(self.Y):
            for i in range(self.X):
                if coord_mode == "real":
                    yield (i * step, j * step, j, i)
                else:
                    yield (i, j, j, i)

    ### Added in v0.2.8
    def export_fit_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        scaled: bool = True,
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
    ) -> str:
        """
        Export fit results for every pixel in wide format:
        x, y, then per-peak parameters on the same row.

        Per-peak columns:
        <peak>_centre, <peak>_fwhm, <peak>_height_scaled, <peak>_height_norm, <peak>_amp, <peak>_scale
        """
        if not hasattr(self, "fitted_params") or self.fitted_params is None:
            raise ValueError("No fitted_params found. Run fit_spectra() first.")

        peak_labels = list(self.peak_params)  # authoritative ordering in your mapping class :contentReference[oaicite:14]{index=14}
        fields = ["x", "y"]

        per_peak_fields = ["centre", "fwhm", "height_scaled", "height_norm", "amp", "scale"]
        for p in peak_labels:
            for f in per_peak_fields:
                fields.append(f"{p}_{f}")

        rows = []
        for x, y, j, i in self._iter_coords(coord_mode=coord_mode):
            params = np.asarray(self.fitted_params[j, i, :], dtype=float)
            if np.any(np.isnan(params)):
                # keep row but leave values empty to preserve grid
                rows.append({"x": x, "y": y})
                continue

            intensity_scale = 1.0
            if scaled and hasattr(self, "norm_scale_map") and np.isfinite(self.norm_scale_map[j, i]):
                intensity_scale = float(self.norm_scale_map[j, i])

            peak_rows = params_to_rows(
                peak_labels=peak_labels,
                params=params,
                intensity_scale=intensity_scale,
            )

            r = {"x": x, "y": y}
            for pr in peak_rows:
                name = pr.peak
                r[f"{name}_centre"] = pr.centre
                r[f"{name}_fwhm"] = pr.fwhm
                r[f"{name}_height_scaled"] = pr.height_scaled
                r[f"{name}_height_norm"] = pr.height_norm
                r[f"{name}_amp"] = pr.amp
                r[f"{name}_scale"] = pr.scale

            rows.append(r)

        meta = {
            "map_kind": "fit_params",
            "spectrum_type": getattr(self, "spectrum_type", None),
            "x_quantity": getattr(self, "x_quantity", None),
            "x_unit": getattr(self, "x_unit", None),
            "coord_mode": coord_mode,
            "step_size": getattr(self, "step_size", None),
            "step_unit": getattr(self, "step_unit", "um"),
            "scaled": scaled,
            "peak_labels": peak_labels,
            "background_remove": getattr(self, "background_remove", None),
            "baseline_method": getattr(self, "baseline_method", None),
            "smoothing": getattr(self, "smoothing", None),
            "smooth_window": getattr(self, "smooth_window", None),
            "smooth_poly": getattr(self, "smooth_poly", None),
        }
        meta = {k: v for k, v in meta.items() if v is not None}

        return write_table(
            rows,
            out_path,
            fieldnames=fields,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )

    
#########################################################################################################################
############################################## PL Integration Mapping ###################################################
#########################################################################################################################

class PL_Integration:
    """Photoluminescence mapping analysis through spectral integration.
    
    Attributes:
        filename (str): Path to input file
        integration_range (tuple): Spectral integration range (min, max) in eV
        step_size (float): Physical step size in micrometers
        poly_degree (int): Background polynomial degree
        background_remove (bool): Enable background subtraction
        X (int): Map width in pixels
        Y (int): Map height in pixels
        energy (ndarray): Spectral axis in eV
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        integration_area (ndarray): Integrated intensities [Y, X]
    """
    def __init__(self, filename, integration_range, step_size=0.3, poly_degree=3,
             background_remove=True, baseline_method="poly"):
        """Initialize PL integration analyzer.
        
        Args:
            filename: Path to .wdf file
            integration_range: Spectral range (min, max) in eV
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            background_remove: Enable background subtraction
            baseline_method: 'poly' or 'gaussian' background
        """
        self.filename = filename
        self.integration_range = integration_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.background_remove = background_remove

        # --- identity metadata for exports (added in v0.2.8) ---
        self.spectrum_type = "Photoluminescence"
        self.x_quantity = "Photon energy"
        self.x_unit = "eV"
        self.step_unit = "um"  # keep consistent with your plotting labels "μm"


        # New in v0.2.5 Baseline configuration (single source of truth; backward compatible)
        self.baseline_method = baseline_method
        self._baseline_method, self._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree
        )

        # integration_range is known here; trim at load-time.
        loader = MappingFileLoader(filename, x_range=self.integration_range, axis="energy")
        self._x_trimmed_on_load = True

        self.X = loader.X
        self.Y = loader.Y
        self.energy = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        self.integration_area = np.zeros((self.Y, self.X))

    ### NEW METHOD in v0.2.7 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        integration_range,
        step_size=0.3,
        poly_degree=3,
        background_remove=True,
        baseline_method="poly",
        clip_nonnegative=False,
    ):
        """
        Construct PL_Integration from in-memory arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Energy axis (eV) with shape [N]
        X, Y : int
            Map dimensions
        integration_range : tuple(float, float)
            (min, max) eV; applied immediately (trim at construction, consistent with __init__)
        """
        obj = cls.__new__(cls)

        # Mirror __init__ fields
        obj.filename = None
        obj.integration_range = integration_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.background_remove = background_remove

        # Baseline configuration (same pattern as __init__)
        obj.baseline_method = baseline_method
        obj._baseline_method, obj._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree
        )

        obj.X = int(X)
        obj.Y = int(Y)

        energy = np.asarray(xdata, dtype=float).ravel()
        cube = np.asarray(spectra, dtype=float)

        # Validate shapes
        if cube.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if cube.shape[0] != obj.Y or cube.shape[1] != obj.X:
            raise ValueError(f"spectra shape {cube.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if cube.shape[2] != energy.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        # Trim at load-time (same semantics as filename-based __init__)
        emin, emax = integration_range
        mask = (energy >= emin) & (energy <= emax)
        if not np.any(mask):
            raise ValueError(
                f"integration_range {integration_range} does not overlap provided energy axis "
                f"[{float(np.min(energy)):.3g}, {float(np.max(energy)):.3g}]."
            )

        obj.energy = energy[mask]
        obj.spectra = cube[:, :, mask]
        obj._x_trimmed_on_load = True

        if clip_nonnegative:
            obj.spectra = np.clip(obj.spectra, a_min=0.0, a_max=None)

        # No optical image when constructed from arrays
        obj.image_viewer = None

        # Output
        obj.integration_area = np.zeros((obj.Y, obj.X), dtype=float)

        return obj

    def show_optical_image(self):
        """Display the optical image."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    ### Updated in v0.2.5 ##
    def remove_background(self, energy, intensity):
        """Background removal via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=energy,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected


    def calculate_integration(self):
        """Calculate integrated area under spectra across all map points.

        Uses Simpson's rule for integration.
        Stores results in integration_area array.
        """
        energy = np.asarray(self.energy, dtype=float).ravel()

        mask = DataImporter.mask_by_xrange(energy, self.integration_range)
        energy_subset = energy[mask]

        for j in range(self.Y):
            for i in range(self.X):
                spectra = np.asarray(self.spectra[j, i, :], dtype=float).ravel()
                spectra_subset = spectra[mask]

                if self.background_remove:
                    spectra_subset = self.remove_background(energy_subset, spectra_subset)

                self.integration_area[j, i] = np.abs(simpson(spectra_subset, energy_subset))


    def plot_integration_heatmap(self, cmap='viridis', filter_range=None, x_range=None, y_range=None):
        """Visualize 2D map of integrated intensities.
        
        Args:
            cmap: Matplotlib colormap name
            filter_range: Data display range [min, max]
            x_range: X display range [start, end]
            y_range: Y display range [start, end]
        """
        # Filter data range
        data = self.integration_area
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
        cbar = plt.colorbar(im, label='Integration Area (a.u.)')
        plt.xlabel("X Position (μm)")
        plt.ylabel("Y Position (μm)")
        plt.title(f"Integration Area Heatmap ({self.integration_range[0]} - {self.integration_range[1]} eV)")
        plt.show()

    def plot_spectrum(self, x, y):
        """Plot raw and processed spectra for single map point.
        
        Args:
            x (int): X coordinate (0-indexed)
            y (int): Y coordinate (0-indexed)
            
        Shows:
            - Raw spectrum (blue)
            - Background-removed spectrum (red, if enabled)
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # Get the original spectrum data
        energy = self.energy[:]
        spectra = self.spectra[y][x][:]

        # Get data within the integration range
        mask = DataImporter.mask_by_xrange(energy, self.integration_range)
        energy_subset = energy[mask]
        spectra_subset = spectra[mask]


        # If background removal is enabled, remove the background signal
        spectra_raw = spectra_subset.copy()

        if self.background_remove:
            spectra_bg_removed = self.remove_background(energy_subset, spectra_subset)
        else:
            spectra_bg_removed = spectra_subset

        plt.figure(figsize=(10, 6))
        plt.plot(energy_subset, spectra_raw, 'b-', label='Original Spectrum')
        if self.background_remove:
            plt.plot(energy_subset, spectra_bg_removed, 'r--', label='Background Removed')
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum at (X={x}, Y={y})")
        plt.legend()
        plt.show()

    ### Added in v0.2.8
    def export_integration_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
        column_name: str = "integration_area",
    ) -> str:
        """
        Export integration_area in wide format:
        x, y, integration_area
        """
        if not hasattr(self, "integration_area") or self.integration_area is None:
            raise ValueError("No integration_area found. Run calculate_integration() first.")

        # coordinate iterator local to this class
        step = float(self.step_size)
        rows = []
        for j in range(self.Y):
            for i in range(self.X):
                x, y = (i * step, j * step) if coord_mode == "real" else (i, j)
                rows.append({"x": x, "y": y, column_name: float(self.integration_area[j, i])})

        fields = ["x", "y", column_name]

        meta = {
            "map_kind": "integration",
            "spectrum_type": getattr(self, "spectrum_type", None),
            "x_unit": getattr(self, "x_unit", None),
            "coord_mode": coord_mode,
            "step_size": getattr(self, "step_size", None),
            "step_unit": getattr(self, "step_unit", "um"),
            "integration_range": getattr(self, "integration_range", None),
            "background_remove": getattr(self, "background_remove", None),
            "baseline_method": getattr(self, "baseline_method", None),
        }
        meta = {k: v for k, v in meta.items() if v is not None}

        return write_table(
            rows,
            out_path,
            fieldnames=fields,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )
    

########################################################################################################################
#################################################### Raman Mapping #####################################################
########################################################################################################################

class RamanMapping:
    """Raman mapping analysis through Lorentzian peak fitting.
    
    Attributes:
        filename (str): Path to .wdf file
        custom_peaks (dict): Peak parameters for fitting
        data_range (tuple): Spectral analysis range (min, max) in cm⁻¹
        step_size (float): Physical step size in micrometers
        poly_degree (int): Background polynomial degree
        normalize (bool): Enable spectrum normalization
        background_remove (bool): Enable background subtraction
        smoothing (bool): Enable spectral smoothing
        baseline_method (str): Background method ('poly' or 'gaussian')
        smooth_window (int): Savitzky-Golay window size
        smooth_poly (int): Savitzky-Golay polynomial order
        gaussian_sigma (int): Gaussian filter width
        peak_params (list): Peak names from custom_peaks
        X (int): Map width in pixels
        Y (int): Map height in pixels
        wavenumber (ndarray): Spectral axis in cm⁻¹
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        peak_positions (ndarray): Fitted peak centers [Y, X, peaks]
        peak_intensities (ndarray): Fitted peak amplitudes [Y, X, peaks]
        fitted_params (ndarray): Full fitting parameters [Y, X, 3*peaks]
        residual_map (ndarray): Fitting residuals [Y, X]
        Peaks_distance (ndarray): A1g-E2g peak distances [Y, X]
        ratio_A1g_E2g (ndarray): A1g/E2g intensity ratios [Y, X]
        ratio_E2g_A1g (ndarray): E2g/A1g intensity ratios [Y, X]
    """
    def __init__(self, filename, custom_peaks, data_range, step_size=0.3, poly_degree=3,
                 normalize=False, background_remove=True, smoothing=True, baseline_method='poly', smooth_window=11,
                 smooth_poly=3, gaussian_sigma=10):
        """Initialize Raman mapping analyzer.
        
        Args:
            filename: Path to .wdf file
            custom_peaks: Peak definitions with bounds {name: (min_params, max_params)}
            data_range: Spectral range (min, max) in cm⁻¹
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            normalize: Normalize spectra to [0,1] range
            background_remove: Enable background subtraction
            smoothing: Enable spectral smoothing
            baseline_method: 'poly' or 'gaussian' background
            smooth_window: Savitzky-Golay window size
            smooth_poly: Savitzky-Golay polynomial order
            gaussian_sigma: Gaussian filter width
        """
        self.filename = filename
        self.custom_peaks = custom_peaks
        self.data_range = data_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.normalize = normalize
        self.background_remove = background_remove
        self.smoothing = smoothing
        self.baseline_method = baseline_method
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.gaussian_sigma = gaussian_sigma
        self.peak_params = list(custom_peaks.keys())

        # --- identity metadata for exports (added in v0.2.8) ---
        self.spectrum_type = "Raman"
        self.x_quantity = "Raman shift"
        self.x_unit = "cm^-1"
        self.step_unit = "um"


        # New in v0.2.5: Baseline configuration (single source of truth)
        self._baseline_method, self._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma
        )

        # data_range is mandatory for RamanMapping in your current signature; trim at load-time.
        loader = MappingFileLoader(filename, x_range=self.data_range, axis="wavenumber")
        self._x_trimmed_on_load = True

        self.X = loader.X
        self.Y = loader.Y
        self.wavenumber = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        
        # Initialize arrays with dynamic dimensions based on number of peaks
        num_peaks = len(self.custom_peaks)

        self.peak_positions = np.full((self.Y, self.X, num_peaks), np.nan)
        self.peak_intensities = np.full((self.Y, self.X, num_peaks), np.nan)
        self.fitted_params = np.full((self.Y, self.X, num_peaks * 3), np.nan)

        ## Updated in v0.2.4 ##
        self.residual_map = np.full((self.Y, self.X), np.nan)
        self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        # Initialise derived maps as NaN (so “not computed / invalid” is visible)
        self.Peaks_distance = np.full((self.Y, self.X), np.nan, dtype=float)
        self.ratio_A1g_E2g = np.full((self.Y, self.X), np.nan, dtype=float)
        self.ratio_E2g_A1g = np.full((self.Y, self.X), np.nan, dtype=float)
        #####################

    ### New in v0.2.7 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        custom_peaks,
        data_range,
        step_size=0.3,
        poly_degree=3,
        normalize=False,
        background_remove=True,
        smoothing=True,
        baseline_method='poly',
        smooth_window=11,
        smooth_poly=3,
        gaussian_sigma=10,
    ):
        """
        Create a RamanMapping instance from in-memory mapping arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Spectral axis with shape [N] (wavenumber in cm^-1)
        X, Y : int
            Map dimensions
        data_range : tuple
            (min, max) in cm^-1 (kept as in the filename-based API)
        """
        obj = cls.__new__(cls)

        obj.filename = None
        obj.custom_peaks = custom_peaks
        obj.data_range = data_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.normalize = normalize
        obj.background_remove = background_remove
        obj.smoothing = smoothing
        obj.baseline_method = baseline_method
        obj.smooth_window = smooth_window
        obj.smooth_poly = smooth_poly
        obj.gaussian_sigma = gaussian_sigma
        obj.peak_params = list(custom_peaks.keys())

        # Baseline config (same as __init__)
        obj._baseline_method, obj._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree,
            gaussian_sigma=gaussian_sigma
        )

        obj.X = int(X)
        obj.Y = int(Y)
        obj.wavenumber = np.asarray(xdata, dtype=float).ravel()
        obj.spectra = np.asarray(spectra, dtype=float)

        # Validate shapes
        if obj.spectra.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if obj.spectra.shape[0] != obj.Y or obj.spectra.shape[1] != obj.X:
            raise ValueError(f"spectra shape {obj.spectra.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if obj.spectra.shape[2] != obj.wavenumber.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        obj.image_viewer = None

        # Not trimmed at load time when built from arrays
        obj._x_trimmed_on_load = False

        # Allocate arrays (same as __init__)
        num_peaks = len(obj.custom_peaks)
        obj.peak_positions = np.full((obj.Y, obj.X, num_peaks), np.nan)
        obj.peak_intensities = np.full((obj.Y, obj.X, num_peaks), np.nan)
        obj.fitted_params = np.full((obj.Y, obj.X, num_peaks * 3), np.nan)

        obj.residual_map = np.full((obj.Y, obj.X), np.nan)
        obj.norm_scale_map = np.full((obj.Y, obj.X), np.nan)

        obj.Peaks_distance = np.full((obj.Y, obj.X), np.nan, dtype=float)
        obj.ratio_A1g_E2g = np.full((obj.Y, obj.X), np.nan, dtype=float)
        obj.ratio_E2g_A1g = np.full((obj.Y, obj.X), np.nan, dtype=float)

        return obj

    ### New in v0.2.7 ###
    def get_reference_spectrum(self, *, x: int, y: int, roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Return a reference spectrum from this already-loaded mapping object.

        Returns
        -------
        (y_ref, wavenumber)
        """
        if roi is not None:
            x0, x1, y0, y1 = roi
            if not (0 <= x0 <= x1 < self.X and 0 <= y0 <= y1 < self.Y):
                raise ValueError("ROI out of bounds.")
            y_ref = np.nanmean(self.spectra[y0:y1+1, x0:x1+1, :], axis=(0, 1))
        else:
            if not (0 <= x < self.X and 0 <= y < self.Y):
                raise ValueError("Pixel out of bounds.")
            y_ref = self.spectra[y, x, :]

        return np.asarray(y_ref, dtype=float).ravel(), np.asarray(self.wavenumber, dtype=float).ravel()

    ### NEW METHOD IN v0.2.4 ###
    def _preprocess_single_spectrum(self, xdata, spec):
        """
        Preprocessing for fitting (always normalised):
        1) optional smoothing
        2) optional background removal
        3) normalisation by peak intensity

        Returns
        -------
        y_norm : ndarray or None
            Peak-normalised spectrum for fitting
        scale : float or None
            Peak intensity used for scaling back (raw units)
        """
        y = np.asarray(spec, dtype=float)

        if self.smoothing:
            y = savgol_filter(y, self.smooth_window, self.smooth_poly)

        if self.background_remove:
            y = self.remove_background(xdata, y)

        scale = np.max(y)
        if scale <= 0:
            return None, None

        return y / scale, scale
    
    @staticmethod
    def custom_peaks_from_ramanfit(raman_fit):
        """
        Build Mapping-compatible custom_peaks dict from a RamanFit instance that
        already loaded its peaks from the library.

        Returns
        -------
        dict: {peak_name: ([lb_center, lb_width, lb_amp], [ub_center, ub_width, ub_amp])}
        """
        import numpy as np

        labels = list(raman_fit.peak_labels)
        lb = np.asarray(raman_fit.lower_bound, dtype=float)
        ub = np.asarray(raman_fit.upper_bound, dtype=float)

        if lb.size != ub.size or lb.size != 3 * len(labels):
            raise ValueError("RamanFit bounds length mismatch with peak_labels.")

        out = {}
        for k, name in enumerate(labels):
            out[name] = (lb[3*k:3*k+3].tolist(), ub[3*k:3*k+3].tolist())
        return out
    
    def _find_peak_index(self, target):
        """
        Resolve a peak index robustly.
        Matches exact name first, then case-insensitive, then substring containment.
        Returns int index or None.
        """
        if target is None:
            return None

        names = list(self.peak_params)
        # exact
        if target in names:
            return names.index(target)

        # case-insensitive exact
        t = target.lower()
        for i, n in enumerate(names):
            if n.lower() == t:
                return i

        # substring match (handles e.g. 'E2g' vs 'E12g(Γ)')
        for i, n in enumerate(names):
            nl = n.lower()
            if t in nl or nl in t:
                return i

        return None

    ### End NEW METHOD in v0.2.4 ###

    def show_optical_image(self):
        """Display optical image with mapping area overlay."""
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    def lorentzian_raman(self, x, *params):
        """Calculate multi-Lorentzian curve for given parameters.
        
        Args:
            x (ndarray): Wavenumber values
            *params: Fitting parameters in sequence [loc1, scale1, amp1, loc2,...]
            
        Returns:
            ndarray: Sum of Lorentzian components
        """
        result = np.zeros_like(x)
        for i in range(0, len(params), 3):
            loc = params[i]
            scale = params[i+1]
            amp = params[i+2]
            result += (scale / ((x - loc)**2 + scale**2)) * amp / np.pi
        return result

    ### UPDATED METHOD IN v0.2.5 ###
    def remove_background(self, wavenumber, intensity):
        """Remove spectral background via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=wavenumber,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected



    ### UPDATED METHOD IN v0.2.4 ###
    def fit_spectra(
        self,
        initial_p0=None,
        warm_start=False,
        reset_on_fail=True,
        maxfev=6400,
        warm_start_rmse_gate=0.06,
        row_reset=True,
        bound_tol=1e-10
    ):
        """
        Fit all map spectra using self.custom_peaks as bounds.

        Behaviour notes:
        - fitting is ALWAYS performed in peak-normalised space
        - self.normalize affects DISPLAY only (intensity maps)
        - supports initial_p0 as vector or dict package {"p0":..., "peak_order":...}
        - warm-start propagation is gated (RMSE + plausibility) to reduce scanline ledges
        - optional row_reset prevents row-to-row propagation artefacts

        Returns
        -------
        params_map : ndarray
            Fitted parameter cube with shape [Y, X, n_params].
            In notebooks, assign the return value or use `_ = fit_spectra(...)`
            to avoid auto-display.

        """
        if not hasattr(self, "custom_peaks") or not isinstance(self.custom_peaks, dict) or len(self.custom_peaks) == 0:
            raise ValueError("custom_peaks is not set or empty. Provide custom_peaks when initialising RamanMapping.")

        # ---------- helpers ----------
        def _params_plausible(params, lb, ub, n_peaks, tol=1e-10):
            """
            Reject fits that are stuck at bounds or have non-physical widths.
            params are [center, width, amp] repeated.
            """
            params = np.asarray(params, dtype=float)
            for k in range(n_peaks):
                c = params[3*k]
                w = params[3*k + 1]
                a = params[3*k + 2]

                # width must be positive and not absurdly small
                if not np.isfinite(w) or w <= 1e-8:
                    return False

                # centre/width at bounds often indicates a constrained "fallback" minimum
                if abs(c - lb[3*k]) < tol or abs(c - ub[3*k]) < tol:
                    return False
                if abs(w - lb[3*k + 1]) < tol or abs(w - ub[3*k + 1]) < tol:
                    return False

                # amplitude exactly at bound is also suspicious in mapping (can indicate saturation)
                if abs(a - lb[3*k + 2]) < tol or abs(a - ub[3*k + 2]) < tol:
                    return False

            return True

        # ---------- mask + x-axis ----------
        if getattr(self, "_x_trimmed_on_load", False):
            xdata = self.wavenumber
            mask = None
        else:
            mask = DataImporter.mask_by_xrange(self.wavenumber, self.data_range)
            xdata = self.wavenumber[mask]


        # ---------- bounds ----------
        lower_bound, upper_bound = [], []
        for params_range in self.custom_peaks.values():
            lower_bound.extend(params_range[0])
            upper_bound.extend(params_range[1])

        lower_bound = np.asarray(lower_bound, dtype=float)
        upper_bound = np.asarray(upper_bound, dtype=float)
        n_params = lower_bound.size
        n_peaks = len(self.peak_params)

        # baseline p0
        p0_base = (lower_bound + upper_bound) / 2.0

        # ---------- optional: seed from RamanFit export ----------
        if initial_p0 is not None:
            if isinstance(initial_p0, dict):
                peak_order_pkg = initial_p0.get("peak_order", None)
                p0_vec = initial_p0.get("p0", None)

                if p0_vec is None:
                    raise ValueError("initial_p0 dict must contain key 'p0' with a numeric vector.")

                if peak_order_pkg is not None:
                    if [p.lower() for p in peak_order_pkg] != [p.lower() for p in self.peak_params]:
                        raise ValueError(
                            "peak_order mismatch between RamanFit and RamanMapping.\n"
                            f"RamanFit: {list(peak_order_pkg)}\n"
                            f"RamanMapping: {list(self.peak_params)}\n"
                            "Ensure both use the same peak ordering."
                        )

                initial_p0 = p0_vec

            initial_p0 = np.asarray(initial_p0, dtype=float)
            if initial_p0.shape != p0_base.shape:
                raise ValueError(f"initial_p0 shape {initial_p0.shape} does not match expected {p0_base.shape}")
            p0_base = np.clip(initial_p0, lower_bound, upper_bound)

        # outputs
        fitted_params = np.full((self.Y, self.X, n_params), np.nan)

        # ensure maps exist
        if not hasattr(self, "norm_scale_map"):
            self.norm_scale_map = np.full((self.Y, self.X), np.nan)
        if not hasattr(self, "residual_map"):
            self.residual_map = np.full((self.Y, self.X), np.nan)

        # main loop
        p0_current = p0_base.copy()

        for j in range(self.Y):

            # IMPORTANT: prevents a bad seed at end of previous row from contaminating next row
            if warm_start and row_reset:
                p0_current = p0_base.copy()

            for i in range(self.X):
                raw_spec = self.spectra[j, i, :] if mask is None else self.spectra[j, i, :][mask]

                spec_norm, scale = self._preprocess_single_spectrum(xdata, raw_spec)
                if spec_norm is None:
                    self.norm_scale_map[j, i] = np.nan
                    self.residual_map[j, i] = np.nan
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

                self.norm_scale_map[j, i] = scale

                try:
                    params, _ = optimize.curve_fit(
                        self.lorentzian_raman,
                        xdata,
                        spec_norm,
                        p0=p0_current,
                        bounds=(lower_bound, upper_bound),
                        maxfev=maxfev
                    )

                    # residual in fit space (normalised)
                    model_norm = self.lorentzian_raman(xdata, *params)
                    residual_norm = spec_norm - model_norm
                    rmse_norm = np.sqrt(np.mean(residual_norm ** 2))
                    self.residual_map[j, i] = rmse_norm

                    fitted_params[j, i, :] = params

                    # store peak centre + intensity (height)
                    for k in range(n_peaks):
                        idx = 3 * k
                        center, width, amp = params[idx:idx + 3]
                        self.peak_positions[j, i, k] = center

                        peak_height_norm = amp / (np.pi * width)
                        if self.normalize:
                            self.peak_intensities[j, i, k] = peak_height_norm
                        else:
                            self.peak_intensities[j, i, k] = peak_height_norm * scale

                    # derived maps: compute ONCE per pixel (not inside the peak loop)
                    idx_a1g = self._find_peak_index("A1g")
                    idx_e2g = self._find_peak_index("E2g")
                    if idx_e2g is None:
                        idx_e2g = self._find_peak_index("E12g")

                    if (idx_a1g is not None) and (idx_e2g is not None):
                        a1g_pos = self.peak_positions[j, i, idx_a1g]
                        e2g_pos = self.peak_positions[j, i, idx_e2g]
                        a1g_I = self.peak_intensities[j, i, idx_a1g]
                        e2g_I = self.peak_intensities[j, i, idx_e2g]

                        self.Peaks_distance[j, i] = (a1g_pos - e2g_pos) if (np.isfinite(a1g_pos) and np.isfinite(e2g_pos)) else np.nan
                        self.ratio_A1g_E2g[j, i] = (a1g_I / e2g_I) if (np.isfinite(a1g_I) and np.isfinite(e2g_I) and e2g_I > 0) else np.nan
                        self.ratio_E2g_A1g[j, i] = (e2g_I / a1g_I) if (np.isfinite(a1g_I) and np.isfinite(e2g_I) and a1g_I > 0) else np.nan
                    else:
                        self.Peaks_distance[j, i] = np.nan
                        self.ratio_A1g_E2g[j, i] = np.nan
                        self.ratio_E2g_A1g[j, i] = np.nan

                    # gated warm-start: RMSE + plausibility
                    if warm_start:
                        ok_rmse = (rmse_norm <= warm_start_rmse_gate)
                        ok_params = _params_plausible(params, lower_bound, upper_bound, n_peaks, tol=bound_tol)

                        if ok_rmse and ok_params:
                            p0_current = params
                        else:
                            if reset_on_fail:
                                p0_current = p0_base.copy()

                except RuntimeError:
                    self.residual_map[j, i] = np.nan
                    if reset_on_fail:
                        p0_current = p0_base.copy()
                    continue

        self.fitted_params = fitted_params
        n_fit = np.sum(~np.isnan(self.residual_map))
        print(f"Successful fits: {n_fit} / {self.X * self.Y}")
        return fitted_params

    
    def plot_spectrum_fit(self, x, y):
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

        fitted_norm = self.lorentzian_raman(xdata, *params)
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

    def plot_residual_distribution(
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

    def plot_ratio_heatmap(self, ratio_type='A1g/E2g', cmap='viridis', filter_range=None, x_range=None, y_range=None):
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

    ### Added in v0.2.8
    def _iter_coords(self, coord_mode: str = "pixel"):
        """
        Yield (x, y, j, i) for every pixel.

        coord_mode:
        - "pixel": x,y are integer pixel indices
        - "real":  x,y are physical coordinates using step_size
        """
        step = float(self.step_size)
        for j in range(self.Y):
            for i in range(self.X):
                if coord_mode == "real":
                    yield (i * step, j * step, j, i)
                else:
                    yield (i, j, j, i)

    ### Added in v0.2.8
    def export_fit_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        scaled: bool = True,
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
    ) -> str:
        """
        Export fit results for every pixel in wide format:
        x, y, then per-peak parameters on the same row.

        Per-peak columns:
        <peak>_centre, <peak>_fwhm, <peak>_height_scaled, <peak>_height_norm, <peak>_amp, <peak>_scale
        """
        if not hasattr(self, "fitted_params") or self.fitted_params is None:
            raise ValueError("No fitted_params found. Run fit_spectra() first.")

        peak_labels = list(self.peak_params)  # authoritative ordering in your mapping class :contentReference[oaicite:14]{index=14}
        fields = ["x", "y"]

        per_peak_fields = ["centre", "fwhm", "height_scaled", "height_norm", "amp", "scale"]
        for p in peak_labels:
            for f in per_peak_fields:
                fields.append(f"{p}_{f}")

        rows = []
        for x, y, j, i in self._iter_coords(coord_mode=coord_mode):
            params = np.asarray(self.fitted_params[j, i, :], dtype=float)
            if np.any(np.isnan(params)):
                # keep row but leave values empty to preserve grid
                rows.append({"x": x, "y": y})
                continue

            intensity_scale = 1.0
            if scaled and hasattr(self, "norm_scale_map") and np.isfinite(self.norm_scale_map[j, i]):
                intensity_scale = float(self.norm_scale_map[j, i])

            peak_rows = params_to_rows(
                peak_labels = list(self.peak_params),
                params=params,
                intensity_scale=intensity_scale,
            )

            r = {"x": x, "y": y}
            for pr in peak_rows:
                name = pr.peak
                r[f"{name}_centre"] = pr.centre
                r[f"{name}_fwhm"] = pr.fwhm
                r[f"{name}_height_scaled"] = pr.height_scaled
                r[f"{name}_height_norm"] = pr.height_norm
                r[f"{name}_amp"] = pr.amp
                r[f"{name}_scale"] = pr.scale

            rows.append(r)

        meta = {
            "map_kind": "fit_params",
            "spectrum_type": getattr(self, "spectrum_type", None),
            "x_quantity": getattr(self, "x_quantity", None),
            "x_unit": getattr(self, "x_unit", None),
            "coord_mode": coord_mode,
            "step_size": getattr(self, "step_size", None),
            "step_unit": getattr(self, "step_unit", "um"),
            "scaled": scaled,
            "peak_labels": peak_labels,
            "background_remove": getattr(self, "background_remove", None),
            "baseline_method": getattr(self, "baseline_method", None),
            "smoothing": getattr(self, "smoothing", None),
            "smooth_window": getattr(self, "smooth_window", None),
            "smooth_poly": getattr(self, "smooth_poly", None),
        }
        meta = {k: v for k, v in meta.items() if v is not None}

        return write_table(
            rows,
            out_path,
            fieldnames=fields,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )
    
    def plot_heatmap(self, data_type='position', cmap='viridis', filter_range=None, 
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

                    y_norm = self.lorentzian_raman(specific_wavenumber, *params)

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


########################################################################################################################
############################################ Raman Integration without Peak fitting ####################################
########################################################################################################################

class Raman_Integration:
    """Raman mapping analysis through spectral integration.
    
    Attributes:
        filename (str): Path to input file
        integration_range (tuple): Spectral integration range (min, max) in cm⁻¹
        step_size (float): Physical step size in micrometers
        poly_degree (int): Background polynomial degree
        background_remove (bool): Enable background subtraction
        X (int): Map width in pixels
        Y (int): Map height in pixels
        wavenumber (ndarray): Spectral axis in cm⁻¹
        spectra (ndarray): Raw spectral data [Y, X, points]
        image_viewer (MappingImage): Optical image handler
        integration_area (ndarray): Integrated intensities [Y, X]
    """   
    def __init__(self, filename, integration_range,
                step_size=0.3, header=False,
                poly_degree=3, background_remove=True,
                baseline_method="poly"):
        """Initialize Raman integration analyzer.
        
        Args:
            filename: Path to .wdf file
            integration_range: Spectral range (min, max) in cm⁻¹
            step_size: Physical step size in micrometers
            poly_degree: Background polynomial degree
            background_remove: Enable background subtraction
        """
        self.filename = filename
        self.integration_range = integration_range
        self.step_size = step_size
        self.poly_degree = poly_degree
        self.background_remove = background_remove

        # --- identity metadata for exports (added in v0.2.8) ---
        self.spectrum_type = "Raman"
        self.x_quantity = "Raman shift"
        self.x_unit = "cm^-1"
        self.step_unit = "um"


        # integration_range is known here; trim at load-time.
        loader = MappingFileLoader(filename, x_range=self.integration_range, axis="wavenumber")
        self._x_trimmed_on_load = True

        self.X = loader.X
        self.Y = loader.Y
        self.wavenumber = loader.xdata
        self.spectra = loader.spectra
        self.image_viewer = MappingImage(filename) if filename.endswith(".wdf") else None
        self.integration_area = np.zeros((self.Y, self.X))

        # New in v0.2.5 Baseline configuration (single source of truth; backward compatible)
        self.baseline_method = baseline_method
        self._baseline_method, self._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree
        )

    ### New classmethod in v0.2.7 ###
    @classmethod
    def from_arrays(
        cls,
        spectra,
        xdata,
        X,
        Y,
        *,
        integration_range,
        step_size=0.3,
        poly_degree=3,
        background_remove=True,
        baseline_method="poly",
        clip_nonnegative=False,
    ):
        """
        Construct Raman_Integration from in-memory arrays (no file IO).

        Parameters
        ----------
        spectra : ndarray
            Mapping cube with shape [Y, X, N]
        xdata : ndarray
            Wavenumber axis (cm^-1) with shape [N]
        X, Y : int
            Map dimensions
        integration_range : tuple(float, float)
            (min, max) cm^-1; applied immediately (trim at construction, consistent with __init__)
        """
        obj = cls.__new__(cls)

        obj.filename = None
        obj.integration_range = integration_range
        obj.step_size = step_size
        obj.poly_degree = poly_degree
        obj.background_remove = background_remove

        # Baseline configuration (same pattern as __init__)
        obj.baseline_method = baseline_method
        obj._baseline_method, obj._baseline_kwargs = BaselineAPI.parse_spec(
            baseline_method,
            poly_degree=poly_degree
        )

        obj.X = int(X)
        obj.Y = int(Y)

        wn = np.asarray(xdata, dtype=float).ravel()
        cube = np.asarray(spectra, dtype=float)

        # Validate shapes
        if cube.ndim != 3:
            raise ValueError("spectra must be a 3D array with shape [Y, X, N].")
        if cube.shape[0] != obj.Y or cube.shape[1] != obj.X:
            raise ValueError(f"spectra shape {cube.shape[:2]} inconsistent with (Y,X)=({obj.Y},{obj.X}).")
        if cube.shape[2] != wn.size:
            raise ValueError("spectra third dimension (N) must match len(xdata).")

        # Trim at load-time (same semantics as filename-based __init__)
        wmin, wmax = integration_range
        mask = (wn >= wmin) & (wn <= wmax)
        if not np.any(mask):
            raise ValueError(
                f"integration_range {integration_range} does not overlap provided wavenumber axis "
                f"[{float(np.min(wn)):.3g}, {float(np.max(wn)):.3g}]."
            )

        obj.wavenumber = wn[mask]
        obj.spectra = cube[:, :, mask]
        obj._x_trimmed_on_load = True

        if clip_nonnegative:
            obj.spectra = np.clip(obj.spectra, a_min=0.0, a_max=None)

        obj.image_viewer = None
        obj.integration_area = np.zeros((obj.Y, obj.X), dtype=float)

        return obj

    def show_optical_image(self):
        if self.image_viewer:
            self.image_viewer.show_optical_image()

    ## Updated in v0.2.5 ##
    def remove_background(self, wavenumber, intensity):
        """Background removal via BaselineAPI (always clips to non-negative)."""
        result = BaselineAPI.subtract(
            x=wavenumber,
            y=intensity,
            method=self._baseline_method,
            clip_nonnegative=True,
            **self._baseline_kwargs
        )
        return result.y_corrected
    
    ## Updated in v0.2.6 ##
    def calculate_integration(self):
        """Calculate integrated area using Simpson's rule.

        Stores results in integration_area array.
        """
        wavenumber = np.asarray(self.wavenumber, dtype=float).ravel()

        mask = DataImporter.mask_by_xrange(wavenumber, self.integration_range)
        wavenumber_subset = wavenumber[mask]

        for j in range(self.Y):
            for i in range(self.X):
                spectra = np.asarray(self.spectra[j, i, :], dtype=float).ravel()
                spectra_subset = spectra[mask]

                if self.background_remove:
                    spectra_subset = self.remove_background(wavenumber_subset, spectra_subset)

                self.integration_area[j, i] = np.abs(simpson(spectra_subset, wavenumber_subset))


    def plot_integration_heatmap(self, cmap='viridis', filter_range=None, x_range=None, y_range=None):
        """Visualize 2D map of integrated intensities.
        
        Args:
            cmap: Matplotlib colormap name
            filter_range: Data display range [min, max]
            x_range: X display range [start, end]
            y_range: Y display range [start, end]
        """
        # Filter data range
        data = self.integration_area
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
        cbar = plt.colorbar(im, label='Integration Area (a.u.)')
        plt.xlabel("X Position (μm)")
        plt.ylabel("Y Position (μm)")
        plt.title(f"Integration Area Heatmap ({self.integration_range[0]} - {self.integration_range[1]} cm⁻¹)")
        plt.show()

    def plot_spectrum(self, x, y):
        """Plot raw and processed spectra for single map point.
        
        Args:
            x: X coordinate (0-indexed)
            y: Y coordinate (0-indexed)
            
        Raises:
            ValueError: For invalid coordinates
        """
        if x < 0 or x >= self.X or y < 0 or y >= self.Y:
            raise ValueError("Invalid coordinates. Please ensure x and y are within the mapping range.")

        # Get the original spectrum data
        wavenumber = self.wavenumber[:]
        spectra = self.spectra[y][x][:]

        # Subset within integration range
        mask = DataImporter.mask_by_xrange(wavenumber, self.integration_range)
        wavenumber_subset = wavenumber[mask]
        spectra_subset = spectra[mask]

        # Keep raw copy for plotting
        spectra_raw = spectra_subset.copy()

        # Background removal (define spectra_bg_removed in all cases)
        if self.background_remove:
            spectra_bg_removed = self.remove_background(wavenumber_subset, spectra_subset)
        else:
            spectra_bg_removed = spectra_subset

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(wavenumber_subset, spectra_raw, 'b-', label='Original Spectrum')
        if self.background_remove:
            plt.plot(wavenumber_subset, spectra_bg_removed, 'r--', label='Background Removed')
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.title(f"Spectrum at (X={x}, Y={y})")
        plt.legend()
        plt.show()


    ### Added in v0.2.8
    def export_integration_map(
        self,
        out_path: str,
        *,
        coord_mode: str = "pixel",
        headers: bool = True,
        include_header: bool = True,
        delimiter: str | None = None,
        column_name: str = "integration_area",
    ) -> str:
        """
        Export integration_area in wide format:
        x, y, integration_area
        """
        if not hasattr(self, "integration_area") or self.integration_area is None:
            raise ValueError("No integration_area found. Run calculate_integration() first.")

        # coordinate iterator local to this class
        step = float(self.step_size)
        rows = []
        for j in range(self.Y):
            for i in range(self.X):
                x, y = (i * step, j * step) if coord_mode == "real" else (i, j)
                rows.append({"x": x, "y": y, column_name: float(self.integration_area[j, i])})

        fields = ["x", "y", column_name]

        meta = {
            "map_kind": "integration",
            "spectrum_type": getattr(self, "spectrum_type", None),
            "x_unit": getattr(self, "x_unit", None),
            "coord_mode": coord_mode,
            "step_size": getattr(self, "step_size", None),
            "step_unit": getattr(self, "step_unit", "um"),
            "integration_range": getattr(self, "integration_range", None),
            "background_remove": getattr(self, "background_remove", None),
            "baseline_method": getattr(self, "baseline_method", None),
        }
        meta = {k: v for k, v in meta.items() if v is not None}

        return write_table(
            rows,
            out_path,
            fieldnames=fields,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )
