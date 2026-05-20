"""
A module for analyzing photoluminescence (PL) spectra through Lorentzian curve fitting.

This module provides tools for preprocessing PL data (smoothing, background subtraction),
fitting Exciton and Trion peaks using Lorentzian functions, and visualizing the results.

Classes:
    PLfit: Main class for processing, fitting, and visualizing PL spectra.
    DataImporter: Class for importing Raman data from .wdf and .txt files (single spectrum only)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ramanpl.exporter import params_to_rows, write_rows, write_table
from ramanpl.preprocessing import SpectralDataset, Pipeline, build_legacy_single_spectrum_pipeline
from ramanpl.schema import (
    normalise_baseline_spec,
    normalise_peak_profile,
    normalise_preprocess_backend,
)
from ..peak_models import sum_peaks, single_peak
from ._single_fit_core import (
    build_single_fit_export_metadata,
    export_p0_dict,
    make_compat_data_importer,
    run_multistart_curve_fit,
)

DataImporter = make_compat_data_importer(axis="energy", default_readlines=(300, 780))


class PLfit:
    """A class for processing and fitting photoluminescence spectra.

    Supports:
    - preprocessing via legacy flags or a user-supplied preprocessing Pipeline
    - peak fitting with either Lorentzian or pseudo-Voigt peak profiles
    - custom peak definitions, peak removal, and deterministic peak ordering
    - export of fitted parameters and mapping-ready initial guesses

    Notes
    -----
    - Fitting is always performed in peak-normalised intensity space.
    - `normalize` controls display/output scaling only.
    - When `preprocessing` is supplied, it overrides the legacy smoothing/background flags.
    """
    def __init__(self, spectra, energy, background_remove=False,
            baseline_method={"method": "poly", "poly_order": 1},
            gaussian_sigma=50, smoothing=False,
            smooth_window=11, smooth_order=3, normalize=True, preprocessing=None,
            preprocessing_backend: str = "native",
            custom_peaks=None, remove_peaks=None, peak_order=None,
            peak_profile: str = "lorentzian"
            ):
        """Initialize PLfit object with data and processing parameters.

        Parameters:
            spectra (array-like): PL intensity values (y-axis)
            energy (array-like): Corresponding energy values in eV (x-axis)
            background_remove (bool): Enable background subtraction (default: False)
            baseline_method : str or dict, optional
                Baseline specification. Preferred modern style is a dict, for example:
                {"method": "poly", "poly_order": 1},
                or
                {"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6}.
            preprocessing (Pipeline or None): Optional preprocessing pipeline. If supplied,
                it overrides the legacy smoothing/background_remove settings.
            peak_profile (str): Peak model type: 'lorentzian' or 'pvoigt'.
            normalize (bool): Controls DISPLAY/OUTPUT scaling only. Fitting is always
                performed in peak-normalised space.
            poly_degree : int, optional
                Deprecated. Use baseline_method={"method": "poly", "poly_degree": degree}
                instead. If supplied, a DeprecationWarning is issued and the value is folded
                into baseline_method for backwards compatibility.
            gaussian_sigma (int): Sigma for Gaussian filter (default: 50)
            smoothing (bool): Enable Savitzky-Golay smoothing (default: False)
            smooth_window (int): Window size for smoothing filter (default: 11)
            smooth_order (int): Polynomial order for smoothing (default: 3)

        Raises:
            ValueError: If invalid baseline method is specified
        """
        self.raw_spectra = np.array(spectra)
        self.energy = np.array(energy)
        self.processed_spectra = np.array(spectra.copy())
        # Store pristine copies before pipeline application so apply_choice() can re-apply from scratch
        self._raw_spectra_pristine = np.asarray(spectra, dtype=float).ravel()
        self._x_axis_pristine = np.asarray(energy, dtype=float).ravel()

        self._smoothed_spectra = None
        self._baseline = None
        self._corrected_spectra = None

        self.spectrum_type = "Photoluminescence"
        self.x_quantity = "Photon energy"
        self.x_unit = "eV"

        self.background_remove = background_remove
        self.baseline_method = normalise_baseline_spec(
            baseline_method,
            gaussian_sigma=gaussian_sigma,
        )

        if str(self.baseline_method.get("method", "")).lower() == "poly":
            self.poly_order = int(self.baseline_method.get("poly_order", 1))
        else:
            self.poly_order = None

        self.gaussian_sigma = gaussian_sigma
        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order
        self.peak_order = peak_order
        self.preprocessing_backend = normalise_preprocess_backend(preprocessing_backend)

        self.peak_profile = normalise_peak_profile(peak_profile)
        self.params_per_peak = 3 if self.peak_profile == "lorentzian" else 4

        ds0 = SpectralDataset(
            x=self.energy,
            y=np.asarray(self.processed_spectra, dtype=float).ravel(),
            modality="PL",
            axis_kind="energy_eV",
            meta={
                "x_trimmed_on_load": bool(getattr(self, "_x_trimmed_on_load", False)),
            },
        )

        if preprocessing is None:
            pipe = build_legacy_single_spectrum_pipeline(
                data_range=None,
                smoothing=bool(smoothing),
                smooth_window=int(smooth_window),
                smooth_order=int(smooth_order),
                background_remove=bool(background_remove),
                baseline_method=self.baseline_method,
                poly_degree=None,
                gaussian_sigma=int(gaussian_sigma),
                backend=self.preprocessing_backend,
            )
        elif isinstance(preprocessing, Pipeline):
            pipe = preprocessing
        else:
            raise TypeError(
                "preprocessing must be None or a ramanpl.preprocessing.Pipeline instance."
            )

        self.preprocessing = pipe
        self.preprocessing_backend = getattr(pipe, "backend", self.preprocessing_backend)
        try:
            self.preprocessing_recipe = pipe.to_dict()
        except Exception:
            self.preprocessing_recipe = None

        ds = pipe.apply(ds0)

        self.energy = np.asarray(ds.x, dtype=float).ravel()
        self.processed_spectra = np.asarray(ds.y, dtype=float).ravel()

        crop_mask = ds.meta.get("crop_mask", None)
        if crop_mask is not None:
            self.raw_spectra = np.asarray(self.raw_spectra, dtype=float).ravel()[crop_mask]
        else:
            self.raw_spectra = np.asarray(self.raw_spectra, dtype=float).ravel()

        self._smoothed_spectra = ds.meta.get("_smoothed_last", None)
        self._baseline = ds.meta.get("_baseline_last", None)
        self.preprocessing_backend_resolved = ds.meta.get("preprocessing_backend", None)
        self.preprocessing_backend_info = ds.meta.get("preprocessing_backend_info", None)
        self._backend_outcome = self.preprocessing_backend_info

        if (self._smoothed_spectra is not None) or (self._baseline is not None):
            self._corrected_spectra = self.processed_spectra.copy()
        else:
            self._corrected_spectra = None

        self.normalize = normalize

        self.peak_intensity = np.max(self.processed_spectra)
        if self.peak_intensity <= 0:
            raise ValueError(
                "Peak intensity is non-positive after preprocessing; cannot normalise for fitting."
            )
        self.intensity_normal = self.processed_spectra / self.peak_intensity

        self.custom_peaks = custom_peaks

        if custom_peaks is None:
            if self.peak_profile == "lorentzian":
                self.lower_bound = [1.95, 0, 0,  1.8, 0, 0]
                self.upper_bound = [2.1, 0.05, 10, 2.0, 0.2, 10]
                self.peak_labels = ["trion", "exciton"]
            else:
                self.lower_bound = [1.95, 0, 0, 0.0,  1.8, 0, 0, 0.0]
                self.upper_bound = [2.1, 0.05, 10, 1.0, 2.0, 0.2, 10, 1.0]
                self.peak_labels = ["trion", "exciton"]
        else:
            if not isinstance(custom_peaks, dict) or len(custom_peaks) == 0:
                raise ValueError("custom_peaks must be a non-empty dict: {name: ([lb...],[ub...])}")

            if peak_order is None:
                self.peak_order = list(custom_peaks.keys())
            else:
                self.peak_order = list(peak_order)
                missing = [k for k in self.peak_order if k not in custom_peaks]
                if missing:
                    raise ValueError(f"peak_order contains keys not in custom_peaks: {missing}")

            self.peak_labels = list(self.peak_order)
            self.lower_bound, self.upper_bound = [], []
            expected = self.params_per_peak

            for name in self.peak_labels:
                lb, ub = custom_peaks[name]
                if len(lb) != expected or len(ub) != expected:
                    if expected == 3:
                        raise ValueError(f"Peak '{name}' bounds must be length-3 lists: [centre, width, amp]")
                    else:
                        raise ValueError(f"Peak '{name}' bounds must be length-4 lists: [centre, width, amp, eta]")
                self.lower_bound += list(lb)
                self.upper_bound += list(ub)

        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

        self.remove_peaks_list = list(remove_peaks) if remove_peaks is not None else []
        if self.remove_peaks_list:
            self.remove_peaks(*self.remove_peaks_list)

        self.params_fit = None
        self.params_cov = None

    # ------------------------------------------------------------------
    # v0.6.2 — Baseline autotune
    # ------------------------------------------------------------------

    def autotune_baseline(
        self,
        *,
        methods=None,
        lam_grid=None,
        plot: bool = True,
        fit_spectrum_kwargs=None,
    ):
        """
        Score a grid of baseline candidates on this spectrum. Does NOT modify self.

        Call self.apply_choice(result.winner) to commit the winning configuration.

        Parameters
        ----------
        methods : list[str] or None
            Subset of {'asls','arpls','airpls','poly','gaussian'}. None → all.
        lam_grid : list[float] or None
            Override lam sweep for iterative methods.
        plot : bool
            If True, return a comparison figure in result.figure.
        fit_spectrum_kwargs : dict or None
            Extra kwargs for the internal fitter (e.g. n_starts).

        Returns
        -------
        BaselineAutotuneResult
        """
        from ramanpl._autotune import autotune_baseline_for_object

        result = autotune_baseline_for_object(
            self,
            seed_coord=None,
            methods=methods,
            lam_grid=lam_grid,
            plot=plot,
            fit_spectrum_kwargs=fit_spectrum_kwargs,
        )
        self._last_autotune_result = result
        return result

    def apply_choice(self, choice: dict) -> None:
        """
        Commit a baseline spec to self.preprocessing and re-apply it.

        Re-applies the full pipeline from the pristine raw spectrum so all
        derived attributes are consistent.

        Parameters
        ----------
        choice : dict
            Baseline spec with at least a 'method' key, e.g.
            {'method': 'airpls', 'lam': 1e5, 'niter': 50}.

        Raises
        ------
        ValueError
            If choice is not a valid spec dict, or if the pipeline has zero or
            more than one BaselineSubtract steps.
        """
        if not isinstance(choice, dict) or "method" not in choice:
            raise ValueError(
                "choice must be a baseline spec dict with at least a 'method' key."
            )

        from ramanpl._autotune import _swap_baseline_step_in_pipeline
        from ramanpl.preprocessing import SpectralDataset
        from ramanpl.schema import baseline_spec_to_runtime, normalise_baseline_spec

        new_pipe = _swap_baseline_step_in_pipeline(self.preprocessing, choice)

        # Re-apply from pristine (pre-pipeline) data
        ds0 = SpectralDataset(
            x=self._x_axis_pristine,
            y=self._raw_spectra_pristine,
            modality="PL",
            axis_kind="energy_eV",
            meta={"x_trimmed_on_load": False},
        )
        ds = new_pipe.apply(ds0)

        # Update pipeline state
        self.preprocessing = new_pipe
        try:
            self.preprocessing_recipe = new_pipe.to_dict()
        except Exception:
            self.preprocessing_recipe = None

        # Refresh legacy baseline attributes
        self.baseline_method = normalise_baseline_spec(choice)
        self._baseline_method, self._baseline_kwargs = baseline_spec_to_runtime(
            self.baseline_method
        )

        # Refresh all derived attributes
        self.energy = np.asarray(ds.x, dtype=float).ravel()
        self.processed_spectra = np.asarray(ds.y, dtype=float).ravel()
        self._smoothed_spectra = ds.meta.get("_smoothed_last", None)
        self._baseline = ds.meta.get("_baseline_last", None)
        self._corrected_spectra = (
            self.processed_spectra.copy()
            if (self._smoothed_spectra is not None or self._baseline is not None)
            else None
        )
        self.preprocessing_backend_resolved = ds.meta.get("preprocessing_backend", None)
        self.preprocessing_backend_info = ds.meta.get("preprocessing_backend_info", None)
        self._backend_outcome = self.preprocessing_backend_info

        self.peak_intensity = float(np.max(self.processed_spectra))
        if self.peak_intensity <= 0:
            raise ValueError(
                "Peak intensity is non-positive after re-applying pipeline with new baseline."
            )
        self.intensity_normal = self.processed_spectra / self.peak_intensity

    def feature_table(self, *, ratios=None, separations=None):
        """
        Return fitted peak descriptors as a single-row DataFrame.

        Parameters
        ----------
        ratios : list of (str, str) or None
            Each ``(P1, P2)`` adds ``{P1}_{P2}_ratio``.
        separations : list of (str, str) or None
            Each ``(P1, P2)`` adds ``{P1}_{P2}_separation``.

        Returns
        -------
        pandas.DataFrame
            One row with per-peak, derived, and QA columns.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("No fitted parameters. Run fit_spectrum() first.")

        import pandas as pd
        from ramanpl import descriptors

        peak_labels = list(self.peak_labels)
        descriptors.validate_peak_pairs(
            list(ratios or []) + list(separations or []), peak_labels
        )

        fitted = self.get_fitted_parameters()
        per_peak = {
            name: {
                "centre": d["position"],
                "fwhm": d["fwhm"],
                "peak_height": d["peak_height"],
                "peak_height_norm": d["height_norm"],
            }
            for name, d in fitted.items()
        }
        diag = getattr(self, "fit_diagnostics", None) or {}
        rmse_val = float(diag.get("rmse", float("nan")))
        qa = {
            "rmse": rmse_val,
            "ok": bool(np.isfinite(rmse_val)),
            "n_starts": float(diag.get("n_starts", float("nan"))),
            "n_params_at_bounds": float(diag.get("n_params_at_bounds", float("nan"))),
        }
        feat = descriptors.build_feature_row(
            per_peak, qa, peak_labels, ratios=ratios, separations=separations
        )
        return pd.DataFrame.from_records([feat])

    ### UPDATED METHOD in v0.3.3 ###
    def update_bounds(self, **kwargs):
        """Update fitting constraints for specific peaks.

        Parameters:
            **kwargs: Peak name and bounds tuple pairs (e.g., Trion=([lb1, lb2, lb3], [ub1, ub2, ub3]))

        Raises:
            ValueError: For unrecognized peak names or invalid bound formats

        Example:
            >>> pl.update_bounds(Trion=([1.9, 0.01, 1], [2.0, 0.1, 5]),
            ...                  Exciton=([1.7, 0.01, 1], [1.9, 0.1, 5]))
        """
        stride = self.params_per_peak

        for peak_name, new_bounds in kwargs.items():
            peak_key = str(peak_name).lower()
            labels_lower = [p.lower() for p in self.peak_labels]

            if peak_key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            if not (isinstance(new_bounds, (tuple, list)) and len(new_bounds) == 2):
                raise ValueError(f"Bounds for '{peak_name}' must be (lb, ub).")

            lb_new, ub_new = new_bounds
            if len(lb_new) != stride or len(ub_new) != stride:
                raise ValueError(
                    f"Peak '{peak_name}' bounds must be length-{stride} lists."
                )

            idx = labels_lower.index(peak_key)
            s = stride * idx

            self.lower_bound[s : s + stride] = list(lb_new)
            self.upper_bound[s : s + stride] = list(ub_new)
            self.p0[s : s + stride] = [(lb_new[i] + ub_new[i]) / 2 for i in range(stride)]

    # UPDATED IN V0.3.3
    def remove_peaks(self, *peak_names):
        """
        Remove peaks from the fitting model.

        Parameters
        ----------
        *peak_names : str
            Peak names to remove (case-insensitive match allowed).

        Raises
        ------
        ValueError
            If a peak name is not present.
        """
        stride = self.params_per_peak

        for peak_name in peak_names:
            labels_lower = [p.lower() for p in self.peak_labels]  # refresh each loop (simplest)
            key = str(peak_name).lower()
            if key not in labels_lower:
                raise ValueError(
                    f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}"
                )

            idx = labels_lower.index(key)

            # remove blocks
            del self.p0[stride * idx : stride * idx + stride]
            del self.lower_bound[stride * idx : stride * idx + stride]
            del self.upper_bound[stride * idx : stride * idx + stride]

            # remove label to keep everything aligned
            del self.peak_labels[idx]

        # Optional: if you maintain peak_order elsewhere, keep it consistent too
        if hasattr(self, "peak_order") and isinstance(self.peak_order, list):
            self.peak_order = list(self.peak_labels)

    def _model(self, x, *params):
        """Internal model dispatcher (Lorentzian or pseudo-Voigt)."""
        return sum_peaks(
            np.asarray(x),
            params,
            profile=("pvoigt" if self.peak_profile == "pvoigt" else "lorentzian"),
            stride=self.params_per_peak,
        )

    def _jac(self, x, *params):
        """Analytical Jacobian for Lorentzian fits (used automatically by fit_spectrum)."""
        from ..peak_models import sum_peaks_jac_lorentzian
        return sum_peaks_jac_lorentzian(x, *params)

    def fit_spectrum(
        self,
        *,
        n_starts: int = 1,
        p0_strategy: str = "midpoint",
        random_state=None,
        diagnose_bounds: bool = True,
        bounds_tol: float = 1e-6,
        return_diagnostics: bool = False,
        maxfev: int = 6400,
    ):
        """
        Perform bounded least-squares fitting with optional multi-start.

        maxfev:
            Maximum number of function evaluations passed to curve_fit.
            Reduce (e.g. 1600) to speed up fitting of well-conditioned spectra.
            Lorentzian fits automatically use an analytical Jacobian, which
            already reduces cost significantly regardless of this value.
        """
        jac = self._jac if self.peak_profile == "lorentzian" else None
        best_params, best_cov, diagnostics = run_multistart_curve_fit(
            model=self._model,
            x=self.energy,
            y=self.intensity_normal,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            base_p0=self.p0,
            n_starts=n_starts,
            p0_strategy=p0_strategy,
            random_state=random_state,
            maxfev=maxfev,
            jac=jac,
            diagnose_bounds=diagnose_bounds,
            bounds_tol=bounds_tol,
            fail_label="PLfit.fit_spectrum",
        )

        self.params_fit = best_params
        self.params_cov = best_cov
        self.fit_diagnostics = diagnostics

        if return_diagnostics:
            return best_params, best_cov, diagnostics
        return best_params, best_cov
    
    ### NEW METHOD in v0.2.9 ###
    def get_fitted_spectrum(self):
        """
        Return fitted spectrum on the same x-grid as the input data.

        Returns
        -------
        x : np.ndarray
            Energy axis (eV)
        y_fit : np.ndarray
            Fitted intensity in the SAME units as self.processed_spectra.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("PLfit has not been fitted yet. Run fit_spectrum() first.")

        x = np.asarray(self.energy, dtype=float).ravel()

        y_fit_norm = self._model(x, *self.params_fit)            # normalised fit space
        y_fit = y_fit_norm * float(self.peak_intensity)          # back to processed intensity scale

        return x.copy(), np.asarray(y_fit, dtype=float).ravel().copy()

    ### Updated v0.3.3 ###
    def get_fitted_parameters(self):
        """
        Return fitted peak parameters as a structured dict.

        Notes
        -----
        Parameter vector layout depends on peak_profile:
        - lorentzian: (centre, HWHM, amp_area) per peak
        - pvoigt:     (centre, FWHM, amp_area, eta) per peak

        Reported peak_height is the *peak maximum* in processed intensity units.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("Fit not available. Run fit_spectrum() first.")
        if not hasattr(self, "peak_labels") or not self.peak_labels:
            raise RuntimeError("No peak labels found; cannot map parameters to peaks.")

        p = np.asarray(self.params_fit, dtype=float).ravel()

        profile = str(getattr(self, "peak_profile", "lorentzian")).lower().strip()
        stride = int(getattr(self, "params_per_peak", 3))
        if profile not in ("lorentzian", "pvoigt"):
            raise RuntimeError(f"Unsupported peak_profile '{profile}' in get_fitted_parameters().")

        expected = stride * len(self.peak_labels)
        if p.size < expected:
            raise RuntimeError(
                f"params_fit has length {p.size}, but expected at least {expected} "
                f"for {len(self.peak_labels)} peaks (stride={stride})."
            )

        fit_scale = float(self.peak_intensity) if hasattr(self, "peak_intensity") else 1.0

        if profile == "pvoigt":
            pass

        out = {}

        for i, name in enumerate(self.peak_labels):
            block = p[stride * i : stride * (i + 1)]
            loc = float(block[0])
            width = float(block[1])       # HWHM (lorentzian) or FWHM (pvoigt)
            amp_area = float(block[2])    # area-like amplitude (both)

            if profile == "lorentzian":
                fwhm = 2.0 * width
                height_norm = (amp_area / (np.pi * width)) if width != 0 else np.nan
                row = dict(
                    position=loc,
                    fwhm=float(fwhm),
                    scale=float(width),          # HWHM
                    amp=float(amp_area),
                    height_norm=float(height_norm),
                    peak_height=float(height_norm * fit_scale),
                )
            else:
                eta = float(block[3])
                fwhm = width  # pVoigt width parameter is FWHM in your peak_models
                y_comp_norm = single_peak(self.energy, block, profile="pvoigt")
                height_norm = float(np.max(y_comp_norm))
                row = dict(
                    position=loc,
                    fwhm=float(fwhm),
                    fwhm_param=float(width),     # explicit: stored width is FWHM
                    eta=float(eta),
                    amp=float(amp_area),
                    height_norm=float(height_norm),
                    peak_height=float(height_norm * fit_scale),
                )

            out[name] = row

        return out


    ### Updated in v0.3.4 to handle both profiles and include metadata
    def fit_table(self, params=None, *, scaled: bool = True):
        """
        Return per-peak fitted parameters as a list of dicts.

        Notes
        -----
        - Lorentzian rows include: Peak, Position(eV), FWHM(eV), Scale, Amp,
        PeakHeight_norm, PeakHeight
        - pseudo-Voigt rows include: Peak, Position(eV), FWHM(eV), Eta, Amp,
        PeakHeight_norm, PeakHeight
        """
        if params is None:
            if not hasattr(self, "params_fit") or self.params_fit is None:
                raise ValueError("No fitted parameters found. Run fit_spectrum() first.")
            params = self.params_fit

        if self.peak_profile == "pvoigt":
            fitted = self.get_fitted_parameters()
            rows = []
            for peak in self.peak_labels:
                d = fitted[peak]
                rows.append({
                    "Peak": peak,
                    "Position(eV)": d["position"],
                    "FWHM(eV)": d["fwhm"],
                    "Eta": d.get("eta", ""),
                    "Amp": d["amp"],
                    "PeakHeight_norm": d["height_norm"],
                    "PeakHeight": d["peak_height"],
                })
            return rows

        intensity_scale = 1.0
        if scaled and hasattr(self, "peak_intensity") and self.peak_intensity is not None:
            intensity_scale = float(self.peak_intensity)

        rows = params_to_rows(
            peak_labels=self.peak_labels,
            params=params,
            intensity_scale=intensity_scale,
        )

        return [
            {
                "Peak": r.peak,
                "Position(eV)": r.centre,
                "FWHM(eV)": r.fwhm,
                "Scale": r.scale,
                "Amp": r.amp,
                "PeakHeight_norm": r.peak_height_norm,
                "PeakHeight": r.peak_height,
            }
            for r in rows
        ]

    ### Updated in v0.3.4 to handle both profiles and include metadata
    def export_fit(
        self,
        out_path: str,
        *,
        params=None,
        delimiter: str | None = None,
        include_header: bool = True,
        scaled: bool = True,
        headers: bool = True,
    ) -> str:
        """
        Export fitted parameters to CSV or TXT/TSV.

        Notes
        -----
        - For Lorentzian fits, exports the standard columns:
        Peak, Centre, FWHM, Scale, Amp, PeakHeight_norm, PeakHeight
        - For pseudo-Voigt fits, exports:
        Peak, Centre, FWHM, Eta, Amp, PeakHeight_norm, PeakHeight

        headers:
            If True, write a metadata header block in TXT/TSV outputs.
            Ignored for CSV.
        """
        if params is None:
            if not hasattr(self, "params_fit") or self.params_fit is None:
                raise ValueError("No fitted parameters found. Run fit_spectrum() first.")
            params = self.params_fit

        intensity_scale = 1.0
        if scaled and hasattr(self, "peak_intensity") and self.peak_intensity is not None:
            intensity_scale = float(self.peak_intensity)

        # # Metadata
        # meta = {
        #     "spectrum_type": getattr(self, "spectrum_type", None),
        #     "x_quantity": getattr(self, "x_quantity", None),
        #     "x_unit": getattr(self, "x_unit", None),

        #     "background_remove": getattr(self, "background_remove", None),
        #     "baseline_method": getattr(self, "baseline_method", None),
        #     "poly_degree": getattr(self, "poly_degree", None),  # deprecated legacy metadata
        #     "gaussian_sigma": getattr(self, "gaussian_sigma", None),

        #     "smoothing": getattr(self, "smoothing", None),
        #     "smooth_window": getattr(self, "smooth_window", None),
        #     "smooth_order": getattr(self, "smooth_order", None),

        #     "normalize": getattr(self, "normalize", None),
        #     "intensity_scale(peak_intensity)": getattr(self, "peak_intensity", None),

        #     "peak_labels": getattr(self, "peak_labels", None),
        #     "custom_peaks": "True" if getattr(self, "custom_peaks", None) is not None else "False",
        #     "remove_peaks": getattr(self, "remove_peaks_list", None),
        #     "peak_profile": getattr(self, "peak_profile", None),
        #     "preprocessing": getattr(self, "preprocessing", None),
        # }
        # meta = {k: v for k, v in meta.items() if v is not None}

        meta = self._build_export_metadata(include_legacy=True)

        # ---- pseudo-Voigt path ----
        if self.peak_profile == "pvoigt":
            fitted = self.get_fitted_parameters()
            rows_dict = []
            for peak in self.peak_labels:
                d = fitted[peak]
                rows_dict.append({
                    "Peak": peak,
                    "Centre": d["position"],
                    "FWHM": d["fwhm"],
                    "Eta": d.get("eta", ""),
                    "Amp": d["amp"],
                    "PeakHeight_norm": d["height_norm"],
                    "PeakHeight": d["peak_height"],
                })

            return write_table(
                rows_dict,
                out_path,
                fieldnames=["Peak", "Centre", "FWHM", "Eta", "Amp", "PeakHeight_norm", "PeakHeight"],
                delimiter=delimiter,
                include_header=include_header,
                meta=meta,
                headers=headers,
                meta_in_csv=False,
            )

        # ---- Lorentzian path only ----
        rows = params_to_rows(
            peak_labels=self.peak_labels,
            params=params,
            intensity_scale=intensity_scale,
        )

        return write_rows(
            rows,
            out_path,
            delimiter=delimiter,
            include_header=include_header,
            meta=meta,
            headers=headers,
        )

    def _build_export_metadata(self, *, include_legacy: bool = True) -> dict:
        """
        Build export metadata for fitted PL spectra.

        Parameters
        ----------
        include_legacy : bool
            If True, include legacy compatibility fields such as baseline_method,
            poly_degree, smoothing flags, etc.
            If False, prefer preprocessing-centred metadata only.

        Returns
        -------
        dict
            Cleaned metadata dictionary suitable for exporter.write_rows/write_table.
        """
        return build_single_fit_export_metadata(
            self,
            include_legacy=include_legacy,
        )

    ### NEW METHOD in v0.3.6 ###
    def export_p0(self):
        """
        Export mapping-ready initial guess vector and ordering metadata.
        """
        return export_p0_dict(self)
    
    # changed in v0.3.3
    def plot_fit(self, params, offset=0.0, scale=1.0, x_lim=(1.7, 2.2)):
        """Visualise processed spectrum, fitted total curve, and per-peak components.

        Contract:
        - Fitting is ALWAYS performed in peak-normalised space: intensity_normal = processed_spectra / peak_intensity
        - self.normalize controls DISPLAY only:
            * True  -> plot in normalised units
            * False -> plot in processed counts
        - Prints:
            * normalised residual (fit space)
            * per-peak position + (FWHM for Lorentzian, width for pVoigt) + peak height (in display units)
        """
        if params is None:
            raise ValueError("params must be provided (e.g. output from fit_spectrum).")

        # Preprocessing comparison (your existing feature)
        self._plot_preprocessing_comparison()

        p = np.asarray(params, dtype=float).ravel()
        labels = list(getattr(self, "peak_labels", []))
        if not labels:
            raise RuntimeError("No peak labels available for plotting components.")

        profile = str(getattr(self, "peak_profile", "lorentzian")).lower().strip()
        stride = int(getattr(self, "params_per_peak", 3))

        expected = stride * len(labels)
        if p.size < expected:
            raise RuntimeError(
                f"params length {p.size} is insufficient for {len(labels)} peaks with stride={stride} "
                f"(expected >= {expected})."
            )

        # ---- Display scaling: fit-space is normalised; display is either normalised or counts
        if self.normalize:
            data_plot = (self.processed_spectra / self.peak_intensity) * scale + offset
            display_multiplier = 1.0  # keep in normalised units
            y_label = "Intensity (a.u.)"
        else:
            data_plot = self.processed_spectra * scale + offset
            display_multiplier = float(self.peak_intensity)  # convert model from normalised to counts
            y_label = "Intensity (counts)"

        # ---- Total fit in fit space (normalised)
        y_fit_norm = self._model(self.energy, *p)
        y_fit_plot = (y_fit_norm * display_multiplier) * scale + offset

        # ---- Residual in fit space (normalised)
        residual = np.sum((self.intensity_normal - y_fit_norm) ** 2) / np.sum(self.intensity_normal ** 2)
        print(f'Normalized Residual: {residual:.4f} (Perfect fit has R = 0)\n')

        # ---- Plot
        plt.figure()
        plt.plot(self.energy, data_plot, "k-", label="Processed Spectrum")
        plt.plot(self.energy, y_fit_plot, "b--", label="Fitted Total Curve")

        # Print per-peak summary header
        if profile == "lorentzian":
            print(f"Per-peak (Lorentzian): position, FWHM, peak height:")
        elif profile == "pvoigt":
            print(f"Per-peak (pseudo-Voigt): position, FWHM, eta, peak height:")
        else:
            raise RuntimeError(f"Unsupported peak_profile '{profile}' in plot_fit().")

        for i, name in enumerate(labels):
            block = p[i * stride : (i + 1) * stride]

            comp_profile = "pvoigt" if profile == "pvoigt" else "lorentzian"
            y_comp_norm = single_peak(self.energy, block, profile=comp_profile)
            y_comp_plot = (y_comp_norm * display_multiplier) * scale + offset

            # Keep legacy colours for Trion/Exciton
            name_l = str(name).lower()
            if name_l == "trion":
                style = "r--"
                label = "Trion"
            elif name_l == "exciton":
                style = "g--"
                label = "Exciton"
            else:
                style = "--"
                label = str(name)

            plt.plot(self.energy, y_comp_plot, style, label=label)

            # Reporting (legacy decimals + rename Amplitude -> Peak height)
            centre = float(block[0])

            if profile == "lorentzian":
                width_hwhm = float(block[1])
                fwhm = 2.0 * width_hwhm

                amp_area = float(block[2])
                height_norm = (amp_area / (np.pi * width_hwhm)) if width_hwhm != 0 else np.nan
                peak_height = float(height_norm * display_multiplier)
                print(f'{label}: {centre:.3f} eV | FWHM: {fwhm:.4f} eV | Peak height: {peak_height:.2f}')
            else:
                fwhm = float(block[1])
                eta = float(block[3])
                peak_height = float(np.max(y_comp_norm) * display_multiplier)
                print(f'{label}: {centre:.2f} eV | FWHM: {fwhm:.2f} eV  | Peak height: {peak_height:.2f} | eta: {eta:.2f}')

        plt.xlabel("Energy (eV)")
        plt.ylabel(y_label)
        plt.xlim(list(x_lim))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # plt.tight_layout()
        plt.show()

    ## Added in build v0.2.7.1
    def _plot_preprocessing_comparison(self):
        """
        Plot raw vs preprocessing outputs on one figure when smoothing/background_remove is enabled.
        """
        do_smooth = self._smoothed_spectra is not None
        do_bg = self._baseline is not None and self._corrected_spectra is not None

        if not (do_smooth or do_bg):
            return

        plt.figure()
        plt.plot(self.energy, self.raw_spectra, label="raw")

        if do_smooth:
            plt.plot(self.energy, self._smoothed_spectra, label="smoothed")

        if do_bg:
            plt.plot(self.energy, self._baseline, label="baseline")
            plt.plot(self.energy, self._corrected_spectra, label="corrected")

        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity (counts)")
        plt.title("Preprocessing comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()
