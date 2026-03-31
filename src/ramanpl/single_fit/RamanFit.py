"""
A module for  importing Raman data from .wdf and .txt files, analyzing Raman spectra using multi-peak Lorentzian fitting with material-specific configurations.

This module provides tools for preprocessing Raman data (smoothing, background subtraction),
fitting multiple peaks using Lorentzian functions, and visualizing the results. The code works with selected materials in the raman_materials.json library.

Classes:
    RamanFit: Main class for processing, fitting, and visualizing Raman spectra.
    DataImporter: Class for importing Raman data from .wdf and .txt files (single spectrum only)
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from importlib import resources

from ramanpl.exporter import params_to_rows, write_rows, write_table
from ramanpl.preprocessing import SpectralDataset, Pipeline, build_legacy_single_spectrum_pipeline
from ramanpl.schema import normalise_baseline_spec, normalise_peak_profile
from ..peak_models import sum_peaks, single_peak
from ._single_fit_core import (
    POLY_DEGREE_SENTINEL,
    build_single_fit_export_metadata,
    export_p0_dict,
    make_compat_data_importer,
    resolve_baseline_method_with_deprecation,
    run_multistart_curve_fit,
)

DataImporter = make_compat_data_importer(axis="wavenumber", default_readlines=(300, 780))


class RamanFit:
    """
    A class for fitting and analysing Raman spectra with configurable peak models.

    Supports:
    - material/substrate library peaks or fully custom peak definitions
    - preprocessing via legacy flags or a user-supplied preprocessing Pipeline
    - Lorentzian and pseudo-Voigt peak profiles
    - peak removal, bound updates, multi-start fitting, and export helpers

    Notes
    -----
    - Fitting is always performed in peak-normalised intensity space.
    - `normalize` controls display/output scaling only.
    - When `preprocessing` is supplied, it overrides the legacy smoothing/background flags.
    """
    def __init__(
        self,
        spectra,
        wavenumber,
        materials=None,
        substrate=None,
        background_remove=False,
        baseline_method={"method": "poly", "poly_order": 3},
        poly_degree= POLY_DEGREE_SENTINEL,
        gaussian_sigma=50,
        smoothing=False,
        smooth_window=11,
        smooth_order=3,
        normalize=False,
        preprocessing=None,
        custom_peaks=None,
        remove_peaks=None,
        peak_order=None,
        peak_profile: str = "lorentzian"
    ):
        """Initialize RamanFit analyzer with data and processing parameters.

        Parameters
        ----------
        spectra : array-like
            Raw intensity values (y-axis)
        wavenumber : array-like
            Raman shift values (x-axis) in cm⁻¹
        materials : list of str, optional
            Material identifiers from library (e.g., ['WS2', 'WO3'])
        substrate : str, optional
            Substrate identifier from library (e.g., 'Si')
        baseline_method : str or dict, optional
            Baseline specification. Preferred modern style is a dict, for example:
            {"method": "poly", "poly_degree": 3}
            or
            {"method": "airpls", "lam": 1e6, "niter": 50, "tol": 1e-6}.
        normalize : bool, optional
            Controls DISPLAY/OUTPUT scaling only. Fitting is always performed in
            peak-normalised space.
        preprocessing : Pipeline or None, optional
            Optional preprocessing pipeline. If supplied, it overrides the legacy
            smoothing/background_remove settings.
        peak_profile : {'lorentzian', 'pvoigt'}, optional
            Peak line-shape model used for fitting.
            Background removal method (default: 'poly')
        poly_degree : int, optional
            Deprecated. Use baseline_method={"method": "poly", "poly_order": 3},
            instead. If supplied, a DeprecationWarning is issued and the value is folded
            into baseline_method for backwards compatibility.
        gaussian_sigma : int, optional
            Sigma for Gaussian filter (default: 50)
        smoothing : bool, optional
            Enable Savitzky-Golay smoothing (default: False)
        smooth_window : int, optional
            Window size for smoothing filter (default: 11)
        smooth_order : int, optional
            Polynomial order for smoothing (default: 3)

        Raises
        ------
        ValueError
            For unrecognized baseline methods or invalid material/substrate IDs
        """
        
        # ------------------------------
        # Peak model definition (library-driven)
        # ------------------------------
        # Start empty; populate from library to avoid duplicated / inconsistent defaults.
        self.lower_bound = []
        self.upper_bound = []
        self.peak_labels = []

        # ------------------------------
        # Peak profile (Lorentzian vs pseudo-Voigt)
        # ------------------------------
        self.peak_profile = normalise_peak_profile(peak_profile)
        self.params_per_peak = 3 if self.peak_profile == "lorentzian" else 4

        # Store user intent for reproducibility/metadata
        self.custom_peaks = custom_peaks
        self.remove_peaks_list = list(remove_peaks) if remove_peaks is not None else []

        # Store identity metadata even if custom_peaks replaces defaults
        self.materials = materials
        self.substrate = substrate

        if custom_peaks is not None:
            # "custom replaces defaults"
            if not isinstance(custom_peaks, dict) or len(custom_peaks) == 0:
                raise ValueError("custom_peaks must be a non-empty dict: {name: ([lb...],[ub...])}")

            if peak_order is None:
                peak_order_eff = list(custom_peaks.keys())
            else:
                peak_order_eff = list(peak_order)
                missing = [k for k in peak_order_eff if k not in custom_peaks]
                if missing:
                    raise ValueError(f"peak_order contains keys not in custom_peaks: {missing}")

            self.peak_labels = list(peak_order_eff)

            expected = self.params_per_peak
            for name in self.peak_labels:
                lb, ub = custom_peaks[name]
                if len(lb) != expected or len(ub) != expected:
                    if expected == 3:
                        raise ValueError(f"Peak '{name}' bounds must be length-3 lists: [centre, width(HWHM), amp(area)]")
                    else:
                        raise ValueError(f"Peak '{name}' bounds must be length-4 lists: [centre, FWHM, amp(area), eta]")
                self.lower_bound += list(lb)
                self.upper_bound += list(ub)

        else:
            # Backwards-compatible library behaviour
            if materials is None:
                materials = ["WS2"]

            self.materials = materials
            self.substrate = substrate

            if materials is not None:
                self.load_material_parameters(materials)
            if substrate is not None:
                self.load_substrate(substrate)

            # Allow deterministic ordering if requested
            self._enforce_peak_order_if_requested(peak_order=peak_order)

        # Initial guess at midpoint of bounds
        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

        # Apply removals last (always wins)
        if self.remove_peaks_list:
            self.remove_peaks(*self.remove_peaks_list)
            
        # Initialise data loaded                                                                                                                               
        self.raw_spectra = np.array(spectra)
        self.wavenumber = np.array(wavenumber)
        self.processed_spectra = np.array(spectra.copy())
        
        ## --------------------------poslished after v0.3.8---------------------------
        baseline_method = resolve_baseline_method_with_deprecation(
            baseline_method=baseline_method,
            poly_degree=poly_degree,
        )

        self._smoothed_spectra = None
        self._baseline = None
        self._corrected_spectra = None

        self.spectrum_type = "Raman"
        self.x_quantity = "Raman shift"
        self.x_unit = "cm^-1"

        self.materials = materials
        self.substrate = substrate

        self.background_remove = background_remove
        self.baseline_method = normalise_baseline_spec(
            baseline_method,
            gaussian_sigma=gaussian_sigma,
        )

        if str(self.baseline_method.get("method", "")).lower() == "poly":
            self.poly_order = int(self.baseline_method.get("poly_order", 3))
            self.poly_degree = self.poly_order  # legacy compatibility field
        else:
            self.poly_order = None
            self.poly_degree = None

        self.gaussian_sigma = gaussian_sigma
        self.smoothing = smoothing
        self.smooth_window = smooth_window
        self.smooth_order = smooth_order

        ds0 = SpectralDataset(
            x=self.wavenumber,
            y=np.asarray(self.processed_spectra, dtype=float).ravel(),
            modality="Raman",
            axis_kind="raman_shift_cm-1",
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
            )
        elif isinstance(preprocessing, Pipeline):
            pipe = preprocessing
        else:
            raise TypeError(
                "preprocessing must be None or a ramanpl.preprocessing.Pipeline instance."
            )

        self.preprocessing = pipe
        try:
            self.preprocessing_recipe = pipe.to_dict()
        except Exception:
            self.preprocessing_recipe = None

        ds = pipe.apply(ds0)

        self.wavenumber = np.asarray(ds.x, dtype=float).ravel()
        self.processed_spectra = np.asarray(ds.y, dtype=float).ravel()

        crop_mask = ds.meta.get("crop_mask", None)
        if crop_mask is not None:
            self.raw_spectra = np.asarray(self.raw_spectra, dtype=float).ravel()[crop_mask]
        else:
            self.raw_spectra = np.asarray(self.raw_spectra, dtype=float).ravel()

        self._smoothed_spectra = ds.meta.get("_smoothed_last", None)
        self._baseline = ds.meta.get("_baseline_last", None)

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


    def _get_material_lib_path(self):
        """
        Return a Traversable handle to the package-level Raman material library.

        The JSON file is kept at:
            ramanpl/raman_materials.json

        This is robust even when RamanFit.py lives inside ramanpl/single_fit/.
        """
        return resources.files("ramanpl").joinpath("raman_materials.json")

    def _load_material_library(self):
        """
        Load the package-level Raman material library JSON.

        Returns
        -------
        dict
            Parsed material library.
        """
        json_ref = self._get_material_lib_path()

        try:
            with json_ref.open("r", encoding="utf-8") as f:
                material_lib = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Material library file not found at package resource: {json_ref}")
        except Exception as e:
            raise ValueError(f"Failed to read Raman material library: {e}") from e

        if not isinstance(material_lib, dict):
            raise ValueError("Raman material library JSON must decode to a dict.")

        return material_lib
    
    def load_material_parameters(self, materials):
        """Load peak parameters from JSON material library.

        Parameters
        ----------
        materials : list of str
            Material identifiers from library (e.g., ['WS2', 'MoS2'])

        Raises
        ------
        ValueError
            If material library file is missing or contains invalid data
        """
        material_lib = self._load_material_library()                
        for material in materials:
            if material not in material_lib:
                raise ValueError(f"Material '{material}' not found in library")
            
            ### Updated in v.0.2.4 ###
            params = material_lib[material]['peaks']
            lb = params['lower_bound']
            ub = params['upper_bound']
            labels = params['peak_labels']

            # Updated in v0.3.3
            # Append peaks but avoid duplicated peak labels (prevents WS2 double-loading etc.)
            for k, name in enumerate(labels):
                if name in self.peak_labels:
                    continue

                # library is Lorentzian-style: [centre, HWHM, area_amp]
                lb3 = lb[3*k:3*k+3]
                ub3 = ub[3*k:3*k+3]

                self.peak_labels.append(name)

                if self.peak_profile == "lorentzian":
                    self.lower_bound.extend(lb3)
                    self.upper_bound.extend(ub3)
                else:
                    # pVoigt expects width = FWHM. Convert Lorentzian HWHM -> FWHM by *2.
                    c_lb, hwhm_lb, a_lb = lb3
                    c_ub, hwhm_ub, a_ub = ub3

                    self.lower_bound.extend([c_lb, 2.0*hwhm_lb, a_lb, 0.01])
                    self.upper_bound.extend([c_ub, 2.0*hwhm_ub, a_ub, 0.99])

        # Verify parameter consistency
        stride = self.params_per_peak
        if (len(self.upper_bound) !=  stride * len(self.peak_labels)):
            raise ValueError("Invalid parameter dimensions after loading peaks (stride mismatch).")

    ### updated in v0.3.3 ###
    def _enforce_peak_order_if_requested(self, peak_order=None):
        """
        Reorder bounds/labels to match a user-provided peak_order (case-insensitive).
        If peak_order is None, do nothing.
        """
        if peak_order is None:
            return

        peak_order = list(peak_order)
        labels_lower = [p.lower() for p in self.peak_labels]
        order_lower = [p.lower() for p in peak_order]

        if sorted(order_lower) != sorted(labels_lower):
            raise ValueError(
                "peak_order must be a permutation of the loaded peak labels.\n"
                f"Loaded: {self.peak_labels}\n"
                f"Requested: {peak_order}"
            )

        stride = self.params_per_peak

        def block(arr, idx):
            return arr[stride*idx:stride*idx+stride]

        new_lb, new_ub, new_labels = [], [], []
        for name_lower in order_lower:
            old_idx = labels_lower.index(name_lower)
            new_labels.append(self.peak_labels[old_idx])
            new_lb += block(self.lower_bound, old_idx)
            new_ub += block(self.upper_bound, old_idx)

        self.peak_labels = new_labels
        self.lower_bound = new_lb
        self.upper_bound = new_ub

    def _build_export_metadata(self, *, include_legacy: bool = True) -> dict:
        """
        Build export metadata for fitted Raman spectra.

        Parameters
        ----------
        include_legacy : bool
            If True, include legacy compatibility fields such as baseline_method,
            poly_degree, smoothing flags, etc.
            If False, prefer the new preprocessing-centred metadata only.

        Returns
        -------
        dict
            Cleaned metadata dictionary suitable for exporter.write_rows/write_table.
        """
        return build_single_fit_export_metadata(
            self,
            include_legacy=include_legacy,
            extra_meta={
                "materials": getattr(self, "materials", None),
                "substrate": getattr(self, "substrate", None),
            },
        )
    
    def export_p0(self):
        """
        Export mapping-ready initial guess (normalised fit space) + ordering metadata.
        """
        return export_p0_dict(self)

    ### Added in v0.2.9 ###
    def get_fitted_spectrum(self):
        """
        Return fitted spectrum on the same x-grid as the input data.

        Returns
        -------
        x : np.ndarray
            Wavenumber axis (cm^-1)
        y_fit : np.ndarray
            Fitted intensity in the SAME units as self.processed_spectra.
        """
        if not hasattr(self, "params_fit") or self.params_fit is None:
            raise RuntimeError("RamanFit has not been fitted yet. Run fit_spectrum() first.")

        x = np.asarray(self.wavenumber, dtype=float).ravel()

        # Fit is performed in normalised space; convert back to processed intensity scale
        y_fit_norm = self._model(x, *self.params_fit)  # model in normalised space
        y_fit = y_fit_norm * float(self.peak_intensity)

        return x.copy(), np.asarray(y_fit, dtype=float).ravel().copy()

    ### Added in v0.2.9 ###
    def get_fitted_parameters(self):
        """
        Return fitted peak parameters as a structured dict.

        Notes
        -----
        Parameter vector layout depends on peak_profile:

        - lorentzian: (loc, HWHM, amp_area) per peak
        - pvoigt:     (loc, FWHM, amp_area, eta) per peak

        Reported peak_height/intensity is the *peak maximum* in processed intensity units
        (i.e. scaled back by peak_intensity when self.normalize=False elsewhere).
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

        # Convert fit-space (normalised) peak height back to processed units.
        # Note: fitting is always normalised, so this is the factor you use everywhere.
        fit_scale = float(self.peak_intensity) if hasattr(self, "peak_intensity") else 1.0

        # For pVoigt peak height, evaluate numerically using the same component model as plot_fit()
        if profile == "pvoigt":
            pass

        out = {}

        for i, name in enumerate(self.peak_labels):
            block = p[stride * i : stride * (i + 1)]

            loc = float(block[0])
            width = float(block[1])          # HWHM (lorentzian) or FWHM (pvoigt)
            amp_area = float(block[2])       # area-like amplitude (both profiles)

            if profile == "lorentzian":
                # Historical convention: width is HWHM, so FWHM = 2*HWHM
                fwhm = 2.0 * width
                height_norm = (amp_area / (np.pi * width)) if width != 0 else np.nan

            else:
                # pVoigt convention in your peak_models: width is already FWHM
                fwhm = width
                eta = float(block[3])
                y_comp_norm = single_peak(self.wavenumber, block, profile="pvoigt")
                height_norm = float(np.max(y_comp_norm))

            height_scaled = float(height_norm * fit_scale)

            row = dict(
                position=loc,
                fwhm=float(fwhm),
                amp=float(amp_area),
                height_norm=float(height_norm),
                peak_height=float(height_scaled),   # preferred name
                intensity=float(height_scaled),     # backwards-compatible alias
            )

            # Keep "scale" for Lorentzian compatibility; for pVoigt, store "eta" and optionally "fwhm_param"
            if profile == "lorentzian":
                row["scale"] = float(width)  # HWHM
            else:
                row["eta"] = float(eta)
                row["fwhm_param"] = float(width)  # explicit: pVoigt width parameter is FWHM

            out[name] = row

        return out


    ### Updated in v0.3.3 to handle both profiles and include metadata
    def fit_table(self, params=None, *, scaled: bool = True):
        """
        Return per-peak fitted parameters as a list of dicts.

        Notes
        -----
        - Lorentzian rows include: Peak, Position(cm^-1), FWHM(cm^-1), Scale, Amp,
        Height_norm, Height_scaled
        - pseudo-Voigt rows include: Peak, Position(cm^-1), FWHM(cm^-1), Eta, Amp,
        Height_norm, Height_scaled
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
                    "Position(cm^-1)": d["position"],
                    "FWHM(cm^-1)": d["fwhm"],
                    "Eta": d.get("eta", ""),
                    "Amp": d["amp"],
                    "Height_norm": d["height_norm"],
                    "Height_scaled": d["peak_height"],
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
                "Position(cm^-1)": r.centre,
                "FWHM(cm^-1)": r.fwhm,
                "Scale": r.scale,
                "Amp": r.amp,
                "Height_norm": r.peak_height_norm,
                "Height_scaled": r.peak_height,
            }
            for r in rows
        ]

    ### Updated in v0.3.3 to handle both profiles and include metadata
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
        - Lorentzian exports use the standard exporter row format.
        - pseudo-Voigt exports use a dict-table export with Eta included.

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

        # DEPRECATED legacy metadata fields are still included for backwards compatibility, but the preferred export metadata is now built from the actual preprocessing pipeline and key attributes. The legacy fields are still populated on the RamanFit instance for user access, but they are no longer canonical or guaranteed to be consistent with the actual preprocessing steps when a Pipeline is used.
        # meta = {
        #     "spectrum_type": getattr(self, "spectrum_type", None),
        #     "x_quantity": getattr(self, "x_quantity", None),
        #     "x_unit": getattr(self, "x_unit", None),

        #     "materials": getattr(self, "materials", None),
        #     "substrate": getattr(self, "substrate", None),

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

            from ramanpl.exporter import write_table
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

    # Updated v0.3.6
    def load_substrate(self, substrate):
        """Load substrate parameters from JSON material library.

        Parameters
        ----------
        substrate : str
            Substrate identifier from library (e.g., 'Si', 'SiO2')

        Raises
        ------
        ValueError
            If substrate not found or not marked as substrate in library
        """
        material_lib = self._load_material_library()

        if substrate not in material_lib or not material_lib[substrate].get('substrate', False):
            raise ValueError(f"Invalid substrate '{substrate}' or not marked as substrate in library")
        
        params = material_lib[substrate]['peaks']
        lb = params["lower_bound"]
        ub = params["upper_bound"]
        labels = params["peak_labels"]

        for k, name in enumerate(labels):
            if name in self.peak_labels:
                continue

            lb3 = lb[3*k:3*k+3]
            ub3 = ub[3*k:3*k+3]

            self.peak_labels.append(name)

            if self.peak_profile == "lorentzian":
                self.lower_bound.extend(lb3)
                self.upper_bound.extend(ub3)
            else:
                c_lb, hwhm_lb, a_lb = lb3
                c_ub, hwhm_ub, a_ub = ub3
                self.lower_bound.extend([c_lb, 2.0*hwhm_lb, a_lb, 0.01])
                self.upper_bound.extend([c_ub, 2.0*hwhm_ub, a_ub, 0.99])
    
    def update_bounds(self, **kwargs):
        """
        Update fitting constraints for specific peaks.

        kwargs format:
            peak=([lb...], [ub...])

        For lorentzian: [centre, HWHM, amp(area)]
        For pvoigt:     [centre, FWHM, amp(area), eta]
        """
        stride = self.params_per_peak
        labels_lower = [p.lower() for p in self.peak_labels]

        for peak_name, new_bounds in kwargs.items():
            key = str(peak_name).lower()
            if key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            if not (isinstance(new_bounds, (tuple, list)) and len(new_bounds) == 2):
                raise ValueError(f"Bounds for '{peak_name}' must be (lb, ub).")

            lb_new, ub_new = new_bounds
            if len(lb_new) != stride or len(ub_new) != stride:
                raise ValueError(f"Bounds for '{peak_name}' must be length-{stride} lists (stride={stride}).")

            idx = labels_lower.index(key)
            s = stride * idx

            self.lower_bound[s:s+stride] = list(lb_new)
            self.upper_bound[s:s+stride] = list(ub_new)
            self.p0[s:s+stride] = [(lb_new[i] + ub_new[i]) / 2 for i in range(stride)]
    
    ## updated in v0.3.3
    def remove_peaks(self, *peak_names):
        """Remove peaks from fitting model.

        Parameters
        ----------
        *peak_names : str
            Names of peaks to remove (e.g., 'E12g(Γ)', 'A1g(Γ)')

        Raises
        ------
        ValueError
            If specified peak names are not in current model
        """
        stride = self.params_per_peak
        for peak_name in peak_names:
            labels_lower = [p.lower() for p in self.peak_labels]
            key = str(peak_name).lower()
            if key not in labels_lower:
                raise ValueError(f"Peak '{peak_name}' is not recognised. Available peaks: {self.peak_labels}")

            idx = labels_lower.index(key)
            del self.p0[stride*idx:stride*idx+stride]
            del self.lower_bound[stride*idx:stride*idx+stride]
            del self.upper_bound[stride*idx:stride*idx+stride]
            del self.peak_labels[idx]

    def _model(self, x, *params):
        """Internal model dispatcher (Lorentzian or pseudo-Voigt)."""
        profile = "pvoigt" if self.peak_profile == "pvoigt" else "lorentzian"
        return sum_peaks(
            np.asarray(x),
            params,
            profile=profile,
            stride=self.params_per_peak,
        )

    # Updated in v0.3.6
    def fit_spectrum(
        self,
        *,
        n_starts: int = 1,
        p0_strategy: str = "midpoint",
        random_state=None,
        diagnose_bounds: bool = True,
        bounds_tol: float = 1e-6,
        return_diagnostics: bool = False,
    ):
        """
        Perform bounded least-squares fitting with optional multi-start.
        """
        best_params, best_cov, diagnostics = run_multistart_curve_fit(
            model=self._model,
            x=self.wavenumber,
            y=self.intensity_normal,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            base_p0=self.p0,
            n_starts=n_starts,
            p0_strategy=p0_strategy,
            random_state=random_state,
            maxfev=6400,
            diagnose_bounds=diagnose_bounds,
            bounds_tol=bounds_tol,
            fail_label="RamanFit.fit_spectrum",
        )

        self.params_fit = best_params
        self.params_cov = best_cov
        self.fit_diagnostics = diagnostics

        if return_diagnostics:
            return best_params, best_cov, diagnostics
        return best_params, best_cov

    # Method to plot the fitted spectrum along with components
    def plot_fit(self, params, offset=0, scale=1.0, x_lim = [250, 750], y_lim = [],
                 x_ticks = [300, 350, 400, 450, 500, 550, 600, 650, 700]):
        """Visualise fitting results and components (Lorentzian + pseudo-Voigt).

        Notes
        -----
        - Fit space is always normalised: intensity_normal = processed_spectra / peak_intensity
        - self.normalize controls DISPLAY only:
            * True  -> plot in normalised a.u.
            * False -> plot in counts
        - Reported "Peak height" is the peak maximum in DISPLAY units.

        Parameters
        ----------
        params : array-like
            Fitting parameters from fit_spectrum()
        offset : float, optional
            Vertical offset for plotting multiple spectra (default: 0)
        scale : float, optional
            Vertical scaling factor (default: 1.0)
        x_lim : list, optional
            X-axis range [min, max] in cm⁻¹ (default: [250, 750])
        y_lim : list, optional
            Y-axis range [min, max] (default: auto-scale)
        x_ticks : list, optional
            X-axis tick positions in cm⁻¹ (default: 300-700 in 50 cm⁻¹ steps)

        Displays
        --------
        - Raw and processed spectra
        - Fitted curve and individual components
        - Quality metrics in console output
        """
        ## Added in v0.2.7.1 for creating processing comparison
        self._plot_preprocessing_comparison()

        plt.figure()

        profile = str(getattr(self, "peak_profile", "lorentzian")).lower().strip()
        stride = int(getattr(self, "params_per_peak", 3))
        p = np.asarray(params, dtype=float).ravel()

        # Display multiplier: convert model (normalised fit space) to display units
        display_multiplier = 1.0 if self.normalize else float(self.peak_intensity)

        # Display-space spectra
        if self.normalize:
            proc_plot = (self.processed_spectra / self.peak_intensity) * scale + offset
            raw_plot = (self.raw_spectra / self.peak_intensity) * scale + offset
            plt.yticks([])
        else:
            proc_plot = self.processed_spectra * scale + offset
            raw_plot = self.raw_spectra * scale + offset

        # Plot spectra
        plt.plot(self.wavenumber, proc_plot, 'k-', label='Processed Spectrum')
        plt.plot(self.wavenumber, raw_plot, 'g-', label='Original Spectrum')

        # Total fitted curve (fit space -> display space)
        y_fit_norm = self._model(self.wavenumber, *p)
        y_fit_plot = (y_fit_norm * display_multiplier) * scale + offset
        plt.plot(self.wavenumber, y_fit_plot, 'b--', label='Fitted Total Curve')

        # Residual in fit space (normalised)
        residual = np.sum((self.intensity_normal - y_fit_norm) ** 2) / np.sum(self.intensity_normal ** 2)

        # Print header (style-preserving)
        if profile == "lorentzian":
            print("\n{:<20} {:<15} {:<13} {:<14} {:<10}".format(
                "Peak", "Position(cm⁻¹)", "FWHM(cm⁻¹)", "Peak height", "Scale"
            ))
        elif profile == "pvoigt":
            print("\n{:<20} {:<15} {:<13} {:<14} {:<10} {:<6}".format(
                "Peak", "Position(cm⁻¹)", "FWHM(cm⁻¹)", "Peak height", "Scale", "eta"
            ))
        else:
            raise RuntimeError(f"Unsupported peak_profile '{profile}' in plot_fit().")

        print("-" * 80)

        # Plot components and calculate parameters
        peak_positions = {}

        for i, name in enumerate(self.peak_labels):
            block = p[i * stride:(i + 1) * stride]
            loc = float(block[0])

            # Store positions for special peaks
            peak_positions[str(name)] = loc

            # Component in fit space, then display space
            comp_profile = "pvoigt" if profile == "pvoigt" else "lorentzian"
            y_comp_norm = single_peak(self.wavenumber, block, profile=comp_profile)
            y_comp_plot = (y_comp_norm * display_multiplier) * scale + offset

            # Plot component (keep your legacy red dashed style)
            plt.plot(self.wavenumber, y_comp_plot, 'r--')

            # Peak height (display units)
            peak_height = float(np.max(y_comp_norm) * display_multiplier)

            if profile == "lorentzian":
                # width stored as HWHM in your Lorentzian convention
                scale_param = float(block[1])
                fwhm = 2.0 * scale_param
                amp_area = float(block[2])

                print("{:<20} {:<15.2f} {:<13.2f} {:<14.2f} {:<10.2f}".format(
                    str(name), loc, fwhm, peak_height, scale_param
                ))

            else:
                # pVoigt width stored as FWHM (per your updated peak_models)
                fwhm = float(block[1])
                amp_area = float(block[2])
                eta = float(block[3])

                # "Scale" column: keep something meaningful and stable for users.
                # For pVoigt we print FWHM again as "Scale" to avoid inventing a new parameter name.
                # (If you prefer, relabel the column to "Width" for pVoigt later.)
                print("{:<20} {:<15.2f} {:<13.2f} {:<14.2f} {:<10.2f} {:<6.2f}".format(
                    str(name), loc, fwhm, peak_height, fwhm, eta
                ))

        # E12g-A1g separation
        if 'E12g' in peak_positions and 'A1g' in peak_positions:
            peak_diff = peak_positions['A1g'] - peak_positions['E12g']
            print(f"\nE12g(Γ)-A1g(Γ) separation: {peak_diff:.2f} cm⁻¹")

        # Residual print (match your legacy wording)
        print(f"\nNormalized Residual: {residual:.4f} (0 = perfect fit)")

        # Plot formatting
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)' if self.normalize else 'Intensity (counts)')
        plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)
        plt.xticks(x_ticks)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

    # Added in v0.2.7.1
    def _plot_preprocessing_comparison(self):
        """
        Plot raw vs preprocessing outputs on one figure when smoothing/background_remove is enabled.
        """
        do_smooth = self._smoothed_spectra is not None
        do_bg = self._baseline is not None and self._corrected_spectra is not None

        if not (do_smooth or do_bg):
            return  # nothing to compare

        plt.figure()

        # Always show raw
        plt.plot(self.wavenumber, self.raw_spectra, label="raw")

        # Smoothed (only if smoothing=True)
        if do_smooth:
            plt.plot(self.wavenumber, self._smoothed_spectra, label="smoothed")

        # Baseline + corrected (only if background_remove=True)
        if do_bg:
            plt.plot(self.wavenumber, self._baseline, label="baseline")
            plt.plot(self.wavenumber, self._corrected_spectra, label="corrected")

        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity (counts)")
        plt.title("Preprocessing comparison")
        plt.legend()
        plt.tight_layout()
        plt.show()
