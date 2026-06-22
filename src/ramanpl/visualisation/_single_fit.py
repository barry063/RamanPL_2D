"""Internal single-spectrum plotting helpers for RamanPL v0.6.8."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ramanpl.peak_models import single_peak


def plot_raman_fit(self, params, offset=0, scale=1.0, x_lim = [250, 750], y_lim = [],
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


def plot_pl_fit(self, params, offset=0.0, scale=1.0, x_lim=(1.7, 2.2)):
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

