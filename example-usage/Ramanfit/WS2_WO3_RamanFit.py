import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class WS2_WO3_RamanFit:
    """
    A class for fitting and analyzing Raman spectra using a multi-peak 
    Lorentzian function to model the Raman shifts.
    
    Attributes:
    ----------
    spectra : array-like
        The Raman intensity data for the spectrum.
    wavenumber : array-like
        The wavenumber axis for the Raman spectrum.
    peak_intensity : float
        Maximum intensity value of the spectrum, used for normalization.
    spectra_normal : array-like
        The normalized intensity data.
    lower_bound : list
        The default lower bounds for the fitting parameters.
    upper_bound : list
        The default upper bounds for the fitting parameters.

    Methods:
    -------
    remove_peaks(*peak_names):
        Remove specific peaks by their names.
    lorentzian_raman(x, *params):
        Lorentzian distribution to fit the Raman peaks.
    update_bounds(**kwargs):
        Update the fitting bounds for specific Raman peaks.
    fit_spectrum():
        Fit the spectrum using the Lorentzian model.
    plot_fit(params, offset=0, scale=1.0):
        Plot the original spectrum, fitted total curve, and component fits.
    """


    def __init__(self, spectra, wavenumber):
        self.spectra = spectra
        self.wavenumber = wavenumber
        self.peak_intensity = max(self.spectra)
        self.spectra_normal = self.spectra / self.peak_intensity  # Normalize the spectrum

        # Set lower and upper bounds for the fit
        self.lower_bound = [
            170, 0, 0,   # LA(M)
            250, 0, 0,   # W=O bending
            353, 0, 0,   # E12g(Γ)
            418, 0, 0,   # A1g(Γ)
            348, 0, 0,   # 2LA(M)
            225, 0, 0,   # A1g(M)-LA(M)
            295, 0, 0,   # 2LA(M)-2E22g(Γ)
            318, 0, 0,   # 2LA(M)-E22g(Γ)

            583, 0, 0,   # A1g(M)+LA(M)
            697, 0, 0,   # 4LA(M)
            800, 0, 0,   # W=O stretching

            519, 0, 0,   # Si
        ]

        self.upper_bound = [
            180, 10, 10,  # LA(M)
            270, 10, 10,  # W=O bending
            358, 5, 10,   # E12g(Γ)
            424, 5, 10,   # A1g(Γ)
            353, 5, 10,   # 2LA(M)
            235, 10, 10,  # A1g(M)-LA(M)
            305, 10, 2,   # 2LA(M)-2E22g(Γ)
            328, 10, 10,  # 2LA(M)-E22g(Γ)

            587, 10, 10,  # A1g(M)+LA(M)
            710, 10, 10,  # 4LA(M)
            810, 5, 10,   # W=O stretching

            522, 10, 10,  # Si
        ]

        # Define peak labels for easy referencing when updating bounds
        self.peak_labels = ['LA(M)', 'W=O bending','E12g(Γ)', 'A1g(Γ)', '2LA(M)', 'A1g(M)-LA(M)', 
                            '2LA(M)-2E22g(Γ)', '2LA(M)-E22g(Γ)', 'A1g(M)+LA(M)', 
                            '4LA(M)','W=O stretching','Si']

        # Ensure p0 is set to the midpoint of the lower and upper bounds
        self.p0 = [(low + high) / 2 for low, high in zip(self.lower_bound, self.upper_bound)]

    
    # Method to update the lower and upper bounds for a specific peak
    def update_bounds(self, **kwargs):
        """
        Update the fitting bounds for specific Raman peaks and adjust p0 accordingly.
        
        Parameters:
        ----------
        kwargs : dict
            Keyword arguments for updating specific peaks. For example:
            update_bounds(Si=([518, 0, 0], [521, 2, 7]))
        """
        for peak_name, new_bounds in kwargs.items():
            if peak_name not in self.peak_labels:
                raise ValueError(f"Peak '{peak_name}' is not a recognized peak name. Available peaks are: {self.peak_labels}")

            if not (isinstance(new_bounds, tuple) and len(new_bounds) == 2 and 
                    isinstance(new_bounds[0], list) and isinstance(new_bounds[1], list) and 
                    len(new_bounds[0]) == 3 and len(new_bounds[1]) == 3):
                raise ValueError(f"Bounds for '{peak_name}' must be a tuple of two lists with three elements each.")

            idx = self.peak_labels.index(peak_name)

            # Update the lower and upper bounds for the specified peak
            self.lower_bound[3 * idx:3 * idx + 3] = new_bounds[0]
            self.upper_bound[3 * idx:3 * idx + 3] = new_bounds[1]

            # Update p0 to the midpoint of the new bounds
            self.p0[3 * idx:3 * idx + 3] = [(new_bounds[0][i] + new_bounds[1][i]) / 2 for i in range(3)]

    def remove_peaks(self, *peak_names):
        """
        Remove specific peaks from the fitting and plotting process.

        Parameters:
        ----------
        peak_names : str
            Names of the peaks to remove. For example:
            remove_peaks('E12g', 'A1g')
        """
        for peak_name in peak_names:
            if peak_name not in self.peak_labels:
                raise ValueError(f"Peak '{peak_name}' is not a recognized peak name. Available peaks are: {self.peak_labels}")
            
            idx = self.peak_labels.index(peak_name)

            # Remove the corresponding parameters, bounds, and label
            del self.p0[3 * idx:3 * idx + 3]
            del self.lower_bound[3 * idx:3 * idx + 3]
            del self.upper_bound[3 * idx:3 * idx + 3]
            del self.peak_labels[idx]

    # Lorentzian function to fit each peak
    @staticmethod
    def lorentzian_raman(x, *params):
        """
        Lorentzian distribution to fit the Raman peaks.
        
        Parameters:
        ----------
        x : array-like
            The wavenumber values for the spectrum.
        params : array-like
            The parameters for each Lorentzian peak (loc, scale, amp).

        Returns:
        -------
        L : array-like
            Sum of Lorentzian distributions for all peaks.
        """
        L = 0
        for i in range(0, len(params), 3):
            loc, scale, amp = params[i:i+3]
            L += (scale / ((x - loc) ** 2 + scale ** 2)) * amp / np.pi
        return L

    # Method to fit the spectrum
    def fit_spectrum(self):
        """
        Fit the Raman spectrum using the Lorentzian function.
        
        Returns:
        -------
        params : array-like
            The optimized fitting parameters for the Raman peaks.
        params_cov : array-like
            Covariance matrix of the fitting parameters.
        """
        params, params_cov = optimize.curve_fit(
            self.lorentzian_raman, 
            self.wavenumber, 
            self.spectra_normal, 
            p0=self.p0, 
            bounds=(self.lower_bound, self.upper_bound), 
            maxfev=6400
        )
        return params, params_cov

    # Method to plot the fitted spectrum along with components
    def plot_fit(self, params, offset=0, scale=1.0, x_lim = [250, 750], y_lim = [],
                 x_ticks = [300, 350, 400, 450, 500, 550, 600, 650, 700]):
        """
        Plot the fitted spectrum along with the individual Lorentzian components for Raman peaks.
        
        Parameters:
        ----------
        params : array-like
            Fitting parameters returned by fit_spectrum.
        offset : float, optional
            Offset to shift the plot vertically (default is 0).
        scale : float, optional
            Scale factor to adjust the plot (default is 1.0).
        x_lim : list, optional
            Plot x scale limit, which is 250 - 750 cm^-1 by default
        y_lim : list, optional
            Plot y scale limit, which is void by default
        x_ticks : list, optional
            X-axis lables, which are 300, 350, 400, 450, 500, 550, 600, 650, 700 cm^-1 by default
        """
        plt.figure()
        
        # Plot original spectrum with scaling and offset
        plt.plot(self.wavenumber, self.spectra * scale + offset, 'k-', label='Original Spectrum')

        # Plot the total fitted curve
        y_fit = self.lorentzian_raman(self.wavenumber, *params)
        plt.plot(self.wavenumber, y_fit * self.peak_intensity, 'b--', label='Fitted Total Curve')

        # Plot each Lorentzian component
        for i in range(round(len(params) / 3)):
            loc = params[3 * i]
            scale = params[3 * i + 1]
            amp = params[3 * i + 2]
            y_fit_single = (scale / ((self.wavenumber - loc) ** 2 + scale ** 2)) * amp / np.pi
            plt.plot(self.wavenumber, y_fit_single * self.peak_intensity, 'r--')

            print(f'Peak {self.peak_labels[i]}: loc={loc:.2f}, scale={scale:.2f}, amp={amp:.2f}')
        
        #Print peak value differences
        Peak_diff = params[9] - params[6]
        print(f'\n  E12g and A1g has peak difference = {Peak_diff:.2f} cm^-1')

        # Add labels and formatting
        plt.xlabel('Raman Shift (1/cm)')
        plt.ylabel('Intensity (a.u.)')
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.xticks(x_ticks)
        plt.yticks([])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.show()

