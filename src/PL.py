import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import find_peaks
from scipy.special import wofz


class PL:
    def __init__(self, filename, refname):
        """
        Initialize a Raman instance.

        Parameters:
        - filename (str): The filename for the data.
        - refname (str): The filename for the reference data.

        Initializes "peak_values","data", "ref_data" and "fitted_data" as None.
        """
        self.filename = filename
        self.refname = refname
        self.peak_values = None     # Initialize peak_values to None
        self.data = None            # Initialize data to None
        self.ref_data = None        # Initialize ref_data to None
        self.fitted_data = None     # Initialize fitted_data to None
    
    def load_data(self, file=None):
        """
        Load data from a text file into the data attribute.

        Parameters:
        - file (str, optional): The filename to load. If None, uses the filename provided during initialization.

        Raises:
        - FileNotFoundError: If the file is not found.
        - Exception: If an error occurs during loading.
        """
        if file is None:
            if self.filename is None:
                raise ValueError("No filename provided.")
            file = self.filename

        if not file.endswith('.txt'):
            raise ValueError(
                "Wrong type of data is read, it must be .txt format")

        try:
            # Read data from the text file and convert to pandas DataFrame
            self.data = pd.read_table(file,
                                      delimiter="\t",
                                      skiprows=[0],
                                      names=['Energy', 'Intensity'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file '{file}' was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {str(e)}")

    def load_reference(self, file = None):
        """
        Load reference data from a text file into the ref_data attribute.

        Parameters:
        - file (str, optional): The refname to load. If None, uses the refname provided during initialization.

        Raises:
        - FileNotFoundError: If the file is not found.
        - Exception: If an error occurs during loading.
        """        
        if file is None:
            if self.refname is None:
                raise ValueError("No refname provided.")
            file = self.refname

        if not file.endswith('.txt'):
            raise ValueError(
                "Wrong type of data is read, it must be .txt format")

        try:
            # Read data from the text file and convert to pandas DataFrame
            self.ref_data = pd.read_table(file,
                                          delimiter="\t",
                                          skiprows=[0],
                                          names=['Energy', 'Intensity'])
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The file '{file}' was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {str(e)}")

    def net_data(self, baseline=0):
        """
        Calculate net intensity values between data and reference data.

        Parameters:
        - baseline (int, optional): sets any negative values to baseline value (defalut = 0).
        """
        if self.data is None or self.ref_data is None:
            raise ValueError("Both data and reference data must be loaded.")
        # Calculate net values between data and reference data in each row
        self.data['Net_Intensity'] = self.data['Intensity'] - \
            self.ref_data['Intensity']
        # Set any negative net values to baseline value
        self.data['Net_Intensity'] = self.data['Net_Intensity'].apply(
            lambda x: max(baseline, x))

    def smooth_data(self, window_size=5, data='Intensity', base = 0):
        """
        Smooth PL data using a moving average and linearly interpolate NaN values.

        Parameters:
        - window_size (int, optional): The size of the moving average window.
        - data (str, optional): The data to smooth ('Intensity' or 'Net_Intensity').
        - base (int, optional): The base line of the background data, 0 by default.

        Raises:
        - ValueError: If data is not loaded or if data is invalid.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before smoothing.")

        valid_data_options = ['Intensity', 'Net_Intensity']

        if data not in valid_data_options:
            raise ValueError("Invalid data option. Use one of the following: "
                             f"{', '.join(valid_data_options)}.")

        y = self.data[data]

        # Apply moving average smoothing
        smoothed_data = y.rolling(window=window_size, center=True).mean()

        # Linearly interpolate NaN entries
        smoothed_data.interpolate(method='linear', inplace=True)
        smoothed_data.fillna(base, inplace=True)

        # Replace the data column with the smoothed and interpolated data
        self.data[data] = smoothed_data

    def normalise(self, data="Intensity", max=None):
        """
        Normalize the PL spectrum data.

        Parameters:
        - data (str, optional): The data column to normalize ('Intensity' or 'Net_Intensity').
        - max (float, optional): The maximum value for normalization. If None, the maximum value in the data is used.

        Raises:
        - ValueError: If data is not loaded or if an invalid data option is provided.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before normalizing.")

        valid_data_options = ['Intensity', 'Net_Intensity']

        if data not in valid_data_options:
            raise ValueError("Invalid data option. Use one of the following: "
                             f"{', '.join(valid_data_options)}.")

        if max is None:
            y = self.data[data]
            max_value = y.max()
        else:
            max_value = max

        # Normalize the data and store it as a new column in self.data
        self.data[f'Normalised_{data}'] = self.data[data] / max_value

    def curve_fit(self, fitting_type='Lorenztian', 
                  data='Intensity', initial_guesses=None, 
                  poly_degree=None, num_peaks=1, plot=True):
        """
        Perform curve fitting on PL spectrum data with multiple peaks.

        Parameters:
        - fitting_type (str, optional): The type of fitting to perform ('Lorenztian', 'Gauss', 'Voigt', 'Polynomial').
        - data (str, optional): The data to fit ('Intensity' or 'Net_Intensity').
        - initial_guesses (list of lists or None, optional): Lists of initial guesses for peak parameters.
          Each inner list should contain guesses for a single peak in the following order:
          [Amplitude, Position, Width, Background].
        - num_peaks (int, optional): The number of peaks to fit.
        - poly_degree (int, optional): The degree of the polynomial to fit.
        - plot (bol, True): plot the curve-fitting, and False to turn it off

        Raises:
        - ValueError: If data is not loaded or if fitting_type or data is invalid.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before curve fitting.")

        valid_fitting_types = ['Lorenztian', 'Gauss', 'Voigt', 'Polynomial']
        valid_data_options = ['Intensity', 'Net_Intensity', 'Normalised_Intensity', 'Normalised_Net_Intensity']

        if fitting_type not in valid_fitting_types:
            raise ValueError("Invalid fitting_type. Use one of the following: "
                             f"{', '.join(valid_fitting_types)}.")

        if data not in valid_data_options:
            raise ValueError("Invalid data option. Use one of the following: "
                             f"{', '.join(valid_data_options)}.")

        x = self.data['Energy']
        y = self.data[data]

        if fitting_type == 'Polynomial' and poly_degree is not None:
            # Fit a polynomial to the data
            params = np.polyfit(x, y, poly_degree)
            fitted_data = np.polyval(params, x)
        else:
        # Define the composite function to fit multiple peaks
            def composite(x, *params):
                result = np.zeros_like(x)
                for i in range(0, len(params), 4):
                    A, x0, gamma, y0 = params[i:i+4]
                    if fitting_type == 'Lorenztian':
                        result += A * (gamma**2) / ((x - x0)**2 + (gamma**2)) + y0
                    elif fitting_type == 'Gauss':
                        result += A * np.exp(-(x - x0)**2 / (2 * gamma**2)) + y0
                    elif fitting_type == 'Voigt':
                        sigma = abs(gamma)
                        result += A * np.real(wofz(((x - x0) + 1j * sigma) / (sigma * np.sqrt(2)))) + y0
                return result

            if initial_guesses is None:
                # Provide hints for initial guesses
                print("Hint for initial guesses: Use the following order for each peak:")
                print("[Amplitude, Position, Width, Background]")
                print("Amplitude: Height or intensity of the peak (positive value).")
                print("Position: Center position of the peak on the x-axis (within the data range).")
                print("Width: Width of the peak (positive value).")
                print("Background: Baseline intensity (can be positive or negative).")
                
                # Initialize initial guesses for all peaks
                initial_guess = []
                for _ in range(num_peaks):
                    initial_guess.extend([max(y), x[y.idxmax()], 10.0, min(y)])
            else:
                # Convert initial_guesses to a list of lists if not already
                if not isinstance(initial_guesses, list):
                    initial_guesses = [initial_guesses]
                initial_guess = [param for guess in initial_guesses for param in guess]

            # Use curve_fit to optimize and obtain optimal parameters
            popt, _ = curve_fit(composite, x, y, p0=initial_guess)

            # Create fitted data using the composite function
            fitted_data = composite(x, *popt)

        # Store the fitted data in the attribute 'fitted_data'
        self.fitted_data = pd.DataFrame({'Energy': x, 'Intensity': fitted_data})

        # Plot the original data and the fitted curve
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label=data)
            plt.plot(x, fitted_data, label=f'{fitting_type} Fit')
            plt.xlabel('Energy (eV)')
            plt.ylabel(data)
            plt.title('Curve Fitting')
            plt.legend()
            plt.grid()
            plt.show()

    def find_peak(self, x, y, threshold=0.1, prominence=10):
        """
        Find and store peaks in the provided Raman spectrum data.

        Parameters:
        - x (array-like): The x-axis data (e.g., Wavenumber).
        - y (array-like): The y-axis data (e.g., Intensity or Net_Intensity).
        - threshold (float, optional): Minimum relative amplitude of peaks.
        - prominence (float, optional): Minimum prominence of peaks.

        Returns:
        - pandas.DataFrame: A DataFrame containing the detected peak wavenumber and intensities.

        Raises:
        - ValueError: If the input data is invalid.
        """
        if len(x) != len(y):
            raise ValueError("Input data lengths must match.")

        # Find peaks in the data using scipy.signal's find_peaks
        peaks, _ = find_peaks(y, height=threshold, prominence=prominence)

        # Store the peak positions and intensities
        peak_positions = x[peaks]
        peak_intensities = y[peaks]

        # Create a DataFrame to store peak information
        peak_data = pd.DataFrame({'Energy': peak_positions, 'Intensity': peak_intensities})

        # Store the peak data in the 'peak_values' attribute
        self.peak_values = peak_data

        # Return the detected peak data
        return peak_data["Energy"]

    def plot_spectrum(self, data='Intensity', indicate_peaks=True, fitted_plot=True,
                      x_lim=[1.93, 2.1],
                      axis_lable = ["Energy (eV)", "Intensity"],
                      data_lable = True,
                      legend = True, grid = True,
                      save=False, save_name=None
                      ):
        """
        Plot the Raman spectrum data and the fitted data on the same graph.

        Parameters:
        - data (str, optional): The data column to plot ('Intensity' or 'Net_Intensity').
        - indicate_peaks (bool, optional): Whether to indicate detected peaks on the plot.
        - fitted_plot (bool, optional): Plot the fitted curve if True
        - x_lim (list, optional): x-axis limits
        - axis_lable (list, optional): lables of the axes
        - data_lable (bool, optional): Lable the data if True
        - Legend (bool, optional): Provide legend if True
        - grid (bool, optional): Show grid if True
        - save (bool, optional): Save image if True
        - save_name (str, optional): filename of the saved image
        
        Raises:
        - ValueError: If data is not loaded or if an invalid data option is provided.
        """
        if self.data is None:
            raise ValueError("Data must be loaded before plotting.")

        valid_data_options = ['Intensity', 'Net_Intensity', 'Normalised_Intensity', 'Normalised_Net_Intensity']

        if data not in valid_data_options:
            raise ValueError("Invalid data option. Use one of the following: "
                             f"{', '.join(valid_data_options)}.")

        x = self.data['Energy']
        y = self.data[data]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=f'Original {data}')

        # Plot the fitted_curve into the same plot, if any.
        if fitted_plot:
            if self.fitted_data is not None:
                fitted_x = self.fitted_data['Energy']
                fitted_y = self.fitted_data['Intensity']
                plt.plot(fitted_x, fitted_y, label=f'Fitted {data}')

        if indicate_peaks and self.peak_values is not None:
            # Indicate peaks on the plot
            peak_x = self.peak_values['Energy']
            peak_y = self.peak_values['Intensity']
            plt.scatter(peak_x, peak_y, color='red', marker='o', label='Peaks')
            # Add labels for peak positions            
            if data_lable:
                for i, x_peak in enumerate(peak_x):
                    plt.annotate(f'{x_peak:.1f}', (x_peak, peak_y.iloc[i]),
                                xytext=(5, 5), textcoords='offset points')

        # Plot and lable the diagram
        sample = self.filename.removesuffix('.txt')
        plt.xlabel(axis_lable[0])
        plt.ylabel(data)
        plt.title(f'Spectrum of sample {sample}')
        plt.xlim(x_lim)
        plt.ylim(bottom=0)
        if legend:
            plt.legend()
        if grid:
            plt.grid()
        
        # OPTIONAl: save the image
        if save:
            if save_name == None:
                save_name = sample + 'png'
            plt.savefig(save_name)  # use savefig() before show()
        
        plt.show()

