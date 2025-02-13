import sys
import numpy as np
import torch

# Load the SALAMA 0D model
# Specify the path to the directory containing the SALAMA0D model
Path_to_SALAMA = '../models/SALAMA0D/'
sys.path.append(Path_to_SALAMA)  # Add the model directory to the system path

# Import the Model class from the model script within the SALAMA0D directory
from model import Model
SALAMA = Model()  # Instantiate SALAMA0D
"""
The SALAMA object is designed to operate on numpy arrays or torch tensors of shape (N, 21):
The dimension of size 21 stands for the following fields (in this order):
 - mixed-layer CAPE (J/kg)
 - ceiling height (m)
 - cloud cover of high-level clouds (%)
 - cloud cover of low-level clouds (%)
 - cloud cover of mid-level clouds (%)
 - total cloud cover (%)
 - column-maximal radar reflectivity (dBZ)
 - echotop pressure (Pa)
 - omega at 500 hPa (Pa/s)
 - pressure normalized to mean-sea level
 - surface pressure
 - relative humidity at 700hPa (%)
 - 2m temperature
 - column-integrated cloud water (kg/m^2)
 - column-integrated cloud water, including sub-grid scale (kg/m^2)
 - column-integrated graupel (kg/m^2)
 - column-integrated ice particles (kg/m^2)
 - column-integrated ice particles, including sub-grid scale (kg/m^2)
 - column-integrated water vapor (kg/m^2)
 - column-integrated water vapor, including sub-grid scale (kg/m^2)
 - total column-integrated water content (kg/m^2)

The dimension of size N stands for the number of examples one wishes to evaluate, e.g. 542040 for the number of forecast grid points
"""

# Generate a random array of meteorological variables for testing
# In practice, you would replace this with actual simulation data, e.g., from the DWD (German Meteorological Service)
rng = np.random.default_rng()  # Create a random number generator instance
inputvals = rng.normal(size=(100, 21))  # Generate random input data with shape (100, 21)


# Compute probabilities using the SALAMA model
probabilities = SALAMA.evaluate(inputvals)  # Evaluate the model on the random input data
print(probabilities.shape)  # Output shape should be (100,), corresponding to first dimension of inputvals










