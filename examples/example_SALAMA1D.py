import sys
import numpy as np
import torch

# Load the SALAMA 1D-2022 model
# Specify the path to the directory containing the SALAMA1D model
Path_to_SALAMA = '../models/SALAMA1D/'
sys.path.append(Path_to_SALAMA)  # Add the model directory to the system path

# Import the Model class from the model script within the SALAMA1D directory
from model import Model
SALAMA = Model(version=2022)  # Instantiate SALAMA1D-2022
"""
The SALAMA object is designed to operate on numpy arrays or torch tensors of shape (N, 10, 65):
The dimension of size 10 stands for the following 10 fields (in this order):
 - zonal wind speed
 - meridional wind speed
 - temperature
 - pressure
 - specific humidity
 - cloud water mixing ratio
 - graupel mixing ratio
 - cloud cover
 - vertical wind speed.

The dimension of size 65 stands for the 65 vertical main levels of ICON-D2(-EPS), from the top to the ground, as documented in 
https://www.dwd.de/SharedDocs/downloads/DE/modelldokumentationen/nwv/icon/icon_dbbeschr_aktuell.html
Vertical wind speed is actually a half-level field in ICON-D2-EPS and, therefore, available on 66 levels. SALAMA 1D has been trained on data in which level 1 
has been discarded to make the data compatible with the other variables.

The dimension of size N stands for the number of examples one wishes to evaluate, e.g. 542040 for the number of forecast grid points
"""

# Generate a random array of meteorological variables for testing
# In practice, you would replace this with actual simulation data, e.g., from the DWD (German Meteorological Service)
rng = np.random.default_rng()  # Create a random number generator instance
inputvals = rng.normal(size=(100, 10, 65))  # Generate random input data with shape (100, 10, 65)


# Compute probabilities using the SALAMA model
probabilities = SALAMA.evaluate(inputvals)  # Evaluate the model on the random input data
print(probabilities.shape)  # Output shape should be (100,), corresponding to first dimension of inputvals










