import sys
import numpy as np
import torch

# Load the SALAMA 1D-2022 model
# Specify the path to the directory containing the SALAMA1D_22 model
Path_to_SALAMA = '../models/SALAMA1D_22/'
sys.path.append(Path_to_SALAMA)  # Add the model directory to the system path

# Import the Model class from the model script within the SALAMA1D_22 directory
from model import Model
SALAMA = Model()  # Instantiate the SALAMA model

# Generate a random 15x15 map of meteorological variables for testing
# In practice, you would replace this with actual simulation data, e.g., from the DWD (German Weather Service)
rng = np.random.default_rng()  # Create a random number generator instance
inputvals = rng.normal(size=(15, 15, 10, 65))  # Generate random input data with shape (15, 15, 10, 65)

# Compute probabilities using the SALAMA model
probabilities = SALAMA.evaluate(inputvals)  # Evaluate the model on the random input data
print(probabilities.shape)  # Output shape should be (15, 15), corresponding to the map dimensions

# Compute saliency for a specific example input
# Generate a single random example with shape (10, 65), representing one set of meteorological variables
x = rng.normal(size=(10, 65))

# Compute the saliency map for the example input
saliency = SALAMA.get_saliency(x)  # Obtain the saliency map, which indicates the importance of each input feature
print(saliency.shape)  # Output shape should match the input shape (10, 65)








