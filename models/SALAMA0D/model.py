import sys
import torch
from architecture import Architecture
import os
import numpy as np


class Model:
	"""
	This class provides methods for evaluating the probability of thunderstorm occurrence 
	based on a set of meteorological variables. The evaluation is performed using a 
	trained machine learning model built with PyTorch.
	"""
	def __init__(self):
		"""
		Initialize the Model class by loading the pre-trained model, scaling parameters, and 
		other necessary configurations for evaluating thunderstorm probability.
	
		Parameters:

		Attributes:
			fieldnames (list[str]): Names of the meteorological fields used in the model.
			shortNameOf (dict): Mapping of field names to their shorthand versions used on the DWD open-data server
			model (torch.nn.Module): The PyTorch model architecture loaded with pre-trained weights.
			scaler_mean (torch.Tensor): Tensor of mean values for input normalization.
			scaler_std (torch.Tensor): Tensor of standard deviation values for input normalization.
			climatology (float): Probability of thunderstorm occurrence based on climatology.
			nt_to_t_ratio (float): Ratio used for model calibration based on climatology.
		"""
		
		# Change working directory to the directory containing this script
		current_working_dir = os.path.dirname(__file__)
		old_working_dir = os.getcwd()
		os.chdir(current_working_dir)

		# field names and their types
		self.fieldnames = ['CAPE_ML', 'CEILING', 'CLCH', 'CLCL', 'CLCM', 'CLCT', 'DBZ_CMAX', 'ECHOTOP', 'OMEGA500', 'PMSL', 'PS', 'RELHUM700', 'T_2M', 'TQC', 'TQC_DIA', 'TQG', 'TQI', 'TQI_DIA', 'TQV', 'TQV_DIA', 'TWATER']
		self.shortNameOf = {'U': 'u', 'V': 'v', 'T': 't', 'P': 'pres', 'QV': 'q', 'QC': 'clwmr', 'QI': 'QI', 'QG': 'grle', 'CLC': 'ccl', 'W': 'wz', 'CAPE_ML': 'CAPE_ML', 'CEILING': 'ceil', 'CLCH': 'CLCH', 'CLCL': 'CLCL', 'CLCM': 'CLCM', 'CLCT': 'CLCT', 'DBZ_CMAX': 'DBZ_CMAX', 'ECHOTOP': 'ECHOTOP', 'OMEGA500': 'w', 'PMSL': 'prmsl', 'PS': 'sp', 'RELHUM700': 'r', 'T_2M': '2t', 'TQC': 'TQC', 'TQC_DIA': 'TQC_DIA', 'TQG': 'TQG', 'TQI': 'TQI', 'TQI_DIA': 'TQI_DIA', 'TQV': 'TQV', 'TQV_DIA': 'TQV_DIA', 'TWATER': 'TWATER'}

		# load model (Feedforward NN with 3 hidden layers, 20 nodes per hidden layer)
		input_dim = 21
		hidden_sizes = np.full(shape=3, fill_value=20, dtype='i')
		output_dim = 1
		self.model = Architecture(input_dim, hidden_sizes, output_dim)
		self.model.load_state_dict(torch.load("state_dict.pth"))

		
		# Load mean and standard deviation for input rescaling (data has been rescaled accordingly during training for normalization)
		self.scaler_mean = torch.load("scaling_mean.pt")
		self.scaler_std = torch.load("scaling_std.pt")

		# Climatology data (probability of thunderstorm occurrence without prior input knowledge)
		self.climatology = 0.0199884572519361
		self.nt_to_t_ratio = (1.-self.climatology)/self.climatology

		# Revert to the old working directory
		os.chdir(old_working_dir)


	def evaluate(self, x, calibration_required=True, consider_ensembles=False):
		"""
		Evaluate the probability of thunderstorm occurrence for a given set of meteorological 
		data using the trained model.

		Parameters:
			x (torch.Tensor or np.ndarray): Input data of shape (*, 21), where * denotes 
											an arbitrary number of dimensions >= 1.

		Returns:
			torch.Tensor or np.ndarray: Predicted probability of thunderstorm occurrence with 
										the same shape as the input except for the last
										dimension.
		"""
		# Ensure input is a torch tensor
		is_numpy = False
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			is_numpy = True

		# Verify and reshape input dimensions to (-1, 21)
		shape = x.size()
		if shape[-1] != 21:
			raise ValueError("Input shape needs to be of the form (*, 65).")
		if len(shape) >= 3:
			x = x.reshape(-1, shape[-1])

		# Rescale fields to order 1
		x = (x - self.scaler_mean) / self.scaler_std
		x = x.float()
		output = torch.sigmoid(self.model(x))

		# Apply model calibration based on Vahid Yousefnia et al. (2024)
		result = output / (output + self.nt_to_t_ratio * (1.0 - output))
		result = result.squeeze(-1)

		# Restore result shape and format
		if len(shape) >= 3:
			result = result.reshape(shape[:-1])
		if is_numpy:
			return result.detach().numpy()
		else:
			return result





		