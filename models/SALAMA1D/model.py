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
    def __init__(self, version=2022):
        """
        Initialize the Model class by loading the pre-trained model, scaling parameters, and 
        other necessary configurations for evaluating thunderstorm probability.
    
        Parameters:
            version (int): The version of the SALAMA1D model to use. 
                       - 2021: Model trained on summer 2021 data.
                       - 2022 (default): Model trained on summers 2021 and 2022 data.

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

        # Field names and their types
        self.fieldnames = ['U', 'V', 'T', 'P', 'QV', 'QC', 'QI', 'QG', 'CLC', 'W']
        self.shortNameOf = {
            'U': 'u', 'V': 'v', 'T': 't', 'P': 'pres', 'QV': 'q', 'QC': 'clwmr', 
            'QI': 'QI', 'QG': 'grle', 'CLC': 'ccl', 'W': 'wz', 'CAPE_ML': 'CAPE_ML', 
            'CEILING': 'ceil', 'CLCH': 'CLCH', 'CLCL': 'CLCL', 'CLCM': 'CLCM', 
            'CLCT': 'CLCT', 'DBZ_CMAX': 'DBZ_CMAX', 'ECHOTOP': 'ECHOTOP', 
            'OMEGA500': 'w', 'PMSL': 'prmsl', 'PS': 'sp', 'RELHUM700': 'r', 
            'T_2M': '2t', 'TQC': 'TQC', 'TQC_DIA': 'TQC_DIA', 'TQG': 'TQG', 
            'TQI': 'TQI', 'TQI_DIA': 'TQI_DIA', 'TQV': 'TQV', 'TQV_DIA': 'TQV_DIA', 
            'TWATER': 'TWATER'
        }

        # Load the pre-trained model
        self.model = Architecture(10, 65, 8, 3, 5)
        self.model.load_state_dict(torch.load(f"state_dict_{version}.pth"))
        
        # Load mean and standard deviation for input rescaling (data has been rescaled accordingly during training for normalization)
        self.scaler_mean = torch.load(f"scaling_mean_{version}.pt").unsqueeze(0).unsqueeze(2)
        self.scaler_std = torch.load(f"scaling_std_{version}.pt").unsqueeze(0).unsqueeze(2)

        # Climatology data (probability of thunderstorm occurrence without prior input knowledge)
        self.climatology = 0.0193
        self.nt_to_t_ratio = (1.0 - self.climatology) / self.climatology

        # Revert to the old working directory
        os.chdir(old_working_dir)

    def evaluate(self, x):
        """
        Evaluate the probability of thunderstorm occurrence for a given set of meteorological 
        data using the trained model.

        Parameters:
            x (torch.Tensor or np.ndarray): Input data of shape (*, 10, 65), where * denotes 
                                            an arbitrary number of dimensions >= 1.

        Returns:
            torch.Tensor or np.ndarray: Predicted probability of thunderstorm occurrence with 
                                        the same shape as the input except for the last two 
                                        dimensions.
        """
        # Ensure input is a torch tensor
        is_numpy = False
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            is_numpy = True

        # Verify and reshape input dimensions to (-1, 10, 65)
        shape = x.size()
        if shape[-2] != 10 or shape[-1] != 65:
            raise ValueError("Input shape needs to be of the form (*, 10, 65).")
        if len(shape) >= 4:
            x = x.reshape(-1, shape[-2], shape[-1])

        # Rescale fields to order 1
        x = (x - self.scaler_mean) / self.scaler_std
        x = x.float()
        output = torch.sigmoid(self.model(x))

        # Apply model calibration based on Vahid Yousefnia et al. (2024)
        result = output / (output + self.nt_to_t_ratio * (1.0 - output))
        result = result.squeeze(-1)

        # Restore result shape and format
        if len(shape) >= 4:
            result = result.reshape(shape[:-2])
        if is_numpy:
            return result.detach().numpy()
        else:
            return result




		
