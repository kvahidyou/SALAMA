import sys
import torch
from architecture import Architecture
import os
import numpy as np

class Model:
    """
    This class provides methods to evaluate the probability of thunderstorm occurrence 
    based on a set of meteorological variables. The evaluation is performed using a 
    trained machine learning model built with PyTorch. The class also includes a method 
    for computing saliency maps, which help understand the importance of input features 
    in the model's predictions.
    """
    def __init__(self):
        """
        Initialize the Model class by setting up the required parameters and loading the 
        pre-trained model, mean and standard deviation for input scaling, and other necessary 
        data. The working directory is temporarily changed to ensure that dependencies are 
        correctly located.
        """
        # Change working directory to the directory containing this script
        current_working_dir = os.path.dirname(__file__)
        old_working_dir = os.getcwd()
        os.chdir(current_working_dir)

        # Field names and their types
        self.fieldnames = ['U', 'V', 'T', 'P', 'QV', 'QC', 'QI', 'QG', 'CLC', 'W']
        self.fieldtypes = ['MAIN_LEVEL_FIELDS', 'HALF_LEVEL_FIELDS']
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
        self.model.load_state_dict(torch.load("state_dict.pth"))
        
        # Load mean and standard deviation for input rescaling
        self.scaler_mean = torch.load("scaling_mean.pt").unsqueeze(0).unsqueeze(2)
        self.scaler_std = torch.load("scaling_std.pt").unsqueeze(0).unsqueeze(2)

        # Climatology data (probability of thunderstorm occurrence without prior input knowledge)
        self.climatology = 0.0199884572519361
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

        # Apply model calibration based on Yousefnia et al. (2024)
        result = output / (output + self.nt_to_t_ratio * (1.0 - output))
        result = result.squeeze(-1)

        # Restore result shape and format
        if len(shape) >= 4:
            result = result.reshape(shape[:-2])
        if is_numpy:
            return result.detach().numpy()
        else:
            return result

    def get_saliency(self, x):
        """
        Compute the saliency map for a given input, which highlights the importance of 
        each input feature in the model's prediction.

        Parameters:
            x (torch.Tensor or np.ndarray): Input data of shape (10, 65).

        Returns:
            np.ndarray: Saliency map with the same shape as the input.
        """
        # Ensure the input is a torch tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        # Ensure the input is at least 3-D
        shape = x.size()
        unsqueezing_needed = False
        if len(shape) < 3:
            unsqueezing_needed = True
            x = x.unsqueeze(0)

        # Set up the model for evaluation and require gradients for input
        self.model.eval()
        x = x.float()
        x.requires_grad_()

        # Perform a forward pass
        class_score = self.model(x)
        prob = torch.sigmoid(class_score) / (
            torch.sigmoid(class_score) + self.nt_to_t_ratio * (1.0 - torch.sigmoid(class_score))
        )

        # Compute the gradient of the output probability with respect to the input
        prob.backward()
        saliency = x.grad.data.abs()  # (10, 65)

        # Restore original input shape if needed
        if unsqueezing_needed:
            saliency = saliency.squeeze(0)

        return saliency.detach().numpy()



		