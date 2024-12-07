import torch
import torch.nn as nn

# Class to shuffle 3D tensors along dimension 1
class ShuffleLayer(nn.Module):
    """
    This layer shuffles the blocks of a 3D tensor along the block dimension 
    during training, adding randomness to the block order in the input data. 
    No shuffling is performed during evaluation.
    """
    def __init__(self):
        super(ShuffleLayer, self).__init__()

    def forward(self, x):
        if self.training:
            shuffled_x = torch.empty_like(x)
            batch_size, num_blocks, _ = x.size()
            for i in range(batch_size):
                indices = torch.randperm(num_blocks)  # Random permutation of blocks
                shuffled_x[i] = x[i, indices]
            return shuffled_x
        else:
            return x  # Do not shuffle in evaluation mode

# Grouped linear layer: Each channel is mapped from in_features to out_features with its own linear layer
# This is used for the sliding blocks implementation in the sparse layer
def ParallelLinear(n_channels, in_features, out_features):
    """
    Creates a grouped 1D convolutional layer, where each channel is mapped individually 
    from `in_features` to `out_features`. This effectively implements a sparse linear 
    layer where each sliding block in the input layer is processed independently.

    Parameters:
        n_channels (int): Number of channels (corresponding to the number of blocks).
        in_features (int): Number of input features per block.
        out_features (int): Number of output features per block.

    Returns:
        nn.Conv1d: A 1D convolutional layer implementing the grouped linear operation.
    """
    return nn.Conv1d(in_channels=n_channels, 
                     groups=n_channels, 
                     out_channels=n_channels*out_features, 
                     kernel_size=in_features)

# ML architecture class
class Architecture(nn.Module):
    """
    This class defines the neural network architecture for predicting thunderstorm 
    occurrences. It includes a custom sparse layer that processes blocks of meteorological 
    data profiles and several linear layers to reduce the dimensionality of the model 
    before producing the final prediction.

    Parameters:
        N_ch (int): Number of input channels (fields).
        N_z (int): Number of vertical levels in the input data.
        kernel_size_z (int): Size of the block along the height dimension.
        stride_z (int): Stride (step size) for the sliding blocks.
        N_heightfeatures (int): Number of output features per block in the sparse layer.
    """
    def __init__(self, N_ch, N_z, kernel_size_z, stride_z, N_heightfeatures):
        super().__init__()
        self.N_ch = N_ch
        self.N_z = N_z

        # Activation function used throughout the network
        self.ActivationFunction = nn.ReLU()

        # Step 1: Unfold input layer with sliding block (filter) and stride, then shuffle the blocks
        self.f1 = kernel_size_z
        self.s1 = stride_z
        if (self.N_z - self.f1) % self.s1 != 0:
            raise ValueError("N_z minus kernel_size_z must be multiple of stride_z")
        
        # Unfold layer: Creates sliding blocks from the input layer
        self.unfold = nn.Unfold(kernel_size=(self.f1, 1), stride=(self.s1, 1))
        self.out_ch_1 = (self.N_z - self.f1) // self.s1 + 1  # Number of blocks produced by unfolding
        self.out_feat_1 = self.f1 * self.N_ch  # Number of features per block

        # Shuffle blocks along the dimension that iterates over the blocks
        self.shuffle_blocks = ShuffleLayer()

        # Step 2: Apply a locally connected 1D layer to each sliding block (sparse linear layer)
        self.out_feat_2 = N_heightfeatures  # Number of output features per block
        self.LocallyConnected1d = ParallelLinear(n_channels=self.out_ch_1, 
                                                 in_features=self.out_feat_1, 
                                                 out_features=self.out_feat_2)
        
        # Step 3: Reduce dimensionality with three linear layers, targeting a final dimension of 21
        self.N_start = self.out_ch_1 * self.out_feat_2  # Starting feature count after the sparse layer
        self.N_final = 21  # Final target dimensionality
        reduction_factor = (1.0 * self.N_final / self.N_start) ** (1.0 / 4)  # Uniform reduction factor per layer
        
        # Intermediate layers progressively reduce the dimensionality
        self.N_hidden_1 = int(self.N_start * reduction_factor)
        self.N_hidden_2 = int(self.N_start * reduction_factor * reduction_factor)
        self.N_hidden_3 = int(self.N_start * reduction_factor * reduction_factor * reduction_factor)
       
        self.LinearRed1 = nn.Linear(in_features=self.N_start, out_features=self.N_hidden_1)
        self.LinearRed2 = nn.Linear(in_features=self.N_hidden_1, out_features=self.N_hidden_2)
        self.LinearRed3 = nn.Linear(in_features=self.N_hidden_2, out_features=self.N_hidden_3)
        self.LinearRed4 = nn.Linear(in_features=self.N_hidden_3, out_features=self.N_final)     
        
        # Step 4: Reduce to a single-node output (final prediction)
        self.LinearSALAMA1 = nn.Linear(21, 20)
        self.LinearSALAMA2 = nn.Linear(20, 20)
        self.LinearSALAMA3 = nn.Linear(20, 20)
        self.LinearSALAMA4 = nn.Linear(20, 1)  # Final output layer
    
    def forwardPass_SparseLayer(self, x):
        """
        Forward pass through the sparse layer, which includes unfolding the input, 
        applying a locally connected 1D layer (sparse linear layer), and shuffling the blocks.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, N_ch, N_z).

        Returns:
            torch.Tensor: Output tensor after processing through the sparse layer.
        """
        # Step 1: Unfold input layer into sliding blocks
        x = x.unsqueeze(-1)  # Add a singleton dimension to match expected input shape
        x = self.unfold(x)  # Unfold into blocks
        x = x.transpose(1, 2)  # Transpose to prepare for the next layer

        # Step 2: Apply locally connected 1D layer to each block (sparse linear layer)
        x = self.LocallyConnected1d(x)
        x = x.view(-1, self.out_ch_1, self.out_feat_2)  # Reshape to (batch_size, num_blocks, N_heightfeatures)

        # Step 3: Shuffle the blocks (only during training)
        x = self.shuffle_blocks(x)
        
        return self.ActivationFunction(x)

    def forwardPass_LinearRed1(self, x):
        """
        Forward pass through the first linear reduction layer.

        Parameters:
            x (torch.Tensor): Input tensor after sparse layer processing.

        Returns:
            torch.Tensor: Output after the first linear reduction layer.
        """
        x = x.view(-1, self.N_start)  # Flatten the tensor for linear layer
        x = self.LinearRed1(x)
        return self.ActivationFunction(x)

    def forwardPass_LinearRed2(self, x):
        """
        Forward pass through the second linear reduction layer.

        Parameters:
            x (torch.Tensor): Input tensor from the previous linear layer.

        Returns:
            torch.Tensor: Output after the second linear reduction layer.
        """
        x = self.LinearRed2(x)
        return self.ActivationFunction(x)

    def forwardPass_LinearRed3(self, x):
        """
        Forward pass through the third linear reduction layer.

        Parameters:
            x (torch.Tensor): Input tensor from the previous linear layer.

        Returns:
            torch.Tensor: Output after the third linear reduction layer.
        """
        x = self.LinearRed3(x)
        return self.ActivationFunction(x)

    def forwardPass_LinearRed4(self, x):
        """
        Forward pass through the fourth linear reduction layer.

        Parameters:
            x (torch.Tensor): Input tensor from the previous linear layer.

        Returns:
            torch.Tensor: Output after the fourth linear reduction layer.
        """
        x = self.LinearRed4(x)
        return self.ActivationFunction(x)

    def forwardPass_SALAMA1(self, x):
        """
        Forward pass through the first SALAMA0D layer.

        Parameters:
            x (torch.Tensor): Input tensor from the last linear reduction layer.

        Returns:
            torch.Tensor: Output after the first SALAMA0D layer.
        """
        x = self.LinearSALAMA1(x)
        return self.ActivationFunction(x)

    def forwardPass_SALAMA2(self, x):
        """
        Forward pass through the second SALAMA0D layer.

        Parameters:
            x (torch.Tensor): Input tensor from the previous SALAMA0D layer.

        Returns:
            torch.Tensor: Output after the second SALAMA0D layer.
        """
        x = self.LinearSALAMA2(x)
        return self.ActivationFunction(x)

    def forwardPass_SALAMA3(self, x):
        """
        Forward pass through the third SALAMA0D layer.

        Parameters:
            x (torch.Tensor): Input tensor from the previous SALAMA0D layer.

        Returns:
            torch.Tensor: Output after the third SALAMA0D layer.
        """
        x = self.LinearSALAMA3(x)
        return self.ActivationFunction(x)

    def forwardPass_SALAMA4(self, x):
        """
        Forward pass through the fourth SALAMA0D layer, producing the final output.

        Parameters:
            x (torch.Tensor): Input tensor from the previous SALAMA0D layer.

        Returns:
            torch.Tensor: Final output tensor (single-node prediction).
        """
        x = self.LinearSALAMA4(x)
        return x  # No activation function for the final output

    def forward(self, x):
        """
        Full forward pass through the architecture, from the sparse layer to the final prediction.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, N_ch, N_z).

        Returns:
            torch.Tensor: Final output tensor (single-node prediction).
        """
        # Sparse layer
        x = self.forwardPass_SparseLayer(x)

        # Linear reduction layers
        x = self.forwardPass_LinearRed1(x)
        x = self.forwardPass_LinearRed2(x)
        x = self.forwardPass_LinearRed3(x)
        x = self.forwardPass_LinearRed4(x)

        # SALAMA0D layers
        x = self.forwardPass_SALAMA1(x)
        x = self.forwardPass_SALAMA2(x)
        x = self.forwardPass_SALAMA3(x)
        x = self.forwardPass_SALAMA4(x)

        return x


