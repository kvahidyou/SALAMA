import torch
import torch.nn as nn

class Architecture(nn.Module):
    """
    A feedforward neural network architecture implemented using PyTorch's nn.Module.

    Attributes:
        layers (nn.ModuleList):
            A sequential list of layers including input, hidden, and output layers.
        activation (nn.ReLU):
            The activation function applied to the outputs of all layers except the output layer.
    """

    def __init__(self, input_dim, hidden_sizes, output_dim):
        """
        Initializes the feedforward neural network.

        Args:
            input_dim (int):
                The number of input features (nodes in the input layer).
            hidden_sizes (list[int]):
                A list containing the number of nodes in each hidden layer.
            output_dim (int):
                The number of output features (nodes in the output layer).
        """
        super().__init__()

        # Initialize the list to store layers
        self.layers = nn.ModuleList()

        # Create the input layer
        self.layers.append(nn.Linear(in_features=input_dim, out_features=hidden_sizes[0]))

        # Create the hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(in_features=hidden_sizes[i-1], out_features=hidden_sizes[i]))

        # Create the output layer
        self.layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_dim))

        # Define the activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor):
                Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor:
                Output tensor with shape (batch_size, output_dim).
        """
        # Apply each layer in the sequence with activation function
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))

        # Apply the last layer without activation function
        x = self.layers[-1](x)

        return x


