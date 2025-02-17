import torch
import torch.nn as nn

from tcngan.models.temporal_block import TemporalBlock

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        """
        Initializes a model (e.g., a Discriminator) for processing sequential data.

        Args:
            input_dim (int): Dimensionality of the input data (e.g., number of features).
            hidden_dims (list, optional): List of integers specifying the number of channels 
                                        in each hidden layer. Defaults to [128, 64].
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        in_dim = input_dim
        for hidden in hidden_dims:
            self.layers.append(
                TemporalBlock(in_dim, hidden, kernel_size=3, dilation=2)
            )
            in_dim = hidden
        
        self.final = nn.Conv1d(in_dim, 1, 3, padding=1)

    def forward(self, x):
        """
        Processes the input sequence using the defined layers.
        """
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.final(x).view(x.size(0), -1))