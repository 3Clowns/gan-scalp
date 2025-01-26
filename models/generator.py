import torch
import torch.nn as nn

from models.generator import TemporalBlock

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dims=[64, 128]):
        """
        Initializes a model (e.g., a Generator) for processing sequential data.

        Args:
            input_dim (int): Dimensionality of the input noise vector (latent space).
            output_dim (int): Dimensionality of the output data (e.g., number of features).
            seq_len (int): Length of the output sequence (temporal dimension).
            hidden_dims (list, optional): List of integers specifying the number of channels 
                                        in each hidden layer. Defaults to [64, 128].
        """
        super().__init__()
        self.init_size = seq_len
        self.layers = nn.ModuleList()
        
        in_dim = input_dim
        for hidden in hidden_dims:
            self.layers.append(
                TemporalBlock(in_dim, hidden, kernel_size=3, dilation=2)
            )
            in_dim = hidden
        
        self.final = nn.Sequential(
            nn.Conv1d(in_dim, output_dim, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Processes input noise vector (z) and generates the output sequence.
        """
        z = z.view(z.size(0), -1, self.init_size)
        for layer in self.layers:
            z = layer(z)
        return self.final(z)