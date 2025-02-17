import torch
import torch.nn as nn


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
        self.seq_length = seq_len
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, output_dim),
        )

    def forward(self, z):
        """
        Processes input noise vector (z) and generates the output sequence.
        """
        batch_size = z.size(0)
        z = z.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        lstm_out, _ = self.lstm(z)
        output = self.fc(lstm_out)
        return output