import torch
import torch.nn as nn


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
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Processes the input sequence using the defined layers.
        """
        lstm_out, (h_n, _) = self.lstm(x)
        validity = self.fc(h_n.squeeze(0))
        return validity