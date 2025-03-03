import torch
import torch.nn as nn

from tcngan.models.temporal_block import TemporalBlock

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32, 16]):
        """
        Initializes a model (e.g., a Discriminator) for processing sequential data.

        Args:
            input_dim (int): Dimensionality of the input data (e.g., number of features).
            hidden_dims (list, optional): List of integers specifying the number of channels 
                                        in each hidden layer. Defaults to [128, 64].
        """
        super().__init__()
        self.hidden_dims = hidden_dims

        self.rnn = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        '''self.rnn_layers = nn.ModuleList([
            nn.LSTM(hidden_dims[i], hidden_dims[i + 1], batch_first=True)
            for i in range(len(hidden_dims) - 1)
        ])'''
        self.fc = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Processes the input sequence using the defined layers.
        """
        out, _ = self.rnn(x)
        '''for rnn_layer in self.rnn_layers:
            out, _ = rnn_layer(out)'''
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)
