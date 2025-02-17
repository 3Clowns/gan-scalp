import torch
import torch.nn as nn

from tcngan.models.temporal_block import TemporalBlock

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dims=[32]):
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
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.rnn = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        '''        
            self.rnn_layers = nn.ModuleList([
            nn.LSTM(hidden_dims[i], hidden_dims[i + 1], batch_first=True)
            for i in range(len(hidden_dims) - 1)
        ])
        '''
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        """
            Processes input noise vector (z) and generates the output sequence.
        """
        z = z.repeat(1, self.seq_len, 1)
        out, _ = self.rnn(z)
        '''for rnn_layer in self.rnn_layers:
            out, _ = rnn_layer(out)'''
        out = self.fc(out)
        return out
