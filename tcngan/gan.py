import torch.nn as nn

from temporal_block import TemporalBlock
from constants import N_ASSETS

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim = 10):
        super().__init__()
        self.tcn = nn.ModuleList([TemporalBlock(input_dim, hidden_dim, kernel_size=1, dilation=1, padding=0),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, dilation=i, padding=i) for i in [1, 2, 4, 8]]])
        self.last = nn.Conv1d(hidden_dim, N_ASSETS, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x

class Discriminator(nn.Module):
    def __init__(self, seq_len, hidden_dim = 10):
        super().__init__()
        self.tcn = nn.ModuleList([TemporalBlock(N_ASSETS, hidden_dim, kernel_size=1, dilation=1, padding=0),
                                 *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, dilation=i, padding=i) for i in [1, 2, 4, 8]]])
        self.last = nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()