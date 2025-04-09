import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_block import TemporalBlock

class Generator(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super().__init__()
        self.tcn = nn.ModuleList([
            TemporalBlock(latent_dim, hidden_dim, kernel_size=1, dilation=1),
            *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, dilation=2**i) for i in range(4)]
        ])
        #self.last = nn.Conv1d(hidden_dim, input_dim, kernel_size=1, stride=1, dilation=1)
        self.heads = nn.ModuleDict({
            'lrl': nn.Conv1d(hidden_dim, 1, kernel_size=1),
            'log_dh': nn.Conv1d(hidden_dim, 1, kernel_size=1),
            'delta_c': nn.Conv1d(hidden_dim, 1, kernel_size=1),
            'delta_o': nn.Conv1d(hidden_dim, 1, kernel_size=1),
            'log_volume': nn.Conv1d(hidden_dim, 1, kernel_size=1)
        })

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = x + sum(skip_layers)
        
        outputs = {}
        outputs['lrl'] = self.heads['lrl'](x)
        outputs['log_dh'] = self.heads['log_dh'](x)
        outputs['delta_c'] = torch.sigmoid(self.heads['delta_c'](x)) * 0.98 + 0.01
        outputs['delta_o'] = torch.sigmoid(self.heads['delta_o'](x)) * 0.98 + 0.01
        outputs['log_volume'] = self.heads['log_volume'](x)
        
        return torch.cat([outputs[k] for k in ['lrl', 'log_dh', 'delta_c', 'delta_o', 'log_volume']], dim=1)

class Discriminator(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim):
        super().__init__()
        self.tcn = nn.ModuleList([
            TemporalBlock(input_dim, hidden_dim, kernel_size=1, dilation=1),
            *[TemporalBlock(hidden_dim, hidden_dim, kernel_size=2, dilation=2**i) for i in range(4)]
        ])
        self.last = nn.Conv1d(hidden_dim, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1), nn.Sigmoid())

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()