import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        '''self.output = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, input_dim)
        )'''

        self.heads = nn.ModuleDict({
            'lrl': nn.Linear(hidden_dim, 1),
            'log_dh': nn.Linear(hidden_dim, 1),
            'delta_c': nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'delta_o': nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'log_volume': nn.Linear(hidden_dim, 1)
        })
        
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )

        self._init_weights()


    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        projected = self.hidden_proj(lstm_out)
        
        params = {}
        for name, head in self.heads.items():
            params[name] = head(projected).squeeze(-1)
        
        params['delta_c'] = params['delta_c'] * 0.98 + 0.01  # [0.01, 0.99]
        params['delta_o'] = params['delta_o'] * 0.98 + 0.01  # [0.01, 0.99]
        
        return torch.stack([
            params['lrl'],
            params['log_dh'],
            params['delta_c'],
            params['delta_o'],
            params['log_volume']
        ], dim=2)
        #return self.output(lstm_out) 

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1]).squeeze()