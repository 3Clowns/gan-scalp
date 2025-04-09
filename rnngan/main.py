import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
import torch.nn as nn
from torch import optim

from gan import Discriminator, Generator
from dataset import TimeSeriesDataset
from train_gan import train_gan
from constants import BATCH_SIZE, WINDOW_SIZE, DEVICE, LATENT_DIM, HIDDEN_DIM, FEATURES, INPUT_DIM, NUM_LAYERS
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('/home/jovyan/scalp-gan/tcngan/stocks_lkoh.csv')
'''def ohlcv_to_features(df):
    features = pd.DataFrame()
    features['lro'] = np.log(df['open'] / df['open'].shift(1))
    features['lrc'] = np.log(df['close'] / df['close'].shift(1))
    features['lrh'] = np.log(df['high'] / df['high'].shift(1))
    features['lrl'] = np.log(df['low'] / df['low'].shift(1))

    features['log_volume'] = np.log(df['volume'] + 1e-8) 
    features =  features.dropna()
    
    scalers = {}
    for feature in FEATURES:
        scaler = StandardScaler()
        features[feature] = scaler.fit_transform(features[[feature]].values).flatten()
        scalers[feature] = scaler
    return features, scalers'''

def ohlcv_to_features(df):
    features = pd.DataFrame(index=df.index)
   
    features['lrl'] = np.log(df['low'] / df['low'].shift(1))   
    delta_h = df['high'] - df['low']
    delta_h = delta_h.clip(lower=1e-8)   
    features['delta_c'] = (df['close'] - df['low']) / delta_h
    features['delta_o'] = (df['open'] - df['low']) / delta_h   
    features['log_volume'] = np.log(df['volume'] + 1e-8)   
    features['log_dh'] = np.log(delta_h)
    
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    
    scalers = {
        'lrl': StandardScaler(),
        'log_dh': StandardScaler(),
        'delta_c': MinMaxScaler(feature_range=(0.01, 0.99)),
        'delta_o': MinMaxScaler(feature_range=(0.01, 0.99)),
        'log_volume': StandardScaler()
    }
    
    for col, scaler in scalers.items():
        features[col] = scaler.fit_transform(features[[col]]).flatten()
        
    return features, scalers    

data, scalers = ohlcv_to_features(df)   

print(f'Use device: {DEVICE}')
print(f"Data size = {data.shape}")

N_EPOCHS = 1000
PLOT_FREQUENCY = 1
SAVE_FREQUENCY = 100

dataset = TimeSeriesDataset(
    data=data,
    seq_len=WINDOW_SIZE,
    features=FEATURES
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=4
)

generator = Generator(
    latent_dim=LATENT_DIM,
    hidden_dim=HIDDEN_DIM,
    input_dim=INPUT_DIM,
    num_layers = NUM_LAYERS
).to(DEVICE)

discriminator = Discriminator(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers = NUM_LAYERS
).to(DEVICE)

'''def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

generator.apply(init_weights)
discriminator.apply(init_weights)'''

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, weight_decay=1e-5)

discriminator_losses, generator_losses = train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, 
                                                dataloader,  df, scalers, n_epochs=N_EPOCHS, plot_frequency=PLOT_FREQUENCY,
                                                save_frequency=SAVE_FREQUENCY, model_prefix='RNN')