import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
from torch import optim

from gan import Discriminator, Generator
from dataset import TimeSeriesDataset
from train_gan import train_gan
from constants import LATENT_DIM, BATCH_SIZE, WINDOW_SIZE, DEVICE

data = pd.read_csv('stocks.csv')
data = data[data["Ticker"] == "LKOH"]
data = pd.DataFrame({
    'log_return': np.log(data['close'] / data['close'].shift(1))
}).dropna().tail(131040)

print(f'Use device: {DEVICE}')
print(f"Data size = {data.shape}")
print(type(data['log_return'].values,))

N_EPOCHS = 800
PLOT_FREQUENCY = 50
SAVE_FREQUENCY = 50

dataset = TimeSeriesDataset(data['log_return'].values, seq_len=WINDOW_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    persistent_workers=True
)

generator = Generator(
    input_dim=LATENT_DIM, 
    hidden_dim=10
).to(DEVICE)

discriminator = Discriminator(
    seq_len=WINDOW_SIZE,
    hidden_dim=10
).to(DEVICE)

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(generator_optimizer, 'min', patience=5)
scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, 'min', patience=5)

discriminator_losses, generator_losses = train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, data, n_epochs=N_EPOCHS, plot_frequency=PLOT_FREQUENCY, save_frequency=SAVE_FREQUENCY, model_prefix='TCN')