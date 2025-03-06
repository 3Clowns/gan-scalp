import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 
import wandb
from rnngan.models.generator import Generator
from rnngan.models.discriminator import Discriminator
from rnngan.data.dataset import TimeSeriesDataset
from rnngan.training.train_gan import train_gan
from rnngan.utils.constants import LATENT_DIM, OUTPUT_DIM, WINDOW_SIZE, DEVICE
print(DEVICE)
data = pd.read_csv('data/stocks.csv')
data = data[data['Ticker']=='LKOH']
data = data.head(100001)
data['close'] = np.log((data['close']/ data['close'].shift(1)))
data = data['close'].dropna()
data=data[::-1]
data = np.array(data)
print(data.shape)


dataset = TimeSeriesDataset(data, seq_len=WINDOW_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False
)

generator = Generator(
    input_dim=LATENT_DIM, 
    output_dim=OUTPUT_DIM,
    seq_len=WINDOW_SIZE,
<<<<<<< HEAD
    hidden_dims=[64]
)

discriminator = Discriminator(
    input_dim=OUTPUT_DIM,
    hidden_dims=[64]
=======
    hidden_dims=[10]
).to(DEVICE)

discriminator = Discriminator(
    input_dim=OUTPUT_DIM,
    hidden_dims=[10]
).to(DEVICE)

wandb.init(
    project="gan-rl-training",
    config={
        "epochs": 20,
        "batch_size": dataloader.batch_size,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss"
    }
>>>>>>> 17bfbeea1a419a6f0ce634e9f3c611f36b3aaf82
)

train_gan(generator, discriminator, dataloader, epochs=20)
