import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader 

from models.generator import Generator
from models.discriminator import Discriminator
from data.dataset import TimeSeriesDataset
from training.train_gan import train_gan
from utils.constants import LATENT_DIM, BATCH_SIZE, WINDOW_SIZE, DEVICE

data = pd.read_csv('tcngan/data/stocks.csv')
data = data[data["Ticker"] == "LKOH"]
data = np.log(data['close'] / data['close'].shift(1))
data = data.dropna()
data = np.array(data)

dataset = TimeSeriesDataset(data, seq_len=WINDOW_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
print(DEVICE)
generator = Generator(
    input_dim=LATENT_DIM, 
    hidden_dim=10
).to(DEVICE)

discriminator = Discriminator(
    seq_len=WINDOW_SIZE,
    hidden_dim=10
).to(DEVICE)

wandb.init(
    project = "scalp_gan",
    config = {
        "epochs": 20,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "Loss": "BCE"
    }
)

train_gan(generator, discriminator, dataloader, epochs=1)