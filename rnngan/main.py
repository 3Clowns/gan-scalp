import numpy as np
import pandas as pd
from torch.utils.data import DataLoader 

from rnngan.models.generator import Generator
from rnngan.models.discriminator import Discriminator
from rnngan.data.dataset import TimeSeriesDataset
from rnngan.training.train_gan import train_gan
from rnngan.utils.constants import LATENT_DIM, OUTPUT_DIM, WINDOW_SIZE

data = pd.read_csv('data/stocks.csv')
data = data['close'].head(100000)
data = np.array(data)

dataset = TimeSeriesDataset(data, seq_len=WINDOW_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

generator = Generator(
    input_dim=LATENT_DIM, 
    output_dim=OUTPUT_DIM,
    seq_len=WINDOW_SIZE,
    hidden_dims=[64, 128]
)

discriminator = Discriminator(
    input_dim=OUTPUT_DIM,
    hidden_dims=[128, 64]
)

train_gan(generator, discriminator, dataloader, epochs=1)