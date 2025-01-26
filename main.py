import numpy as np
from torch.utils.data import DataLoader 

from models.generator import Generator
from models.discriminator import Discriminator
from data.dataset import TimeSeriesDataset
from training.train_gan import train_gan
from utils.constants import LATENT_DIM, OUTPUT_DIM, WINDOW_SIZE

data = np.loadtxt("data/timeseries.csv")

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

train_gan(generator, discriminator, dataloader, epochs=1000)