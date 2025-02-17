import torch
import torch.nn as nn
import time
from tcngan.utils.constants import LATENT_DIM
def train_gan(generator, discriminator, dataloader, epochs, lr=0.0002):
    """
    Trains a Generative Adversarial Network (GAN).

    Args:
        generator (nn.Module): The generator model that creates fake data.
        discriminator (nn.Module): The discriminator model that distinguishes real data from fake data.
        dataloader (DataLoader): DataLoader providing real data for training.
        epochs (int): Number of training epochs.
        lr (float, optional): Learning rate for optimizers. Defaults to 0.0002.
    """
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        for real_data in dataloader:
            start = time.time()
            batch_size = real_data.size(0)
            
            opt_d.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM)
            fake_data = generator(z)
            real_loss = criterion(discriminator(real_data), torch.ones(batch_size, 1))
            fake_loss = criterion(discriminator(fake_data.detach()), torch.zeros(batch_size, 1))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
            opt_g.zero_grad()
            validity = discriminator(fake_data)
            g_loss = criterion(validity, torch.ones(batch_size, 1))
            g_loss.backward()
            opt_g.step()
            print(f'Batch done in {time.time() - start} seconds')