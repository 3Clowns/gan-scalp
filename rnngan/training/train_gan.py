import torch
import torch.nn as nn
import time
from rnngan.utils.constants import LATENT_DIM
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
    st = 0
    fn = 0
    for epoch in range(epochs):
        st = time.time()
        for real_data in dataloader:
            start = time.time()
            batch_size = real_data.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            real_data = real_data.unsqueeze(-1)
            # Train Discriminator
            opt_d.zero_grad()
            real_outputs = discriminator(real_data)
            d_real_loss = criterion(real_outputs, real_labels)

            z = torch.randn(batch_size, real_data.size(1), generator.input_dim)
            fake_data = generator(z)
            fake_outputs = discriminator(fake_data.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            fake_outputs = discriminator(fake_data)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            opt_g.step()
            print(f'Batch done in {time.time() - start} seconds')
        fn = time.time() - st
        print(f'Epoch {epoch} done in {fn} seconds.')
