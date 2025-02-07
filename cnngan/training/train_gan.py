import torch
import torch.nn as nn

from utils.constants import LATENT_DIM
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
    mse_loss = nn.MSELoss()

    for epoch in range(1000):
        for real_data in dataloader:
            batch_size = real_data.size(0)
        
            opt_d.zero_grad()
            
            real_pred = discriminator(real_data)
            d_loss_real = criterion(real_pred, torch.ones(batch_size, 1))
            
            z = torch.randn(batch_size, LATENT_DIM)
            fake_data = generator(z).detach()
            fake_pred = discriminator(fake_data)
            d_loss_fake = criterion(fake_pred, torch.zeros(batch_size, 1))
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_d.step()

            #generator loves to be learnt
            opt_g.zero_grad()
            
            z = torch.randn(batch_size, LATENT_DIM)
            gen_data = generator(z)
            g_loss_adv = criterion(discriminator(gen_data), torch.ones(batch_size, 1))
            g_loss_mse = mse_loss(gen_data, real_data)  
            g_loss = g_loss_adv + 0.5 * g_loss_mse 
            
            g_loss.backward()
            opt_g.step()