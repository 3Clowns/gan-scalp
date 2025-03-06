import torch
import torch.nn as nn
import time
import wandb
from utils.constants import LATENT_DIM, DEVICE, WINDOW_SIZE
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
        generator_losses = []
        discriminator_losses = []
        for real_data in dataloader:
            start = time.time()
            real_data.to(DEVICE)
            batch_size = real_data.size(0)
            
            z = torch.randn(batch_size, LATENT_DIM, WINDOW_SIZE).to(DEVICE)
            with torch.no_grad():
                fake_data = generator(z)
            real_labels = torch.ones(real_data.shape[0]).to(DEVICE)
            fake_labels = torch.zeros(real_data.shape[0]).to(DEVICE)

            # Train the discriminator
            opt_d.zero_grad()
            # Compute discriminator loss on real samples
            real_loss = criterion(discriminator(real_data), real_labels)
            # Compute discriminator loss on fake samples
            fake_loss = criterion(discriminator(fake_data), fake_labels)
            # Compute the total discriminator loss
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            opt_d.step()
            
            # Train the generator
            opt_g.zero_grad()
            # Generate fake samples and compute generator loss
            fake_data = generator(z)
            print(discriminator(fake_data).min(), discriminator(fake_data).max())
            generator_loss = criterion(discriminator(fake_data), real_labels)
            generator_loss.backward()
            opt_g.step()
            wandb.log({
                "d_loss_batch": discriminator_loss.item(),
                "g_loss_batch": generator_loss.item(),

            })
            discriminator_losses.append(discriminator_loss.item())
            generator_losses.append(generator_loss.item())
            print(f'Batch done in {time.time() - start} seconds')
        wandb.log({
                "d_loss_epoch": sum(discriminator_losses)/len(discriminator_losses),
                "g_loss_epoch": sum(generator_losses)/len(generator_losses),
            })