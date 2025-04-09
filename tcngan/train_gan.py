import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer
from IPython.display import clear_output
from visualize import plot_gan
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from constants import LATENT_DIM, DEVICE, WINDOW_SIZE, LAMBDA, TRAIN_G, TRAIN_D, INIT_PRICE

SAVE_PATH = Path('models/')
SAVE_PATH.mkdir(exist_ok=True)
        
def train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader) -> tuple[float, float]:
    generator.train()
    discriminator.train()

    scaler_G = GradScaler()
    scaler_D = GradScaler()
    
    generator_losses = []
    discriminator_losses = []
    
    criterion = nn.BCELoss()

    for idx, real_samples in enumerate(tqdm(dataloader, desc = "Train", leave = False)):
        
        real_samples = real_samples.to(DEVICE)
        real_labels = torch.ones(real_samples.shape[0]).to(DEVICE)
        fake_labels = torch.zeros(real_samples.shape[0]).to(DEVICE)
        
        z = torch.randn(real_samples.shape[0], LATENT_DIM, WINDOW_SIZE).to(DEVICE)

        with torch.no_grad():
            raw_fake = generator(z)

        # Discriminator
        for _ in range(TRAIN_D):
            discriminator_optimizer.zero_grad()
            real_loss = criterion(discriminator(real_samples), real_labels)
            fake_loss = criterion(discriminator(raw_fake.detach()), fake_labels)
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            discriminator_optimizer.step()       

        # Generator
        for _ in range(TRAIN_G):
            z = torch.randn(real_samples.shape[0], LATENT_DIM, WINDOW_SIZE).to(DEVICE)
            raw_fake = generator(z)
            generator_optimizer.zero_grad()
            generator_loss = criterion(discriminator(raw_fake), real_labels)
            generator_loss.backward()
            #torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            generator_optimizer.step()
            
        discriminator_losses.append(discriminator_loss.item())    
        generator_losses.append(generator_loss.item())
        
    return np.mean(generator_losses), np.mean(discriminator_losses)


def save_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, 
            generator_losses, discriminator_losses, epoch: int, model_prefix: str):
    """
    Save GAN checkpoint with losses history
    """
    model_path = SAVE_PATH / model_prefix
    model_path.mkdir(exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
        'generator_losses': generator_losses,
        'discriminator_losses': discriminator_losses
    }, model_path / f'checkpoint_{epoch}')

def load_gan(model_prefix: str, generator=None, discriminator=None, 
            generator_optimizer=None, discriminator_optimizer=None,
            generator_losses=None, discriminator_losses=None,
            epoch: int | None = None):
    """
    Load GAN checkpoint with losses history
    Returns: (epoch, generator_losses, discriminator_losses)
    """
    model_path = SAVE_PATH / model_prefix
    assert model_path.exists()
    
    if epoch is None:
        files = list(model_path.iterdir())
        epochs = [int(f.name.split('_')[-1]) for f in files if f.name.startswith('checkpoint_')]
        epoch = max(epochs) if epochs else 0

    checkpoint = torch.load(model_path / f'checkpoint_{epoch}', map_location='cpu')
    
    # Load models
    if generator and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
    if discriminator and 'discriminator_state_dict' in checkpoint:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator.eval()

    # Load optimizers
    if generator_optimizer and 'generator_optimizer_state_dict' in checkpoint:
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    if discriminator_optimizer and 'discriminator_optimizer_state_dict' in checkpoint:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

    # Load losses history with backward compatibility
    generator_losses = checkpoint.get('generator_losses', [])
    discriminator_losses = checkpoint.get('discriminator_losses', [])

def train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, df_real: pd.DataFrame, 
            scalers, n_epochs: int, plot_frequency: int, save_frequency: int, model_prefix: str) -> tuple[list[float], list[float]]:
    torch.manual_seed(1)
    start = timer()

    generator_losses = []
    discriminator_losses = []

    for epoch in range(1, n_epochs + 1):
        print(f"\n--- Starting epoch {epoch}/{n_epochs} ---")
        generator_loss, discriminator_loss = train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader)

        generator_losses.append(generator_loss)
        discriminator_losses.append(discriminator_loss)

        # Plot samples
        if epoch % plot_frequency == 0 or epoch == n_epochs:
            clear_output(wait=True)
            train_time = timer() - start
            print(f'{plot_frequency} epochs train time: {train_time:.1f}s. Estimated train time: {((n_epochs - epoch) * train_time / plot_frequency / 60):.1f}m')
            start = timer()
            plot_gan(generator, generator_losses, discriminator_losses, epoch, df_real, scalers)

        # Save model
        if epoch % save_frequency == 0 or epoch == n_epochs:
            save_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, generator_losses, discriminator_losses, epoch, model_prefix)

    return generator_losses, discriminator_losses