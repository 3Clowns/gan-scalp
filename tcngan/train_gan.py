import torch
import torch.nn as nn
from constants import LATENT_DIM, DEVICE, WINDOW_SIZE
import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer
from IPython.display import clear_output
from visualize import plot_gan
from tqdm import tqdm
from torch.amp import GradScaler, autocast

SAVE_PATH = Path('models/')
SAVE_PATH.mkdir(exist_ok=True)
            
        
def train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader) -> tuple[float, float]:
    generator.train()
    discriminator.train()

    generator_losses = []
    discriminator_losses = []
    
    criterion = nn.BCELoss()

    for real_samples in tqdm(dataloader, desc = "Train", leave = True):
        
        real_samples = real_samples.to(DEVICE)
            
        z = torch.randn(real_samples.size(0), LATENT_DIM, WINDOW_SIZE).to(DEVICE)
        
        with torch.no_grad():
            fake_samples = generator(z)
        real_labels = torch.ones(real_samples.shape[0]).to(DEVICE)
        fake_labels = torch.zeros(real_samples.shape[0]).to(DEVICE)

        # Train the discriminator
        discriminator_optimizer.zero_grad()
        real_loss = criterion(discriminator(real_samples), real_labels)
        fake_loss = criterion(discriminator(fake_samples.detach()), fake_labels)
        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()
        generator_loss = criterion(discriminator(fake_samples), real_labels)
        generator_loss.backward()
        generator_optimizer.step()
        
        discriminator_losses.append(discriminator_loss.item())
        generator_losses.append(generator_loss.item())
    
    return np.mean(generator_losses), np.mean(discriminator_losses)


def save_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch: int, model_prefix: str):
    """
    Save GAN checkpoint
    """
    model_path = SAVE_PATH / model_prefix
    model_path.mkdir(exist_ok=True)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
    }, model_path / f'checkpoint_{epoch}')


def load_gan(model_prefix: str, generator=None, discriminator=None, generator_optimizer=None, discriminator_optimizer=None, epoch: int | None = None):
    """
    Load GAN checkpoint
    Load only models that are not None
    Load latest epoch if not specified
    """
    model_path = SAVE_PATH / model_prefix
    assert model_path.exists()
    if epoch is None:
        # Find latest checkpoint
        files = list(model_path.iterdir())
        assert len(files) > 0
        for file in files:
            assert file.name.startswith('checkpoint_')
        epochs = [int(file.name.removeprefix('checkpoint_')) for file in files]
        epoch = max(epochs)

    print(f'Load {epoch} epoch checkpoint')
    checkpoint = torch.load(model_path / f'checkpoint_{epoch}')
    assert checkpoint['epoch'] == epoch

    # Load models
    if generator is not None:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
    if discriminator is not None:
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        discriminator.eval()

    # Load optimizers
    if generator_optimizer is not None:
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    if discriminator_optimizer is not None:
        discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])


def train_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader, df_returns_real: pd.DataFrame, n_epochs: int, plot_frequency: int, save_frequency: int, model_prefix: str) -> tuple[list[float], list[float]]:
    torch.manual_seed(1)
    start = timer()

    generator_losses = []
    discriminator_losses = []

    for epoch in range(1, n_epochs + 1):
        print("Start epoch...")
        generator_loss, discriminator_loss = train_epoch(generator, discriminator, generator_optimizer, discriminator_optimizer, dataloader)
        print("Finish epoch...")

        generator_losses.append(generator_loss)
        discriminator_losses.append(discriminator_loss)

        # Plot samples
        if epoch % plot_frequency == 0 or epoch == n_epochs:
            print("Start plotting...")
            clear_output(wait=True)
            train_time = timer() - start
            print(f'{plot_frequency} epochs train time: {train_time:.1f}s. Estimated train time: {((n_epochs - epoch) * train_time / plot_frequency / 60):.1f}m')
            start = timer()
            plot_gan(generator, generator_losses, discriminator_losses, epoch, df_returns_real)
            print("Finish plotting...")

        # Save model
        if epoch % save_frequency == 0 or epoch == n_epochs:
            print("Start saving..")
            save_gan(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch, model_prefix)
            print("Finish saving...")

    return generator_losses, discriminator_losses