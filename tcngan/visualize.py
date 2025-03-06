import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from constants import DEVICE, WINDOW_SIZE, N_ASSETS, LATENT_DIM, BATCH_SIZE

def get_shifted_noise(
    n_samples: int,
    batch_size: int = BATCH_SIZE,
    latent_dim: int = LATENT_DIM,
    window_size: int = WINDOW_SIZE,
    device: str = DEVICE
) -> torch.Tensor:
    """
    Генерирует шум с перекрывающимися окнами для временной корреляции.
    
    Args:
        batch_size: Количество примеров в батче.
        latent_dim: Размерность латентного пространства (число активов).
        window_size: Длина временного окна.
        device: Устройство для тензора (CPU/GPU).
        
    Returns:
        Тензор шума размерности (batch_size, latent_dim, window_size).
    """
    if batch_size == 1:
        return torch.randn(batch_size, latent_dim, window_size, device=device)
    
    base_noise = torch.randn(latent_dim, window_size + batch_size - 1, device=device)
    
    shifted_noise = torch.zeros((batch_size, latent_dim, window_size), device=device)
    for i in range(batch_size):
        shifted_noise[i] = base_noise[:, i:i + window_size]
    
    return shifted_noise

@torch.no_grad()
def generate_samples(generator) -> pd.DataFrame:
    """
    Generate random samples from generator
    """
    generator.eval()
    z = get_shifted_noise(n_samples=1).to(DEVICE)
    samples = generator(z).cpu()
    
    samples = samples[0].permute(1, 0)
    print(samples.shape)
    return pd.DataFrame(samples.numpy(), columns=['log_return'])

@torch.no_grad()
def plot_gan(generator, generator_losses: list[float], discriminator_losses: list[float], epoch: int, df_returns_real: pd.DataFrame):
    """
    Print statistics
    Plot distribution
    Plot cumulative returns
    """
    plot_column = "log_return" 
    print("SMTH")
    df_returns_fake = generate_samples(generator)

    # Исправленный вывод стандартных отклонений
    print(f'Fake std: {df_returns_fake[plot_column].std():.4f}')
    print(f'Real std: {df_returns_real[plot_column].std():.4f}')

    
    # Plot returns distribution
    plt.figure(figsize=(10, 5))
    
    # График распределения
    plt.subplot(1, 2, 1)
    sns.histplot(df_returns_real[plot_column], stat='density', label='real')
    sns.histplot(df_returns_fake[plot_column], stat='density', label='fake')
    
    # Границы значений
    plt.axvline(df_returns_real[plot_column].min(), linestyle='dashed', color='C0')
    plt.axvline(df_returns_real[plot_column].max(), linestyle='dashed', color='C0')
    plt.axvline(df_returns_fake[plot_column].min(), linestyle='dashed', color='C1')
    plt.axvline(df_returns_fake[plot_column].max(), linestyle='dashed', color='C1')
    
    plt.legend()
    plt.title(f'Distribution ({epoch} epoch)')

    # Plot cumulative returns
    plt.subplot(1, 2, 2)
    df_returns_real.iloc[:WINDOW_SIZE].cumsum()[plot_column].plot(label='Real')
    df_returns_fake.set_index(df_returns_real.index[:WINDOW_SIZE]).cumsum()[plot_column].plot(label='Fake')
    plt.title('Cumulative returns')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch+1), generator_losses)
    plt.title('Generator Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch+1), discriminator_losses)
    plt.title('Discriminator Loss')
    
    plt.tight_layout()
    plt.show()