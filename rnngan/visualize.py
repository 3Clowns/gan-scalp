import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from constants import DEVICE, WINDOW_SIZE, INIT_PRICE, LATENT_DIM, FEATURES
import os

save_dir: str = "plots_rnngan"

@torch.no_grad()
def generate_samples(generator, seq_len: int = None, n_samples: int = 1) -> pd.DataFrame:
    generator.eval()
    seq_len = seq_len or WINDOW_SIZE
    
    z = torch.randn(n_samples, seq_len, LATENT_DIM).to(DEVICE)
    
    samples = generator(z).cpu().numpy()
    
    return pd.DataFrame(
        samples.reshape(-1, len(FEATURES)),
        columns=FEATURES
    )

'''@torch.no_grad()
def denormalize_features(features: pd.DataFrame, scalers: dict, initial_value: float) -> pd.DataFrame:
    def safe_exp(x):
        return np.exp(np.clip(x, -700, 700))

    denorm = pd.DataFrame(index=features.index)
    
    for feature in FEATURES:
        if feature in scalers:
            denorm[feature] = scalers[feature].inverse_transform(features[[feature]])
            
            if feature in ['lro', 'lrc', 'lrh', 'lrl']:
                denorm[feature] = np.clip(denorm[feature], -7, 7)
            elif feature == 'log_volume':
                denorm[feature] = np.clip(denorm[feature], -20, 20)

    initial_value = max(1e-8, initial_value)
    restored = pd.DataFrame(index=features.index)
    restored.loc[features.index[0], ['open', 'close', 'high', 'low']] = initial_value
    
    for i in range(1, len(features)):
        prev_idx = features.index[i-1]
        curr_idx = features.index[i]
        
        restored.at[curr_idx, 'open'] = np.clip(
            restored.at[prev_idx, 'open'] * safe_exp(denorm['lro'].iloc[i]),
            1e-8, 1e30
        )
        restored.at[curr_idx, 'close'] = np.clip(
            restored.at[prev_idx, 'close'] * safe_exp(denorm['lrc'].iloc[i]),
            1e-8, 1e30
        )
        
        current_open = restored.at[curr_idx, 'open']
        current_close = restored.at[curr_idx, 'close']
        
        temp_high = np.clip(
            restored.at[prev_idx, 'high'] * safe_exp(denorm['lrh'].iloc[i]),
            0.5 * min(current_open, current_close),
            2.0 * max(current_open, current_close)
        )
        
        temp_low = np.clip(
            restored.at[prev_idx, 'low'] * safe_exp(denorm['lrl'].iloc[i]),
            0.5 * min(current_open, current_close),
            1.0 * min(current_open, current_close)
        )
        
        restored.at[curr_idx, 'high'] = max(
            temp_high,
            max(current_open, current_close)
        )
        restored.at[curr_idx, 'low'] = min(
            temp_low,
            min(current_open, current_close)
        )

    restored['high'] = np.clip(restored['high'], 
        restored[['open', 'close']].min(axis=1), 
        restored[['open', 'close']].max(axis=1) * 2.0
    )
    
    restored['low'] = np.clip(restored['low'],
        restored[['open', 'close']].min(axis=1) * 0.5,
        restored[['open', 'close']].min(axis=1)
    )
    
    restored['volume'] = np.clip(
        safe_exp(denorm['log_volume']) - 1e-8,
        0.0,
        1e30
    )
    
    spread_mask = (restored['high'] - restored['low']) > (restored['high'] * 0.5)
    restored.loc[spread_mask, 'high'] = restored.loc[spread_mask, 'low'] * 1.5
    
    return restored[['open', 'high', 'low', 'close', 'volume']].round(6)'''

@torch.no_grad()
def denormalize_features(features: pd.DataFrame, scalers: dict, initial_low: float) -> pd.DataFrame:
    def safe_exp(x):
        return np.exp(np.clip(x, -700, 700))

    denorm = pd.DataFrame()
    params = ['lrl', 'log_dh', 'delta_c', 'delta_o', 'log_volume']
    
    PRICE_MAX = 16000.0
    VOLUME_MAX = 600000.0
    
    for param in params:
        denorm[param] = scalers[param].inverse_transform(features[[param]].values.reshape(-1, 1)).flatten()
        
        '''if param in ['delta_c', 'delta_o']:
            denorm[param] = np.clip(denorm[param], 0.0, 1.0)'''

    restored = pd.DataFrame(index=features.index,
                          columns=['open', 'high', 'low', 'close', 'volume'])
    

    assert not np.isinf(denorm['log_dh']).any(), "Обнаружены Inf в log_dh!"
    assert not np.isinf(denorm['delta_c']).any(), "Обнаружены Inf в delta_c!"
    assert not np.isinf(denorm['lrl']).any(), "Обнаружены Inf в lrl!"
    assert not np.isinf(denorm['delta_o']).any(), "Обнаружены Inf в delta_o!"
    assert not np.isinf(denorm['log_volume']).any(), "Обнаружены Inf в log_volume!"

    assert not np.isnan(denorm['log_dh']).any(), "Обнаружены NaN в log_dh!"
    assert not np.isnan(denorm['delta_c']).any(), "Обнаружены NaN в delta_c!"
    assert not np.isnan(denorm['lrl']).any(), "Обнаружены NaN в lrl!"
    assert not np.isnan(denorm['delta_o']).any(), "Обнаружены NaN в delta_o!"
    assert not np.isnan(denorm['log_volume']).any(), "Обнаружены NaN в log_volume!"
    
    restored.iloc[0] = {
        'low': np.clip(initial_low, 0.0, PRICE_MAX),
        'high': np.clip(initial_low + safe_exp(denorm['log_dh'][0]), 0.0, PRICE_MAX),
        'close': np.clip(initial_low + denorm['delta_c'][0] * safe_exp(denorm['log_dh'][0]), 0.0, PRICE_MAX),
        'open': np.clip(initial_low + denorm['delta_o'][0] * safe_exp(denorm['log_dh'][0]), 0.0, PRICE_MAX),
        'volume': np.clip(safe_exp(denorm['log_volume'][0]), 0.0, VOLUME_MAX)
    }
    
    for i in range(1, len(features)):
        prev_low = restored.at[i-1, 'low']
        
        lrl = np.clip(denorm['lrl'][i], -0.35, 0.35)
        delta_h = np.clip(safe_exp(denorm['log_dh'][i]), 1e-8, PRICE_MAX)
        
        low = np.clip(prev_low * safe_exp(lrl), 0.0, PRICE_MAX)
        
        close = np.clip(low + denorm['delta_c'][i] * delta_h, 0.0, PRICE_MAX)
        open_ = np.clip(low + denorm['delta_o'][i] * delta_h, 0.0, PRICE_MAX)
        high = np.clip(low + delta_h, 0.0, PRICE_MAX)
        
        restored.at[i, 'low'] = low
        restored.at[i, 'high'] = np.maximum(high, np.maximum(open_, close))
        restored.at[i, 'close'] = close
        restored.at[i, 'open'] = open_
        restored.at[i, 'volume'] = np.clip(safe_exp(denorm['log_volume'][i]), 0.0, VOLUME_MAX)

        if restored.at[i, 'high'] < restored.at[i, 'close'] or restored.at[i, 'high'] < restored.at[i, 'open']:
            print("MISTAKE_HIGH")
            print(restored.at[i, 'high'], restored.at[i, 'close'],restored.at[i, 'open'])
        if restored.at[i, 'low'] > restored.at[i, 'close'] or restored.at[i, 'low'] > restored.at[i, 'open']:
            print("MISTAKE_LOW")
            print(restored.at[i, 'low'], restored.at[i, 'close'],restored.at[i, 'open'])

    restored['high'] = np.clip(restored['high'], 0.0, PRICE_MAX)
    restored['low'] = np.clip(restored['low'], 0.0, PRICE_MAX)
    restored['close'] = np.clip(restored['close'], 0.0, PRICE_MAX)
    restored['open'] = np.clip(restored['open'], 0.0, PRICE_MAX)
    
    return restored[['open', 'high', 'low', 'close', 'volume']].round(6)

@torch.no_grad()
def plot_gan(generator, generator_losses: list[float], discriminator_losses: list[float], epoch: int, df_real: pd.DataFrame, scalers: dict):

    n_samples = len(df_real) // WINDOW_SIZE
    df_fake_norm = generate_samples(generator, seq_len=WINDOW_SIZE, n_samples=n_samples)
    
    df_fake = denormalize_features(df_fake_norm, scalers, INIT_PRICE)

    os.makedirs(save_dir, exist_ok=True)
    # Real data
    plt.figure(figsize=(20, 12))
    
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    for i, feature in enumerate(price_cols, 1):
        plt.subplot(3, 2, i)
        
        real_sample = df_real[feature].values 
        fake_sample = df_fake[feature].values
        
        plt.plot(real_sample, label='Real', alpha=0.8, linewidth=1.5)
        plt.plot(fake_sample, label='Generated', alpha=0.8, linestyle='--')
        
        plt.title(f'Feature: {feature}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch{epoch}_real_data.png", dpi=200)
    plt.close()

    # Kernel Density Estimate
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(['open', 'high', 'low', 'close'], 1):
        plt.subplot(3, 2, i)
        sns.kdeplot(df_real[feature], label='Real', fill=True, alpha=0.5)
        sns.kdeplot(df_fake[feature], label='Fake', fill=True, alpha=0.5)
        plt.title(f'{feature} distribution')
        plt.legend()
        plt.xlim(0, 8500)
    plt.subplot(3, 2, 5)
    sns.kdeplot(df_real['volume'], label='Real', fill=True, alpha=0.5)
    sns.kdeplot(df_fake['volume'], label='Fake', fill=True, alpha=0.5)
    plt.title(f'{feature} distribution')
    plt.legend()
    plt.xlim(0, 20000)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch{epoch}_norm_kde.png", dpi=200)
    plt.close()
    
    # Сorrelation matrices
    '''numerical_df = df_real.select_dtypes(include=[np.number])
    corr_real = numerical_df.corr()
    corr_fake = df_fake.corr()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(corr_real, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Real Data Correlations')

    plt.subplot(1, 2, 2)
    sns.heatmap(corr_fake, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Fake Data Correlations')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch{epoch}_corr.png", dpi=200)
    plt.close()'''

    stats = {
        'epoch': epoch,
        'gen_loss_last': generator_losses[-1],
        'disc_loss_last': discriminator_losses[-1],
        'gen_loss_mean': np.mean(generator_losses),
        'gen_loss_var': np.var(generator_losses),
        'disc_loss_mean': np.mean(discriminator_losses),
        'disc_loss_var': np.var(discriminator_losses)
    }

    stats_file = f"{save_dir}/training_stats.csv"
    if not os.path.exists(stats_file):
        pd.DataFrame([stats]).to_csv(stats_file, index=False)
    else:
        pd.DataFrame([stats]).to_csv(stats_file, mode='a', header=False, index=False)

    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(generator_losses, color='blue', label='Generator Loss')
    plt.title('Generator Loss Dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(discriminator_losses, color='red', label='Discriminator Loss')
    plt.title('Discriminator Loss Dynamics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch{epoch}_losses.png", dpi=150, bbox_inches='tight')
    plt.close()