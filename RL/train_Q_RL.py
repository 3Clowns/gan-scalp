from stable_baselines3 import DQN
from eval import ValidationCallback
from typing import Callable
from envs import create_masked_env
from train import CustomRNNFeatureExtractor


def train_dqn_rl(df_train, df_val, ticker, window_size, alpha, total_timesteps, eta,
                 eval_freq, lr, batch_size, action_reward, wrong_action_reward, 
                 scale_reward, lstm_layers, hidden_size, features_dim, device, 
                 logging_callback: Callable, exploration_params: dict = None) -> DQN:
    """
    DQN training function with RNN support and custom feature extraction
    
    Args:
        exploration_params: Dictionary with exploration settings
            exploration_final_eps (float): Final exploration epsilon
            exploration_fraction (float): Fraction of training with exploration
            target_update_interval (int): Steps between target network updates
    """
    # Policy configuration with RNN
    policy_kwargs = {
        "features_extractor_class": CustomRNNFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "window_size": window_size,
            "n_features": df_train[ticker].shape[1],
            "hidden_size": hidden_size,
            "num_lstm_layers": lstm_layers
        },
        "net_arch": [256, 128],  # Q-network architecture
        "normalize_images": False,
    }

    # Environment creation
    env = create_masked_env(
        df=df_train[ticker],
        window_size=window_size,
        alpha=alpha,
        eta=eta,
        action_reward=action_reward,
        wrong_action_reward=wrong_action_reward,
        scale_reward=scale_reward
    )

    # Default exploration parameters
    if not exploration_params:
        exploration_params = {
            "exploration_final_eps": 0.01,
            "exploration_fraction": 0.2,
            "target_update_interval": 1000
        }

    # Model initialization
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr,
        policy_kwargs=policy_kwargs,
        batch_size=batch_size,
        device=device,
        buffer_size=100_000,
        learning_starts=10_000,
        gamma=0.99,
        **exploration_params
    )

    # Callbacks setup
    validation_callback = ValidationCallback(
        df_train=df_train[ticker],
        df_val=df_val[ticker],
        eval_freq=eval_freq,
        window_size=window_size,
        alpha=alpha,
        action_reward=action_reward,
        war=wrong_action_reward,
        vc=logging_callback,
        eta=eta,
        verbose=1,
        scale_reward=scale_reward
    )

    # Training
    model.learn(
        total_timesteps=total_timesteps,
        callback=validation_callback,
        progress_bar=True
    )

    return model