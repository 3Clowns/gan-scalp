import copy

from click import progressbar
from torch.onnx.symbolic_opset9 import dropout

from envs import MoexTradingEnv, TestingTradingEnv, create_masked_env
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import prepare_data, time_split
from eval import ValidationCallback
from eval import evaluate_agent

import gymnasium
import torch as th
import torch.nn as nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from model_wrapper import RLModelWrapper


def train_rnn_rl(df_train, df_val, ticker, window_size, alpha, total_timesteps, eta,
                 eval_freq, lr, n_epochs, batch_size, n_steps, 
                 action_reward, scale_reward, lstm_layers, hidden_size,
                 features_dim, ent_coef, device, clip_range, gae_lambda, logging_callback, day_penalty) -> RLModelWrapper:
    
    policy_kwargs = dict(
        features_extractor_class=CustomRNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,  
            window_size=window_size,
            #n_features=12,
            n_features=1,
            hidden_size=hidden_size,   
            num_lstm_layers=lstm_layers,
            dropout=0.2,
            use_log_return=False,
        ),
        net_arch=[dict(pi=[features_dim, 256], vf=[features_dim, 256])],
        normalize_images=False,

    )

    train_env_config = {
        "df": df_train,
        "window_size": window_size,
        "alpha": alpha,
        "eta": eta,
        "action_reward": action_reward,
        "scale_reward": scale_reward,
        "day_penalty": day_penalty,
    }

    rl_config = {
        "policy": "MlpPolicy",
        "verbose": 1,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "policy_kwargs": policy_kwargs,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "device": device,
        "clip_range": clip_range,
    }

    val_env_config = copy.deepcopy(train_env_config)
    val_env_config["df"] = df_val
    val_env_config["day_penalty"] = 0
    val_env_config["eta"] = 0

    val_callback_config = {
        "df_train": val_env_config["df"],
        "df_val": train_env_config["df"],
        "eval_freq": eval_freq,
        "window_size": train_env_config["window_size"],
        "alpha": train_env_config["alpha"],
        "action_reward": train_env_config["action_reward"],
        "vc": logging_callback,
        "eta": train_env_config["eta"],
        "verbose": 1,
        "scale_reward": train_env_config["scale_reward"],
    }

    print(train_env_config)

    model_wrapper = RLModelWrapper(
        algo="maskable_ppo",
        train_env=MoexTradingEnv(**train_env_config),
        val_env=MoexTradingEnv(**val_env_config),
        test_env=TestingTradingEnv(**val_env_config),
        config=rl_config,
        val_callback_config=val_callback_config,
        train_env_config=train_env_config,
        val_env_config=val_env_config,
    )

    print("Model wrapper created")

    model_wrapper.train(total_timesteps=total_timesteps, progress_bar=True)
    
    return model_wrapper


class CustomRNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int, 
                 window_size: int, n_features: int, hidden_size: int, num_lstm_layers: int, dropout: float, use_log_return: bool = False,
            close_idx: int = 0,):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.use_log_return = use_log_return
        self.close_idx = close_idx

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout = dropout,

        )
        
        self.extra_features = 10  # position, row, reward/profit, last_minute, entry_price, mask
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + self.extra_features, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]

        time_series = observations[:, :self.window_size * self.n_features]
        time_series = time_series.view(batch_size, self.window_size, self.n_features)

        # print(time_series)

        if self.use_log_return:
            close = time_series[:, :, self.close_idx]  # shape = [B, T]
            close_clamped = close.clamp(min=1e-6)
            ret = (close_clamped[:, 1:] / close_clamped[:, :-1]).log()
            close[:, 1:] = ret
            close[:, 0] = 0.0

        # print(time_series)

        extra_features = observations[:, -self.extra_features:]
        
        lstm_out, _ = self.lstm(time_series)
        lstm_last = lstm_out[:, -1, :]  

        # print(extra_features)

        combined = th.cat([lstm_last, extra_features], dim=1)
        
        features = self.fc(combined)
        
        return features