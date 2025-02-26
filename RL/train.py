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


def train_rnn_rl(df_train, df_val, ticker, window_size, alpha, total_timesteps, eta,
                 eval_freq, lr, n_epochs, batch_size, n_steps, 
                 action_reward, wrong_action_reward, scale_reward, lstm_layers, hidden_size,
                 features_dim, ent_coef, device, clip_range, gae_lambda, logging_callback):
    
    policy_kwargs = dict(
        features_extractor_class=CustomRNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,  
            window_size=window_size,
            n_features=12,   
            hidden_size=hidden_size,   
            num_lstm_layers=lstm_layers,
        ),
        net_arch=[dict(pi=[features_dim, 128], vf=[features_dim, 256])],
        normalize_images=False,

    )

    env = create_masked_env(df=df_train, window_size=window_size, alpha=alpha, eta=eta, 
                        action_reward=action_reward, wrong_action_reward=wrong_action_reward, 
                        scale_reward=scale_reward)
    
    # vec_env = DummyVecEnv([lambda: env])

    print("Vector enviroment created successfully")

    model = MaskablePPO("MlpPolicy", env, verbose=1, n_steps=n_steps, n_epochs=n_epochs,
                learning_rate=lr, batch_size=batch_size, policy_kwargs=policy_kwargs,
                gae_lambda=gae_lambda, ent_coef=ent_coef, device=device, clip_range=clip_range)
    
    print("Model created")
    

    validation_callback = ValidationCallback(df_train, df_val, eval_freq=eval_freq,
                                          window_size=window_size, alpha=0,
                                          action_reward=action_reward,
                                          war=wrong_action_reward, vc=logging_callback,
                                          eta=eta, verbose=1, scale_reward=scale_reward)
    
    print("Started train loop")
    
    model.learn(total_timesteps=total_timesteps, callback=validation_callback)
    
    return model

class CustomRNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int, 
                 window_size: int, n_features: int, hidden_size: int, num_lstm_layers: int):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout = 0.1,
        )
        
        self.extra_features = 9  # position, row, reward/profit, entry_price, mask
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + self.extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        time_series = observations[:, :self.window_size * self.n_features].view(
            batch_size, self.window_size, self.n_features
        )
        extra_features = observations[:, -self.extra_features:]
        
        lstm_out, _ = self.lstm(time_series)
        lstm_last = lstm_out[:, -1, :]  
        
        combined = th.cat([lstm_last, extra_features], dim=1)
        
        features = self.fc(combined)
        
        return features