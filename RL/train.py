from envs import MoexTradingEnv
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import prepare_data, time_split
from eval import ValidationCallback
from eval import evaluate_agent

import gym
import torch as th
import torch.nn as nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def train_vanilla_rl(df_train, df_val, ticker, window_size=10, alpha=0.0001, total_timesteps=10000, eta=100,
                     eval_freq=1000, lr=0.0003, n_epochs=10, batch_size=64, n_steps=2048, policy=None, action_reward=0.2, wrong_action_reward=-0.2, scale_reward=10, logging_callback=None):
    

    config = {
        "ticker": ticker,
        "window_size": window_size,
        "alpha": alpha,
        "total_timesteps": total_timesteps,
        "learning_rate": lr,
        "eta": eta,
        "eval_freq": eval_freq,
        "n_epochs" : n_epochs,
        "action_reward" : action_reward,
        "wrong_action_reward" : wrong_action_reward,
        "scale_reward" : scale_reward,
    }
    
    logging_callback(config)
    # wandb.log(config)

    # df = prepare_data(data_dict, ticker)
    # df_train, df_val, df_test = time_split(df, train_ratio=0.7, val_ratio=0.2)

    env = MoexTradingEnv(df_train, window_size=window_size, alpha=alpha, eta=eta, action_reward=action_reward, wrong_action_reward=wrong_action_reward, scale_reward=scale_reward)
    vec_env = DummyVecEnv([lambda: env])

    if policy is not None:
        model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=n_steps, n_epochs=n_epochs, learning_rate=lr, batch_size=batch_size, policy_kwargs=policy, gae_lambda=0.95, ent_coef=0.01, device="cpu", clip_range_vf=0.2)

    validation_callback = ValidationCallback(df_train, df_val, eval_freq=eval_freq, window_size=window_size, alpha=0, action_reward=action_reward, war=wrong_action_reward, lc = logging_callback)
    model.learn(total_timesteps=total_timesteps, callback=validation_callback)

    return model


import itertools
from stable_baselines3 import PPO

def grid_search_ppo(df_train, df_val, ticker):
    wandb.init(project="RL")


    param_grid = {
        'window_size': [10, 20, 30],
        'learning_rate': [1e-5, 3e-5, 1e-4],
        'n_epochs': [5, 10, 15],
        'batch_size': [64, 128, 256],
        'n_steps': [256, 512, 1024],
        'action_reward': [0.01, 0.1, 0.2],
        'wrong_action_reward': [-50, -100, -200],
        'policy_architectures': [
            dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))
        ]
    }

    # Фиксированные параметры
    fixed_params = {
        'total_timesteps': 500000,
        'eta': 0,
        'alpha': 0.0
    }

    # Создаем все возможные комбинации параметров
    keys = param_grid.keys()
    combinations = itertools.product(*param_grid.values())
    
    best_profit = float('-inf')
    best_params = None

    # Перебираем все комбинации
    for combo in combinations:
        current_params = dict(zip(keys, combo))
        
        try:
            # Обучаем модель с текущими параметрами
            model = train_vanilla_rl(
                df_train=df_train,
                df_val=df_val,
                ticker=ticker,
                window_size=current_params['window_size'],
                lr=current_params['learning_rate'],
                n_epochs=current_params['n_epochs'],
                batch_size=current_params['batch_size'],
                n_steps=current_params['n_steps'],
                policy=current_params['policy_architectures'],
                action_reward=current_params['action_reward'],
                wrong_action_reward=current_params['wrong_action_reward'],
                **fixed_params
            )

            # Оцениваем модель
            current_profit = evaluate_agent(model, df_val, max_steps=len(df_val), alpha=0, eta=0, action_reward=0, war=-0.5)

            # Логируем результаты
            wandb.log({
                "current_profit": current_profit,
                **current_params
            })

            # Обновляем лучшие параметры
            if current_profit > best_profit:
                best_profit = current_profit
                best_params = current_params

        except Exception as e:
            print(f"Error with parameters {current_params}: {str(e)}")
            continue

    # Логируем лучшие параметры
    wandb.log({
        "best_profit": best_profit,
        "best_params": best_params
    })

    return best_params, best_profit


def train_rnn_rl(df_train, df_val, ticker, window_size=10, alpha=0.0001, total_timesteps=10000, eta=100,
                 eval_freq=1000, lr=0.0003, n_epochs=10, batch_size=64, n_steps=2048, 
                 action_reward=0.2, wrong_action_reward=-0.2, scale_reward=10, lstm_layers=1, hidden_size=128,
                 features_dim=64, ent_coef=0.1, device=None, logging_callback=None):
    
    policy_kwargs = dict(
        features_extractor_class=CustomRNNFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,  # Размерность выходного эмбеддинга
            window_size=window_size,
            n_features=12,    # Количество признаков для каждого временного шага
            hidden_size=hidden_size,   # Размер скрытого слоя RNN
            num_lstm_layers=lstm_layers,
        ),
        net_arch=[dict(pi=[features_dim, 32], vf=[features_dim, 32])]  # Архитектура policy и value networks
    )

    env = MoexTradingEnv(df_train, window_size=window_size, alpha=alpha, eta=eta, 
                        action_reward=action_reward, wrong_action_reward=wrong_action_reward, 
                        scale_reward=scale_reward)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=n_steps, n_epochs=n_epochs,
                learning_rate=lr, batch_size=batch_size, policy_kwargs=policy_kwargs,
                gae_lambda=0.95, ent_coef=ent_coef, device=device, clip_range_vf=0.2)
    
    
    validation_callback = ValidationCallback(df_train, df_val, eval_freq=eval_freq,
                                          window_size=window_size, alpha=0,
                                          action_reward=action_reward,
                                          war=wrong_action_reward, vc=logging_callback)
    
    model.learn(total_timesteps=total_timesteps, callback=validation_callback)
    
    return model

class CustomRNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int, 
                 window_size: int, n_features: int, hidden_size: int, num_lstm_layers: int):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Дополнительные слои для обработки дополнительных признаков
        self.extra_features = 4  # position, row, reward/profit, entry_price
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + self.extra_features, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        # Разделяем временной ряд и дополнительные признаки
        time_series = observations[:, :self.window_size * self.n_features].view(
            batch_size, self.window_size, self.n_features
        )
        extra_features = observations[:, -self.extra_features:]
        
        # Пропускаем временной ряд через LSTM
        lstm_out, _ = self.lstm(time_series)
        lstm_last = lstm_out[:, -1, :]  # Берем последний выход LSTM
        
        # Конкатенируем с дополнительными признаками
        combined = th.cat([lstm_last, extra_features], dim=1)
        
        # Финальное преобразование
        features = self.fc(combined)
        
        return features