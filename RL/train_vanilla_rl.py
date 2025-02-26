from envs import MoexTradingEnv
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import prepare_data, time_split
from eval import ValidationCallback
from eval import evaluate_agent

import torch as th
import torch.nn as nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

import itertools


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

    env = MoexTradingEnv(df_train, window_size=window_size, alpha=alpha, 
                         eta=eta, action_reward=action_reward, 
                         wrong_action_reward=wrong_action_reward, 
                         scale_reward=scale_reward)
    
    env = ActionMasker(env, mask_fn=lambda env: env.get_action_mask())
    
    vec_env = DummyVecEnv([lambda: env])

    if policy is not None:
        model = MaskablePPO("MlpPolicy", vec_env, verbose=1, n_steps=n_steps, n_epochs=n_epochs, learning_rate=lr, batch_size=batch_size, policy_kwargs=policy, gae_lambda=0.95, ent_coef=0.01, device="cpu", clip_range_vf=0.2)

    validation_callback = ValidationCallback(df_train, df_val, eval_freq=eval_freq, window_size=window_size, alpha=0, action_reward=action_reward, war=wrong_action_reward, lc = logging_callback, verbose=1, eta=0, scale_reward=scale_reward)
    model.learn(total_timesteps=total_timesteps, callback=validation_callback)

    return model
