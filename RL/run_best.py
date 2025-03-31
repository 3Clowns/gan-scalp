
from stable_baselines3 import PPO
from stable_baselines3.common import vec_env
from envs import MoexTradingEnv, TestingTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import os 
from create_data import create_dataset, load_dataset, save_dataset
from utils import time_split, prepare_data
import torch
from save_best import test_agent_with_actions
from sb3_contrib import MaskablePPO
from envs import create_masked_env

filename = "data.pkl"
directory_path = "./"
path = os.path.join(directory_path, filename)

params = {
    "action_reward": 0,
    "wrong_action_reward": -0.01,
    "eta": 0,
    "window_size": 20,
    "scale_reward": 100,
}

if os.path.exists(path):
    data = load_dataset(filename)
else:
    data = create_dataset()
    save_dataset(data, path)

data = prepare_data(data)
df_train, df_val, df_test = time_split(data)
ticker = "VTBR"
n_gpus = torch.cuda.device_count()
gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [None]

env = create_masked_env(df=df_val[ticker], **params)

fixed_params = {
        
        "learning_rate": 1e-4,
        "n_epochs": 3,
        "batch_size": 128,
        "n_steps": 2048,
        #"total_epochs":  6,
        #"hidden_dim_rnn": 128,
        #"lstm_layers": 2,
        #"features_dim": 64,
        "ent_coef": 0.1,
        "gae_lambda" : 0.99,
}

#model = PPO("MlpPolicy", vec_env, verbose=1,)
model = MaskablePPO("MlpPolicy", env, verbose=1, **fixed_params,
                gae_lambda=fixed_params["gae_lambda"])
    
# model = MaskablePPO.load("saved_models/trial_0_gpu_0")
test_agent_with_actions(model, df_test[ticker], max_steps=len(df_val[ticker]),
            window_size=params['window_size'],
            alpha=0, eta=0, action_reward=0, war=0,
            scale_reward=params["scale_reward"],
            max_holding=40,)