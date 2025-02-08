import os
import sys
import apimoex
import pandas as pd
import requests
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, kstest
import gym
from gym import spaces
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from train import train_vanilla_rl
from eval import evaluate_agent

from create_data import create_dataset, save_dataset, load_dataset
from utils import time_split, prepare_data
from train import grid_search_ppo

wandb.init(project='RL')


def main():
    filename = "data.pkl"
    directory_path = "./"
    path = os.path.join(directory_path, filename)

    if os.path.exists(path):
        data = load_dataset(filename)
    else:
        data = create_dataset()
        save_dataset(data, path)

    

    total_timestamps = 500000
    eta = 0
    n_steps=512



    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    data = prepare_data(data)
    df_train, df_val, df_test = time_split(data)
    ticker = "VTBR"

    grid_search_ppo(df_train=df_train[ticker], df_val=df_val[ticker], ticker=ticker)

    #model = train_vanilla_rl(df_train[ticker], df_val[ticker], ticker, window_size=20,  alpha=0.0, total_timesteps=total_timestamps, eta=eta, eval_freq=n_steps,
    #                         lr=3e-5, n_epochs=5, batch_size=2, n_steps=n_steps, policy=policy_kwargs, action_reward=0.01, wrong_action_reward=-100)
    
    #profit = evaluate_agent(model, df_val[ticker], max_steps=len(df_val), alpha=0)

    #wandb.log({"Total profit": profit})
    wandb.finish()


if __name__ == "__main__":
    main()
