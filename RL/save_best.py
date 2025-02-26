import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from train import train_rnn_rl
from train_vanilla_rl import train_vanilla_rl
import os
from create_data import load_dataset, create_dataset, save_dataset
from utils import prepare_data, time_split
from eval import evaluate_agent, test_agent
import torch
import time 
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Queue
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import random
import pandas as pd 
import json 
import wandb
from envs import TestingTradingEnv, MoexTradingEnv, create_masked_env
from eval import test_agent_with_actions

def run_trial(args):
    params, gpu_id, df_train, df_val, ticker = args
    try:
        run_name = f"trial_{params['trial_number']}_gpu_{gpu_id}"
        
        wandb.init(
            project="RL LKOH MaskedPPO",
            reinit=True,
            name=run_name,
            config=params
        )
        
        step_counter = 0
        def custom_log(metrics):
            nonlocal step_counter
            wandb.log(metrics, step=step_counter)
            step_counter += 1
        
        device = torch.device(f'cuda:{gpu_id}')
        if gpu_id is not None:
           torch.cuda.set_device(device)
        
        model = train_rnn_rl(
            df_train=df_train[ticker],
            df_val=df_val[ticker],
            ticker=ticker,
            window_size=params['window_size'],
            lr=params['learning_rate'],
            n_epochs=params['n_epochs'],
            batch_size=params['batch_size'],
            n_steps=params['n_steps'],
            action_reward=params['action_reward'],
            wrong_action_reward=params['wrong_action_reward'],
            total_timesteps=len(df_train[ticker]) * params['total_epochs'],
            eta=params['eta'],
            scale_reward=params['scale_reward'],
            alpha=0.0,
            lstm_layers=params['lstm_layers'],
            hidden_size=params['hidden_dim_rnn'],
            features_dim=params['features_dim'],
            ent_coef=params['ent_coef'],
            gae_lambda=params["gae_lambda"],
            clip_range=params["clip_range"],
            device=device,
            logging_callback=custom_log,
            eval_freq=1000,
        )

        print("Training finished")
        
        profit, _, _, _, _ = evaluate_agent(
            model, df_val[ticker], 
            # max_steps=len(df_val[ticker]), 
            max_steps=len(df_val[ticker]),
            window_size=params['window_size'],
            alpha=0, eta=0, action_reward=0, war=0,
            scale_reward=params["scale_reward"], max_holding=40,
        )

        print(f"Profit evaluate agent: {profit}")
        
        real_profit, _, _, _, _ = test_agent(
            model, df_val[ticker],
            max_steps=len(df_val[ticker]),
            #max_steps=1000,
            window_size=params['window_size'],
            alpha=0, eta=0, action_reward=0, war=0,
            scale_reward=params["scale_reward"], max_holding=40,
        )

        print(f"Profit test agent: {real_profit}")


        test_agent_with_actions(
            model, df_val[ticker], max_steps=len(df_val[ticker]),
            window_size=params['window_size'],
            alpha=0, eta=0, action_reward=0, war=0,
            scale_reward=params["scale_reward"],
            max_holding=40,
        )

        print(f"Testing completed")
        
        custom_log({
            "trial_value": real_profit,
            "Val profit in train env": profit,
            **params
        })

        os.makedirs("saved_models", exist_ok=True)

        model_path = f"saved_models/trial_{params['trial_number']}_gpu_{gpu_id}"
        model.save(model_path)

        wandb.finish()
        return params, real_profit
        
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        return params, float('-inf')



def run_best_hp(df_train, df_val, ticker, n_trials=100, max_concurrent=6, gpu_ids=[0]):
    fixed_params = {
        "action_reward": 0.0,
        "wrong_action_reward": -0.0,
        "window_size": 50,
        "learning_rate": 1e-4,
        "n_epochs": 15,
        "batch_size": 256,
        "n_steps": 4096,
        "scale_reward": 1000,
        "total_epochs":  10,
        "hidden_dim_rnn": 512,
        "lstm_layers": 12,
        "features_dim": 512,
        "ent_coef": 0.01,
        "eta" : 1,
        "clip_range" : 0.2,
        "gae_lambda" : 0.9,
    }
    
    trial_args = []
    for i in range(n_trials):
        params = {}
        params.update(fixed_params)
        params['trial_number'] = i
        gpu_id = gpu_ids[i % len(gpu_ids)]
        trial_args.append((params, gpu_id, df_train, df_val, ticker))
    
    results = []

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(run_trial, args) for args in trial_args]
        
        for future in futures:
            try:
                params, value = future.result()
                results.append((params, value))
                print(f"Trial completed: {params['trial_number']}, Value: {value}")
            except Exception as e:
                print(f"Error processing trial: {str(e)}")
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    results_df = pd.DataFrame([
        {**params, "value": value}
        for params, value in results
    ])
    results_df.to_csv("/place/home/max-tm/everything/RL/gan-scalp/RL/grid_search_results.csv", index=False)
    
    best_params, best_value = results[0]
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    
    return best_params, best_value, results

if __name__ == "__main__":
    filename = "data.pkl"
    directory_path = "./"
    path = os.path.join(directory_path, filename)

    if os.path.exists(path):
        data = load_dataset(filename)
    else:
        data = create_dataset()
        save_dataset(data, path)

    data = prepare_data(data)
    df_train, df_val, df_test = time_split(data)
    ticker = "LKOH"
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [None]

    print(gpu_ids)
    
    best_params, best_value, all_results = run_best_hp(
        df_train=df_train,
        df_val=df_val,
        ticker=ticker,
        n_trials=1,
        max_concurrent=1,
        gpu_ids=gpu_ids
    )
    
    print("Best parameters:", best_params)
    print("Best value:", best_value)