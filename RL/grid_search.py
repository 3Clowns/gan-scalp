import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from train import train_rnn_rl
import os
from create_data import load_dataset, create_dataset, save_dataset
from utils import prepare_data, time_split
from eval import evaluate_agent, test_agent, test_agent_with_actions
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
import itertools

def generate_all_combinations(param_grid, n_trials):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    combinations = []
    for values in itertools.product(*param_values):
        combo = dict(zip(param_names, values))
        combinations.append(combo)
    
    return random.sample(combinations, min(n_trials, len(combinations)))

def run_trial(args):
    params, gpu_id, df_train, df_val, ticker = args
    try:
        run_name = f"trial_{params['trial_number']}_gpu_{gpu_id}"
        
        wandb.init(
            project="RL MaskedPPO hps search 2.0 day_penalty=10 no augment big_data new_era fixed",
            reinit=True,
            name=run_name,
            config=params
        )
        
        step_counter = 0
        def custom_log(metrics):
            nonlocal step_counter
            wandb.log(metrics, step=step_counter)
            step_counter += 1
        
        device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
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
            device=device,
            clip_range=params['clip_range'],
            gae_lambda=params['gae_lambda'],
            logging_callback=custom_log,
            eval_freq=1000,
        )
        
        profit, _, _, _, _ = evaluate_agent(
            model, df_train[ticker], 
            max_steps=len(df_train[ticker]), 
            window_size=params['window_size'],
            alpha=0, eta=0, action_reward=0, war=0,
            scale_reward=params["scale_reward"],
            max_holding=40,
        )
        
        real_profit, _, _, _, _ = test_agent(
            model, df_val[ticker],
            max_steps=len(df_val[ticker]),
            window_size=params['window_size'],
            alpha=0, eta=0, action_reward=0, war=0,
            scale_reward=params["scale_reward"],
            max_holding=40,
        )

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
            "Train profit in train env": profit,
            **params
        })

        del model
        torch.cuda.empty_cache()
        
        wandb.finish()
        return params, real_profit
        
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        if wandb.run is not None:
            wandb.finish()
        return params, float('-inf')


def generate_random_params(param_grid, n_trials):

    params_list = []
    
    for _ in range(n_trials):
        trial_params = {}
        
        for param_name, param_values in param_grid.items():
            trial_params[param_name] = random.choice(param_values)
        params_list.append(trial_params)
    
    return params_list

def run_grid_search(df_train, df_val, ticker, n_trials=100, max_concurrent=6, gpu_ids=[0]):
    param_grid = {
        #"window_size": list(range(5, 51, 5)),
        #"learning_rate": [4e-6, 1e-5, 3e-5, 7e-5, 1e-4, 5e-4, 1e-3],
        #"n_epochs": [3, 7, 11, 15],
        #"batch_size": [128, 256, 512, 1024, 2048],
        #"n_steps": [256, 512, 1024, 2048, 4096],
        #"scale_reward": [1, 10, 100, 1000],
        #"total_epochs": [4, 8, 16],
        #"hidden_dim_rnn": [64, 128, 256, 512],
        #"lstm_layers": [2, 4, 8],
        #"features_dim": [64, 128, 256, 512],
        "ent_coef": [1e-3, 1e-2, 1e-1, 0],
        #"eta" : [0, 1, 0.1, 0.01, 10],
        "clip_range" : [0.5, 0.2, 0.1, 0.05],
        "gae_lambda" : [0.95, 0.85, 0.8, 0.65, 0.5]
    }
    
    fixed_params = {
        "action_reward": 0,
        "wrong_action_reward": -0.0,

        "window_size": 50,
        "learning_rate": 2e-5,
        "n_epochs": 15,
        "batch_size": 1024,
        "n_steps": 4096,
        "scale_reward": 100,
        "total_epochs": 10,
        "hidden_dim_rnn": 256,
        "lstm_layers": 4,
        "features_dim": 512,
        # "ent_coef": [1e-4, 1e-3, 1e-2, 1e-1, 0.5],
        "eta" : 0.01,
        # "clip_range" : [0.2, 0.1, 0.01, 0.05],
        # "gae_lambda" : [0.9, 0.95, 0.99, 0.995, 0.85, 0.5]
    }
    
    combinations = generate_all_combinations(param_grid, n_trials)

    trial_args = []
    for i, combination in enumerate(combinations):
        params = combination
        params.update(fixed_params)
        params['trial_number'] = i
        gpu_id = gpu_ids[i % len(gpu_ids)]
        trial_args.append((params, gpu_id, df_train, df_val, ticker))
    
    results = []
    

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        # Запускаем все trials
        futures = [executor.submit(run_trial, args) for args in trial_args]
        
        # Собираем результаты
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
    filename = "mid_data.pkl"
    directory_path = "./"
    path = os.path.join(directory_path, filename)

    if os.path.exists(path):
        data = load_dataset(filename)
    else:
        data = create_dataset(num_weeks=72)
        save_dataset(data, path)

    data = prepare_data(data)
    df_train, df_val, df_test = time_split(df_dict=data, train_ratio=0.7, val_ratio=0.15)
    ticker = "LKOH"
    n_gpus = torch.cuda.device_count()
    gpu_ids = list(range(n_gpus)) if n_gpus > 0 else [None]

    print(gpu_ids)
    
    best_params, best_value, all_results = run_grid_search(
        df_train=df_train,
        df_val=df_val,
        ticker=ticker,
        n_trials=100,
        max_concurrent=6,
        gpu_ids=gpu_ids
    )
    
    print("Best parameters:", best_params)
    print("Best value:", best_value)