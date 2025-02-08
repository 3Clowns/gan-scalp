import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from train import train_vanilla_rl, train_rnn_rl
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
from concurrent.futures import ThreadPoolExecutor


from concurrent.futures import ProcessPoolExecutor

def run_trial(trial, gpu_id):
    return objective(trial, gpu_id)

def run_trials_parallel(trial_queue, gpu_id, max_concurrent=6):
    trials = []
    while not trial_queue.empty():
        try:
            trial = trial_queue.get_nowait()
            trials.append(trial)
        except queue.Empty:
            break

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(run_trial, trial, gpu_id): trial for trial in trials}
        for future in futures:
            try:
                value = future.result()
                # Здесь можно добавить обработку результата для каждого trial
            except Exception as e:
                print(f"Error in trial: {str(e)}")



'''def run_trials_batch(trial_queue, gpu_id, study_name, storage_url=None, max_concurrent=6):
    """Функция для запуска нескольких trials на одной GPU"""
    # study = optuna.load_study(study_name=study_name, storage=storage_url)


    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        while True:
            try:
                trials_batch = []
                for _ in range(max_concurrent):
                    try:
                        trial = trial_queue.get_nowait()
                        trials_batch.append(trial)
                    except queue.Empty:
                        break
                
                if not trials_batch:
                    break
                
                futures = []
                for trial in trials_batch:
                    future = executor.submit(objective, trial, gpu_id)
                    futures.append((trial, future))
                
                for trial, future in futures:
                    try:
                        value = future.result()
                        trial.set_user_attr('gpu_id', gpu_id)
                        # study.tell(trial, value)
                    except Exception as e:
                        print(f"Error in trial {trial.number} on GPU {gpu_id}: {str(e)}")
                        # study.tell(trial, float('-inf'))
                
            except Exception as e:
                print(f"Batch processing error on GPU {gpu_id}: {str(e)}")'''

def optimize_hyperparameters_parallel(df_train, df_val, ticker, n_trials=100, max_concurrent=6, gpu_id=0):
    """Оптимизация с несколькими параллельными trials на одной GPU
    study_name = f"ppo_optimization_{int(time.time())}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )

    trial_queue = Queue()
    for _ in range(n_trials):
        trial = study.ask()
        trial_queue.put(trial)

    run_trials_parallel(trial_queue, gpu_id, max_concurrent)"""
    study_name = f"ppo_optimization_{int(time.time())}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, gpu_id=0), n_trials=100, n_jobs=1, catch=(Exception, ))

    return study.best_params, study.best_value



def objective(trial, gpu_id=None):
    """Objective function для Optuna."""
    import wandb
    import tempfile
    # tmp_wandb_dir = tempfile.mkdtemp(prefix=f"wandb_trial_{trial.number}")
    # os.environ["WANDB_DIR"] = tmp_wandb_dir


    window_size = trial.suggest_int("window_size", 5, 30)
    learning_rate = trial.suggest_loguniform("learning_rate", 4e-6, 1e-3)
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024, 2048])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
    # action_reward = trial.suggest_uniform("action_reward", 0.0001, 0.01)
    # wrong_action_reward = trial.suggest_uniform("wrong_action_reward", -200, -10)
    action_reward = 0
    wrong_action_reward = -0.01
    #eta = trial.suggest_loguniform("eta", 1e-3, 1)
    eta = 0
    scale_reward = trial.suggest_loguniform("scale_reward", 1, 1000)
    overall_epochs = trial.suggest_categorical("total epochs", [1, 2, 4, 8, 16])
    hidden_dim_rnn = trial.suggest_categorical("hidden_dim_rnn", [64, 128, 256])
    lstm_layers = trial.suggest_categorical("lstm_layers", [2, 4, 8])
    features_dim = trial.suggest_categorical("features_dim", [64, 128, 256])
    ent_coef = trial.suggest_loguniform("entropy_coef", 1e-4, 0.5)

    device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None and torch.cuda.is_available() else 'cpu')
    if gpu_id is not None:
        torch.cuda.set_device(device)
    
    try:
        run_name = f"trial_{trial.number}_gpu_{gpu_id}"

        wandb.init(project="RL 7.0", reinit=True, name=run_name, config={
                "window_size": window_size,
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "n_steps": n_steps,
                "action_reward": action_reward,
                "wrong_action_reward": wrong_action_reward,
                "eta": eta,
                "scale_reward": scale_reward,
                "hidden_dim_rnn": hidden_dim_rnn,
                "lstm_layers": lstm_layers,
                "features_dim": features_dim,
                "total_epochs": overall_epochs,
                "ent_coef": ent_coef,
            },
        )

        step_counter = 0
        def custom_log(metrics):
            nonlocal step_counter
            wandb.log(metrics, step=step_counter)
            step_counter += 1

        model = train_rnn_rl(
            df_train=df_train[ticker],
            df_val=df_val[ticker],
            ticker=ticker,
            window_size=window_size,
            lr=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_steps=n_steps,
            action_reward=0,
            wrong_action_reward=wrong_action_reward,
            total_timesteps=len(df_train[ticker]) * overall_epochs,
            eta=eta,
            scale_reward=scale_reward,
            alpha=0.0,
            lstm_layers=lstm_layers,
            hidden_size=hidden_dim_rnn,
            features_dim=features_dim,
            ent_coef=ent_coef,
            device=device,
            logging_callback=custom_log,
        )

        profit, _, _, _ = evaluate_agent(model, df_val[ticker], max_steps=len(df_val[ticker]), window_size=window_size, alpha=0, eta=0, action_reward=0, war=0)
        real_profit, _, _, _ = test_agent(model, df_val[ticker], max_steps=len(df_val[ticker]), window_size=window_size, alpha=0, eta=0, action_reward=0, war=0)
        
        custom_log({
            "trial_value": real_profit,
            "window_size": window_size,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "n_steps": n_steps,
            "action_reward": action_reward,
            "wrong_action_reward": wrong_action_reward,
            "Val profit in train env" : profit,
        })

        wandb.finish()
        
        return real_profit

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        wandb.finish()
        return float('-inf')
    
        

def optimize_hyperparameters(df_train, df_val, ticker, n_trials=100, gpu_id=None):
    """Основная функция оптимизации гиперпараметров."""


    
    # Настройка оптимизации
    study = optuna.create_study(
        study_name="ppo_optimization",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    wandb.log({
        "best_trial_value": trial.value,
        "best_params": trial.params
    })
    
    return trial.params, trial.value


"""def optimize_hyperparameters_parallel(df_train, df_val, ticker, n_trials=100, n_gpus=2):
    
    def objective_wrapper(trial):
        gpu_id = trial.number % n_gpus
        return objective(trial, gpu_id)
    
    study = optuna.create_study(
        study_name="ppo_optimization",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=n_gpus * 4, catch=(Exception,))
    
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    #wandb.log({
    #    "best_trial_value": trial.value,
    #    "best_params": trial.params
    #})
    
    return trial.params, trial.value"""


mp.set_start_method("spawn", force=True)
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
ticker = "VTBR"
best_params, best_value = optimize_hyperparameters_parallel(df_train, df_val, ticker, n_trials=100)