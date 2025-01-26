from envs import MoexTradingEnv
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import prepare_data, time_split
from eval import ValidationCallback


def train_vanilla_rl(df_train, df_val, ticker, window_size=10, alpha=0.0001, total_timesteps=10000, eta=100,
                     eval_freq=1000, lr=0.0003, n_epochs=10, batch_size=64, n_steps=2048, policy=None):
    wandb.init(project="RL", config={
        "ticker": ticker,
        "window_size": window_size,
        "alpha": alpha,
        "total_timesteps": total_timesteps,
        "learning_rate": lr,
        "eta": eta,
        "eval_freq": eval_freq,
        "n_epochs" : n_epochs,
    })

    # df = prepare_data(data_dict, ticker)
    # df_train, df_val, df_test = time_split(df, train_ratio=0.7, val_ratio=0.2)

    env = MoexTradingEnv(df_train, window_size=window_size, alpha=alpha, eta=eta)
    vec_env = DummyVecEnv([lambda: env])

    if policy is not None:
        model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=n_steps, n_epochs=n_epochs, learning_rate=lr, batch_size=batch_size, policy_kwargs=policy, gae_lambda=0.95, ent_coef=0.01)

    validation_callback = ValidationCallback(df_train, df_val, eval_freq=eval_freq, window_size=window_size, alpha=0)
    model.learn(total_timesteps=total_timesteps, callback=validation_callback)

    # wandb.finish()

    return model
