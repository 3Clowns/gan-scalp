from envs import MoexTradingEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb


def evaluate_agent(model, df_val, max_steps=10_000, window_size=50, alpha=0.02):
    env = MoexTradingEnv(df_val, window_size=window_size, alpha=alpha)
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    zero_actions = 0
    wrong_actions = 0
    non_zero_actions = 0

    while not done and steps < max_steps:
        # Predict action
        action, _states = model.predict(obs, deterministic=True)

        if action == 0:
            zero_actions += 1
        else:
            non_zero_actions += 1

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    return total_reward, zero_actions, non_zero_actions


class ValidationCallback(BaseCallback):
    def __init__(self, df_train, df_val, eval_freq=1000, verbose=1, window_size=50, alpha=0.02):
        super(ValidationCallback, self).__init__(verbose)
        self.df_val = df_val
        self.df_train = df_train
        self.eval_freq = eval_freq
        self.window_size = window_size
        self.alpha = alpha
        self.validation_rewards = []
        self.train_rewards = []

    def _on_step(self):
        # Perform validation every `eval_freq` steps
        if self.n_calls % self.eval_freq == 0:
            val_reward, za_val, nza_val = evaluate_agent(self.model, self.df_val, window_size=self.window_size,
                                                         alpha=self.alpha)
            train_reward, za, nza = evaluate_agent(self.model, self.df_train, window_size=self.window_size,
                                                   alpha=self.alpha)
            self.validation_rewards.append(val_reward)
            self.train_rewards.append(train_reward)

            if self.verbose > 0:
                print(f"Validation Reward at step {self.n_calls}: {val_reward:.2f}")
            wandb.log({"validation_reward": val_reward, "timestep": self.n_calls})
            wandb.log({"train_reward": train_reward, "timestep": self.n_calls})
            wandb.log({f"Zero actions on train": za, f"Non zero actions on train": nza})
            wandb.log({f"Zero actions on val": za_val, f"Non zero actions on val": nza_val})

        return True
