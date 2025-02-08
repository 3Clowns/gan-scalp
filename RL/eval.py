from envs import MoexTradingEnv, TestingTradingEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from stable_baselines3.common.logger import Logger, configure
import os 
import time


def evaluate_agent(model, df_val, max_steps=10_000, window_size=50, alpha=0.02, eta=0, action_reward=0, war=0):
    env = MoexTradingEnv(df_val, window_size=window_size, alpha=alpha, eta=eta, action_reward=action_reward, wrong_action_reward=war)
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
            # total_reward += action_reward

        obs, reward, done, info = env.step(action)

        # print(info["Wrong Action"])

        if info["Wrong Action"] == True:
            wrong_actions += 1


        total_reward += reward
        steps += 1

    return total_reward, zero_actions, non_zero_actions, wrong_actions


def test_agent(model, df_val, max_steps=10_000, window_size=50, alpha=0.0, eta=0, action_reward=0, war=-0.5):
    env = TestingTradingEnv(df_val, window_size=window_size, alpha=alpha, eta=eta, action_reward=action_reward, wrong_action_reward=war)
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    zero_actions = 0
    wrong_actions = 0
    non_zero_actions = 0

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        if action == 0:
            zero_actions += 1
        else:
            non_zero_actions += 1

        obs, reward, done, info = env.step(action)

        if info["Wrong Action"] == True:
            wrong_actions += 1

        total_reward += reward
        steps += 1

    return total_reward, zero_actions, non_zero_actions, wrong_actions


class ValidationCallback(BaseCallback):
    def __init__(self, df_train, df_val, eval_freq=1000, verbose=1, window_size=50, alpha=0.02, eta=0, action_reward=0.0, war=0, vc=None):
        super(ValidationCallback, self).__init__(verbose)
        self.df_val = df_val
        self.df_train = df_train
        self.eval_freq = eval_freq
        self.window_size = window_size
        self.alpha = alpha
        self.validation_rewards = []
        self.train_rewards = []
        self.action_reward = action_reward
        self.eta = eta
        self.war = war
        self.zero_actions_row = 0
        self.best_reward = -1
        self.vc = vc
        self.train_step = 0
        self.val_step = 0
        self.trial_logger = None


    def _on_training_start(self):
        log_path = f"/place/home/max-tm/everything/RL/gan-scalp/RL/logs/trial_{time.time()}"
        os.makedirs(log_path, exist_ok=True)
        self.trial_logger = configure(log_path, ["stdout", "csv"])
        self.model.set_logger(self.trial_logger)

    def _on_step(self):
        if len(self.model.logger.name_to_value) > 0:  # Проверяем, есть ли метрики для логирования
            self.train_step += 1
            
            metrics = {
                f"train/{key}": value
                for key, value in self.model.logger.name_to_value.items()
            }
            metrics["train_step"] = self.train_step
            
            if self.vc is not None:
                self.vc(metrics)



        if self.n_calls % (self.eval_freq * 64) == 0:
            self.val_step += 1

            val_reward, za_val, nza_val, wa_val = evaluate_agent(self.model, self.df_val, max_steps=len(self.df_val), window_size=self.window_size,
                                                         alpha=self.alpha, eta=self.eta, action_reward=self.action_reward, war=self.war)
            
            val_reward_true, _, _, _ = test_agent(self.model, self.df_val, max_steps=len(self.df_val), window_size=self.window_size, alpha=0, eta=0, action_reward=0, war=-0.0)

            self.validation_rewards.append(val_reward)

            if val_reward_true > self.best_reward:
                self.best_reward = val_reward_true
                

            if self.verbose > 0:
                print(f"Validation Reward at step {self.n_calls}: {val_reward:.2f}")


            validation_metrics = {
                "validation/reward": val_reward,
                "validation/true_reward": val_reward_true,
                "validation/zero_actions": za_val,
                "validation/non_zero_actions": nza_val,
                "validation/wrong_actions": wa_val,
                "val_step": self.val_step
            }
            
            #wandb.log(validation_metrics)
            
            if self.vc is not None:
                self.vc(validation_metrics)


            
            # self.vc({"validation_reward": val_reward, "timestep": self.n_calls})
            # wandb.log({"validation_reward": val_reward, "timestep": self.n_calls})
            # wandb.log({"True validation_reward": val_reward_true, "timestep": self.n_calls})
            # wandb.log({"Zero actions on val": za_val, "Non zero actions on val": nza_val})
            # wandb.log({"Wrong actions on val" : wa_val})

            if nza_val <= 10:
                self.zero_actions_row += 1
            else:
                self.zero_actions_row = 0

        if self.zero_actions_row > 2:
            return False
        
        return True
