from envs import MoexTradingEnv, TestingTradingEnv
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from stable_baselines3.common.logger import Logger, configure
import os
import time
from envs import create_masked_env
from sb3_contrib.common.wrappers import ActionMasker
import sys
import traceback

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_agent(model, trading_env, max_steps, with_action_logs=True):
    """
    Evaluate the trading agent on a validation dataset.

    Parameters:
        model: Trained model
        trading_env: Base trading environment.
        max_steps: Maximum number of evaluation steps.
        with_action_logs: save actions logs.
    Returns:
        A tuple of (total_reward, zero_actions, non_zero_actions, wrong_actions, metrics).
    """
    logger.info("Starting agent evaluation.")
    print("Starting agent evaluation.")

    if max_steps > len(trading_env.env_method("get_df")[0]):
        print("Warning, max_steps > dataframe size")

    try:
        obs = trading_env.reset()
        total_reward = 0.0
        steps = 0
        zero_actions = 0
        non_zero_actions = 0
        wrong_actions = 0  # Placeholder: update if logic for wrong actions is added

        logger.info("Evaluation loop started.")

        log_file = None
        if not os.path.exists("logs") and with_action_logs:
            os.mkdir("logs")

        if with_action_logs:
            log_file = open("logs/logs.txt", "a")

        # Main evaluation loop.
        while steps < max_steps and trading_env.env_method("is_reset")[0] < 2:
            action_masks = trading_env.env_method("get_action_mask")[0]
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

            zero_actions += int(action[0] == 0)
            non_zero_actions += int(action[0] != 0)

            current_price = trading_env.env_method("get_current_price")[0]
            obs, rewards, dones, infos = trading_env.step(action)
            done = dones[0]
            total_reward += rewards[0]
            steps += 1

            if with_action_logs and action != 0:
                log_file.write("Action {}\n".format(action))
                log_file.write(f"Step {trading_env.env_method('get_current_step')[0]}:\n")
                for key, value in infos[0].items():
                    log_file.write(f"{key}: {value}\n")
                log_file.write(f"Current price: {current_price}\n")
                log_file.write(f"Reward: {infos[0]['Reward']}\n")
                log_file.write("-" * 30 + "\n\n")

            logger.debug(f"Step {steps}: Action={action}, Reward={rewards[0]}, Done={done}")

        logger.info(f"Evaluation completed: {steps} steps, Total Reward={total_reward}.")
        metrics = trading_env.env_method("calculate_metrics")[0]

        if with_action_logs:
            log_file.write("\nFinal Results:\n")
            log_file.write(f"Total steps: {steps}\n")
            log_file.write(f"Total reward: {total_reward:.2f}\n")
            # risk_free_rate = 0.20 / (252 * 24 * 60)
            # log_file.write(f"Sharpe ratio: {trading_env.calculate_sharpe_ratio(risk_free_rate)}")
            for key, value in trading_env.env_method("calculate_metrics")[0].items():
                log_file.write(f"{key}: {value}\n")

    except Exception as e:
        logger.exception("An error occurred during evaluation:")
        raise
    finally:
        trading_env.close()
        logger.info("Environment closed.")

    return total_reward, zero_actions, non_zero_actions, wrong_actions, metrics


class ValidationCallback(BaseCallback):
    def __init__(self, df_train, df_val, eval_freq, verbose, window_size, alpha, eta, action_reward, vc,
                 scale_reward, train_env_config, val_env_config):
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
        self.zero_actions_row = 0
        self.best_reward = -1
        self.vc = vc  # Validation callback wandb
        self.train_step = 0
        self.val_step = 0
        self.trial_logger = None
        self.scale_reward = scale_reward
        self.train_env_config = train_env_config
        self.val_env_config = val_env_config

    def _on_training_start(self):
        log_path = f"./RL/logs/trial_{time.time()}"
        os.makedirs(log_path, exist_ok=True)
        self.trial_logger = configure(log_path, ["stdout", "csv"])
        self.model.set_logger(self.trial_logger)

    def _on_step(self):
        try:
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

                assert len(self.df_val) > 10 or len(self.df_val) == 0, "Evalution is probably wrong"

                val_reward, za_val, nza_val, wa_val, _ = evaluate_agent(
                    self.model, create_masked_env(MoexTradingEnv(**self.train_env_config)), max_steps=min(20000, len(self.df_val)) - 1, with_action_logs=True)

                val_reward_true, za_val_true, nza_val_true, wa_val_true, info = evaluate_agent(self.model, create_masked_env(TestingTradingEnv(**self.val_env_config)), max_steps=min(20000, len(self.df_val)) - 1, with_action_logs=True)
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
                    "val_step": self.val_step,
                    "validation/zero_actions_true": za_val_true,
                    "validation/non_zero_actions_true": nza_val_true,
                    "validation/wrong_actions_true": wa_val_true,
                }

                if self.vc is not None:
                    self.vc(validation_metrics)
                    self.vc(info)

                if nza_val <= 10:
                    self.zero_actions_row += 1
                else:
                    self.zero_actions_row = 0

            if self.zero_actions_row > 2:
                return False

            return True
        except Exception:
            print("Error in _on_step:")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
            raise
