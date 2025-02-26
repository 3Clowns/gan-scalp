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

def evaluate_agent(model, df_val, max_steps, window_size, alpha, eta, action_reward, war, scale_reward, max_holding):
    env = create_masked_env(df=df_val, window_size=window_size, alpha=alpha, eta=eta, action_reward=action_reward, wrong_action_reward=war, scale_reward=scale_reward, max_holding=max_holding)
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    zero_actions = 0
    wrong_actions = 0
    non_zero_actions = 0

    while not done and steps < max_steps:
        action_masks = env.env_method("get_action_mask")[0]  #
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

        if action == 0:
            zero_actions += 1
        else:
            non_zero_actions += 1

        obs, rewards, dones, infos = env.step([action])

        #if infos["Wrong Action"] == True:
        #    wrong_actions += 1
        # print(rewards, dones, infos)

        total_reward += rewards[0]
        steps += 1

    metrics = env.env_method("calculate_metrics")[0]
    env.close()

    return total_reward, zero_actions, non_zero_actions, wrong_actions, metrics


def test_agent(model, df_val, max_steps, window_size, alpha, eta, action_reward, war, scale_reward, max_holding):
    
    env = create_masked_env(base_env=TestingTradingEnv, df=df_val, window_size=window_size, alpha=alpha, eta=eta, action_reward=action_reward, wrong_action_reward=war, scale_reward=scale_reward, max_holding=max_holding)
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    zero_actions = 0
    wrong_actions = 0
    non_zero_actions = 0
    infos = None

    while not done and steps < max_steps:
        action_masks = env.env_method("get_action_mask")[0]  #
        action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

        if action == 0:
            zero_actions += 1
        else:
            non_zero_actions += 1

        obs, rewards, dones, infos = env.step([action])

        total_reward += rewards[0]
        steps += 1

    # print(infos)
    

    metrics = env.env_method("calculate_metrics")[0]
    env.close()

    return total_reward, zero_actions, non_zero_actions, wrong_actions, metrics


class ValidationCallback(BaseCallback):
    def __init__(self, df_train, df_val, eval_freq, verbose, window_size, alpha, eta, action_reward, war, vc, scale_reward):
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
        self.scale_reward = scale_reward


    def _on_training_start(self):
        log_path = f"/place/home/max-tm/everything/RL/gan-scalp/RL/logs/trial_{time.time()}"
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
                    self.model, self.df_val, max_steps=len(self.df_val), window_size=self.window_size,
                    alpha=self.alpha, eta=self.eta, action_reward=self.action_reward, war=self.war, 
                    scale_reward=self.scale_reward, max_holding=40)
                
                val_reward_true, za_val_true, nza_val_true, wa_val_true, info = test_agent(self.model, self.df_val, 
                    max_steps=len(self.df_val), window_size=self.window_size, alpha=0, 
                    eta=0, action_reward=0, war=-0.0, scale_reward=self.scale_reward, max_holding=40)

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
                    "validation/zero_actions_true" : za_val_true,
                    "validation/non_zero_actions_true" : nza_val_true,
                    "validation/wrong_actions_true" : wa_val_true,
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
        except Exception as e:
            print("Error in _on_step:")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
            raise




def test_agent_with_actions(model, df, max_steps, window_size, alpha=0, eta=0, action_reward=0, war=0, scale_reward=1.0, max_holding=40):
    
    timestamp = time.time()
    log_filename = f"trading_logs/trial_{timestamp}.txt"
    os.makedirs("trading_logs", exist_ok=True)

    print("Testing started")

    max_steps = len(df)

    with open(log_filename, 'w') as log_file:
        env = create_masked_env(
            base_env=TestingTradingEnv,            
            df=df,
            window_size=window_size,
            alpha=0,
            eta=0,
            volume=1,
            action_reward=action_reward,
            wrong_action_reward=0,
            scale_reward=scale_reward,
            max_holding=40,
        )
        
        obs = env.reset()
        done = False
        total_reward = 0
        n_steps = 0
        actions_taken = []
        rewards = []
        
        log_file.write(f"Trading Log for Trial\n")
        log_file.write(f"Started at: {timestamp}\n")
        log_file.write("-" * 50 + "\n\n")
        
        while not done and n_steps < max_steps:
            action_masks = env.env_method("get_action_mask")[0]  #
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)

            assert len(env.env_method("get_action_mask")) == 1, "N_envs is not equal to one"

            next_obs, reward, dones, infos = env.step([action])
            n_steps = env.env_method("get_current_step")[0] - 1
            
            current_price = df.loc[n_steps]['close']
            
            if not infos[0]["Wrong Action"]:
                log_file.write(f"Step {n_steps}:\n")
                for key, value in infos[0].items():
                    log_file.write(f"{key}: {value}\n")
                log_file.write(f"Current price: {current_price}\n")
                log_file.write(f"Reward: {infos[0]['Reward']}\n")
                log_file.write("-" * 30 + "\n\n")
            
            done = dones[0]
            total_reward += reward[0]
            obs = next_obs
            n_steps += 1
            actions_taken.append(action)
            rewards.append(reward[0])
        
        log_file.write("\nFinal Results:\n")
        log_file.write(f"Total steps: {n_steps}\n")
        log_file.write(f"Total reward: {total_reward:.2f}\n")

        risk_free_rate = 0.20 / (252 * 24 * 60)
        # log_file.write(f"Sharpe ratio: {env.calculate_sharpe_ratio(risk_free_rate)}")
        for key, value in env.env_method("calculate_metrics")[0].items():
            log_file.write(f"{key}: {value}\n")
    
    print("Testing completed...")
    
    return _, total_reward, actions_taken, rewards


