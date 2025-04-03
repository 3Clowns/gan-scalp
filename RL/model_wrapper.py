import wandb
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, DQN
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Callable

from envs import MoexTradingEnv
from eval import ValidationCallback
from envs import create_masked_env


class RLModelWrapper:

    def __init__(
            self,
            algo: str,
            train_env: MoexTradingEnv,
            val_env=None,
            test_env=None,
            config: dict = None,
            val_callback_config: dict = None,
            train_env_config: dict = None,
            val_env_config: dict = None,
            dump_params_to_stdout: bool = True,
    ):
        if config is None:
            config = {}
        self.config = config
        self.algo = algo

        self.train_env = self._wrap_env(train_env) if train_env else None
        self.val_env = self._wrap_env(val_env) if val_env else None
        self.test_env = self._wrap_env(test_env) if test_env else None
        self.train_env_config = train_env_config
        self.val_env_config = val_env_config
        self.dump_params_to_stdout = dump_params_to_stdout

        print("All envs initialized.")

        self.model = None
        self.logging_callback = ValidationCallback(train_env_config=train_env_config, val_env_config=val_env_config, **val_callback_config)
        self.run_name = config.get("run_name", "default_run")
        wandb.init(project=config.get("project_name", "RL_Project"), name=self.run_name, config=self.config)

    def _wrap_env(self, env):
        if env is None:
            return None

        if not hasattr(env, "get_action_mask"):
            env = DummyVecEnv([lambda: env])
            print("Non masked env created successfully.")
            return env

        masked_env = create_masked_env(base_env=env)
        print("Masked env created successfully.")

        return masked_env

    def create_model(self):

        if not self.train_env:
            raise ValueError("No train_env found.")

        algo = self.algo.lower()
        policy_type = self.config.get("policy_type", "MlpPolicy")

        model_params_maskable_ppo = {
            k: self.config.get(k, v)
            for k, v in self.config.items()
            if k in MaskablePPO.__init__.__code__.co_varnames
        }

        if self.dump_params_to_stdout:
            print("STDOUT model params: ", model_params_maskable_ppo)

        if algo == "maskable_ppo":
            self.model = MaskablePPO(
                env=self.train_env,
                **model_params_maskable_ppo,
            )
        elif algo == "ppo":
            self.model = PPO(
                policy_type,
                self.train_env,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1
            )
        elif algo == "dqn":
            self.model = DQN(
                policy_type,
                self.train_env,
                learning_rate=lr,
                batch_size=batch_size,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algo: {algo}")

        print(f"Model created: {algo.upper()} with config: {self.config}")

    def train(self, total_timesteps: int, progress_bar: bool = True):
        if self.model is None:
            print("Model hasn't been initialized, creating model via provided config...")
            self.create_model()

        print(f"Training started for {total_timesteps} timesteps.")
        self.model.learn(total_timesteps=total_timesteps, callback=self.logging_callback, progress_bar=progress_bar)
        print("Model trained successfully.")

    def evaluate(self, env_type: str = "val", max_steps: int = 10000) -> float:
        if env_type not in ("val", "test"):
            raise ValueError("env_type has to be 'val' or 'test'.")
        env = self.val_env if env_type == "val" else self.test_env
        if env is None:
            print(f"{env_type} is None, skipping evaluation.")
            return 0.0

        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            if hasattr(env, "get_action_mask"):
                action_masks = env.env_method("get_action_mask")[0]
                action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)
            done = dones[0]
            total_reward += rewards[0]
            steps += 1

        print(f"{env_type.capitalize()} reward = {total_reward:.2f} on {steps} steps.")
        return total_reward

    def save(self, path: str):
        if self.model is None:
            raise ValueError("Nothing to save.")
        self.model.save(path)
        print(f"Model saved to {path}")
        torch.save(self.config, path + "_config.pt")
        print(f"Config saved to {path + '_config.pt'}.")

    def load(self, model_path: str):
        config_path = model_path + "_config.pt"
        loaded_config = torch.load(config_path)
        self.config.update(loaded_config)
        self.create_model()
        self.model = self.model.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path} with config from {config_path}")
