from abc import ABC

import gym
import numpy as np
import pandas as pd

from gym import spaces

import wandb
from stable_baselines3.common.monitor import Monitor


class MoexTradingEnv(gym.Env, ABC):
    """
    A custom trading environment for a single ticker's time series data.
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 10, alpha: float = 0.0001, volume=1, eta=1):
        super(MoexTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.alpha = alpha  # penalty coefficient

        # Define action space: {0: hold, 1: open long, 2: close long, 3: open short, 4: close short}
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((window_size) * 6 + 2,), dtype=np.float32
        )

        # Internal variables to track state
        self.current_step = 0
        self.position = "none"  # can be 'long' or 'short' or None
        self.entry_price = 0.0
        self.entry_step = 0  # track how long the position has been held
        self.volume = volume
        self.eta = eta
        self.wrong_action_reward = -0.02
        self.pos_dict = {"none": 0, "long": 1, "short": -1}
        self.alpha = alpha
        self.row = 0

        self.df["volume"] = np.log(self.df["volume"])

    def _get_observation(self):
        """
        Returns the last `window_size` candles as a flattened array:
        [open, high, low, close, volume, ... repeated window_size times ...].
        """
        start = self.current_step - self.window_size + 1
        if start < 0:
            # If we are at the beginning, pad with the first row
            pad_size = abs(start)

            first_row = self.df.iloc[0]
            hour = pd.to_datetime(first_row['begin']).hour
            pad_data = np.append(first_row[['open', 'high', 'low', 'close', 'volume']].values, hour)
            front_pad = np.tile(pad_data, (pad_size, 1))

            pad_df = pd.DataFrame(front_pad, columns=['open', 'high', 'low', 'close', 'volume', 'hour'])
            current_data = self.df.iloc[:self.current_step + 1][['open', 'high', 'low', 'close', 'volume']]
            datetime_format = '%Y-%m-%d %H:%M:%S'

            current_data['hour'] = pd.to_datetime(self.df.iloc[:self.current_step + 1]['begin'],
                                                  format=datetime_format).dt.hour

            # front_pad = np.tile(self.df.iloc[0][['open','high','low','close','volume']].values, (pad_size,1))
            # obs_data = pd.concat([
            #    pd.DataFrame(front_pad, columns=['open','high','low','close','volume']),
            #    self.df.iloc[:self.current_step+1][['open','high','low','close','volume']]
            # ], ignore_index=True)

            obs_data = pd.concat([pad_df, current_data], ignore_index=True)

        else:
            datetime_format = '%Y-%m-%d %H:%M:%S'

            # obs_data = self.df.iloc[start:self.current_step+1][['open','high','low','close','volume']]
            obs_data = self.df.iloc[start:self.current_step + 1][['open', 'high', 'low', 'close', 'volume']]
            obs_data['hour'] = pd.to_datetime(self.df.iloc[start:self.current_step + 1]['begin'],
                                              format=datetime_format).dt.hour

        obs = np.concatenate((obs_data.values.flatten(), np.array([self.pos_dict[self.position], self.row])))
        # print(obs)
        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        reward = 0.0
        current_price = self.df.iloc[self.current_step]['close']

        if action == 1:  # open long
            if self.position == "none":
                self.position = 'long'
                self.entry_price = current_price
                self.entry_step = self.current_step
            else:
                reward += self.wrong_action_reward
            self.row = 0

        elif action == 2:  # close long
            if self.position == 'long':
                # Profit is (sell_price - buy_price)
                reward = (current_price - self.entry_price) * self.volume
                # Subtract our holding penalty
                hold_time = (self.current_step - self.entry_step)
                reward -= (hold_time * self.alpha)

                self.position = "none"
            else:
                reward += self.wrong_action_reward
            self.row = 0

        elif action == 3:  # open short
            if self.position == "none":
                self.position = 'short'
                self.entry_price = current_price
                self.entry_step = self.current_step
            else:
                reward += self.wrong_action_reward
            self.row = 0

        elif action == 4:  # close short
            if self.position == 'short':
                # Profit is (buy_price - sell_price)
                reward = (self.entry_price - current_price) * self.volume
                # Subtract our holding penalty
                hold_time = (self.current_step - self.entry_step)
                reward -= (hold_time * self.alpha)

                self.position = "none"
            else:
                reward += self.wrong_action_reward
            self.row = 0
        else:
            self.row += 1

        # If action == 0 or the position is still open, we can give no immediate reward
        # but we might incorporate a negative drift penalty for open position if desired.

        if action == 0 and self.position == "none":
            reward -= (self.alpha * self.eta) * (self.row > 20)

        return reward

    def step(self, action):
        reward = self._calculate_reward(action)

        self.current_step += 1
        done = False
        if self.current_step >= len(self.df) - 1:
            done = True

        obs = self._get_observation()
        info = {
            "Action": f"{action}",
            "Reward": f"{reward}",
        }

        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.position = "none"
        self.entry_price = 0.0
        self.entry_step = 0

        return self._get_observation()
