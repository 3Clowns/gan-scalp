from abc import ABC

import gym
import numpy as np
import pandas as pd
import ta


from gym import spaces

import wandb
from stable_baselines3.common.monitor import Monitor


class MoexTradingEnv(gym.Env, ABC):
    """
    A custom trading environment for a single ticker's time series data.
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 10, alpha: float = 0.0001, volume=1, eta=1, action_reward=0.0, wrong_action_reward=-0.2, scale_reward=10, max_holding=40, 
                 override_scale_function: dict = {"override" : False, "function" : lambda x: x}):
        super(MoexTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((window_size) * 12 + 4,), dtype=np.float32
        )

        self.current_step = window_size
        self.position = "none"  # can be 'long' or 'short' or None
        self.entry_price = 0.0
        self.entry_step = 0  # track how long the position has been held
        self.volume = volume
        self.eta = eta
        self.pos_dict = {"none": 0, "long": 1, "short": -1}
        self.alpha = alpha 
        self.row = 0
        self.action_reward = action_reward
        self.wrong_action_reward = wrong_action_reward
        self.scale_reward = scale_reward
        self.max_holding = max_holding

        def scale_reward_function(r):
            return r * self.scale_reward
        
        if override_scale_function["override"]:
            self.scale_reward_function = override_scale_function["function"]
        else:
            self.scale_reward_function = scale_reward_function

        self._add_technical_indicators()
        self._normalize_data()

    def _normalize_data(self):
        
        self.price_scaler = self.df['close'].std()
        self.volume_scaler = self.df['volume'].std()
        
        price_columns = ['open', 'high', 'low', 'close', 'ema5', 'ema20', 'bb_upper', 'bb_lower']
        self.df[price_columns] = self.df[price_columns].divide(self.price_scaler)
        
        self.df['volume'] = self.df['volume'] / self.volume_scaler
        
        self.df['rsi'] = self.df['rsi'] / 100
        
        macd_std = self.df['macd'].std()
        self.df['macd'] = self.df['macd'] / macd_std
        self.df['macd_signal'] = self.df['macd_signal'] / macd_std

    def _add_technical_indicators(self):
        """Adding ti"""
        
        self.df['ema5'] = ta.trend.ema_indicator(self.df['close'], window=5)
        self.df['ema20'] = ta.trend.ema_indicator(self.df['close'], window=20)
        
        self.df['rsi'] = ta.momentum.rsi(self.df['close'], window=14)
        
        macd = ta.trend.MACD(self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        
        bollinger = ta.volatility.BollingerBands(self.df['close'])
        self.df['bb_upper'] = bollinger.bollinger_hband()
        self.df['bb_lower'] = bollinger.bollinger_lband()
        
        self.df = self.df.fillna(method='bfill')

    def _get_observation(self):
        """
        Returns the last `window_size` candles as a flattened array:
        [open, high, low, close, volume, ... repeated window_size times ...].
        """
        start = self.current_step - self.window_size + 1
        datetime_format = '%Y-%m-%d %H:%M:%S' 
        current_date = pd.to_datetime(self.df.iloc[self.current_step]['begin'], format=datetime_format).date()
        dates = pd.to_datetime(self.df.iloc[start:self.current_step + 1]['begin'], format=datetime_format).dt.date
        

        """ If new day has started --> skip first window size of steps for stability """
        if not all(dates == current_date):
            day_start = dates[dates == current_date].index[0]
            self.current_step = day_start + self.window_size - 1
            start = self.current_step - self.window_size + 1

        obs_data = self.df.iloc[start:self.current_step + 1][
            ['open', 'high', 'low', 'close', 'volume', 
                'ema5', 'ema20', 'rsi', 'macd', 'macd_signal', 'bb_upper']
        ]

        obs_data['hour'] = pd.to_datetime(self.df.iloc[start:self.current_step + 1]['begin'],
                                            format=datetime_format).dt.hour

        current_price = self.df.iloc[self.current_step]['close'] * self.price_scaler

        """ Concat current prices with reward for right action (for easy vf learning) and entry_price for stock (zero if none)"""
        if self.position == "long":
            obs = np.concatenate((obs_data.values.flatten(), np.array([self.pos_dict[self.position], self.row, self.scale_reward_function(current_price - self.entry_price), self.entry_price])))
        if self.position == "short":
            obs = np.concatenate((obs_data.values.flatten(), np.array([self.pos_dict[self.position], self.row, (-1) * self.scale_reward_function(current_price - self.entry_price), self.entry_price])))
        if self.position == "none":
            obs = np.concatenate((obs_data.values.flatten(), np.array([self.pos_dict[self.position], self.row, self.action_reward, self.entry_price])))


        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        reward = 0.0
        current_price = self.df.iloc[self.current_step]['close'] * self.price_scaler
        self.wrong_action_flag = False


        if action == 1:  # open long
            if self.position == "none":
                self.position = 'long'
                self.entry_price = current_price
                self.entry_step = self.current_step
                reward += self.action_reward
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True
            self.row = 0

        elif action == 2:  # close long
            if self.position == 'long':
                profit = (current_price - self.entry_price)
                reward = self.scale_reward_function(profit)
                self.entry_price = 0
                self.position = "none"
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True

            self.row = 0

        elif action == 3:  # open short
            if self.position == "none":
                self.position = 'short'
                self.entry_price = current_price
                self.entry_step = self.current_step
                reward += self.action_reward
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True
            self.row = 0

        elif action == 4:  # close short
            if self.position == 'short':
                profit = (self.entry_price - current_price)
                reward = self.scale_reward_function(profit)
                self.entry_price = 0
                self.position = "none"
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True


            self.row = 0
            
        else:
            
            self.row += 1

            if self.row > self.max_holding:
                reward -= self.eta
            
           
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
            "Wrong Action" : self.wrong_action_flag,
        }

        return obs, reward, done, info

    def reset(self):
        self.current_step = self.window_size
        self.position = "none"
        self.entry_price = 0.0
        self.entry_step = 0
        self.row = 0
        

        return self._get_observation()


class TestingTradingEnv(MoexTradingEnv):
    def __init__(self, df, window_size = 10, alpha = 0.0001, volume=1, eta=1, action_reward=0, wrong_action_reward=-0.2):
        super().__init__(df, window_size, alpha, volume, eta, action_reward, wrong_action_reward)

    def _get_observation(self):
        return super()._get_observation()
    
    def step(self, action):
        return super().step(action)
    
    def reset(self):
        return super().reset()
    
    def _calculate_reward(self, action):
        reward = 0.0
        current_price = self.df.iloc[self.current_step]['close'] * self.price_scaler
        self.wrong_action_flag = False

        if action == 1:  
            if self.position == "none":
                self.position = 'long'
                self.entry_price = current_price
                self.entry_step = self.current_step
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True

        elif action == 2:  
            if self.position == 'long':
                reward = (current_price - self.entry_price) * self.volume
                self.entry_price = 0
                self.position = "none"
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True

        elif action == 3:  # open short
            if self.position == "none":
                self.position = 'short'
                self.entry_price = current_price
                self.entry_step = self.current_step
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True

        elif action == 4:  # close short
            if self.position == 'short':
                reward = (self.entry_price - current_price) * self.volume
                self.entry_price = 0
                self.position = "none"
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True            
            
        return reward