from abc import ABC

import gymnasium
import numpy as np
import pandas as pd
import ta
import datetime


from gymnasium import spaces

import wandb
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import traceback



class MoexTradingEnv(gymnasium.Env, ABC):
    """
    A custom trading environment for a single ticker's time series data.
    """

    def __init__(self, df: pd.DataFrame, window_size: int = 10, alpha: float = 0.0001, volume=1, eta=1, action_reward=0.0, wrong_action_reward=-0.2, scale_reward=10, max_holding=40, 
                 override_scale_function: dict = {"override" : False, "function" : lambda x: x}):
        super(MoexTradingEnv, self).__init__()

        assert df.index.is_monotonic_increasing, "Dataframe index must be sorted"
        assert not df[['open','high','low','close']].isnull().any().any(), "NaN values in OHLC data"

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.action_space = spaces.Discrete(5)
        self.aug_prob = 0.0
        
        self.num_time_features = 12
        self.num_additional_info = 4 + 5 + 1  # +5 для маски действий
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=((window_size) * self.num_time_features + self.num_additional_info,),
            dtype=np.float32
        )


        self.current_step = window_size
        self.position = "none"  
        self.entry_price = 0.0
        self.entry_step = 0 
        self.volume = volume
        self.eta = eta
        self.pos_dict = {"none": 0, "long": 1, "short": -1}
        self.alpha = alpha 
        self.row = 0
        self.action_reward = action_reward
        self.wrong_action_reward = wrong_action_reward # DEPRECATED SINCE 02.09.25 SET TO ZERO
        self.wrong_action_reward = 0
        self.scale_reward = scale_reward
        self.max_holding = max_holding
        self.returns = []
        self.profits = []
        self.longs = 0
        self.shorts = 0
        self.transfer_day_flag = False
        self.day_penalty = 10
        self.current_augmented_date = None
        self.was_reset = 0

        def scale_reward_function(r):
            return r * self.scale_reward
        
        if override_scale_function["override"]:
            self.scale_reward_function = override_scale_function["function"]
        else:
            self.scale_reward_function = scale_reward_function

        self._add_technical_indicators()
        self._normalize_data()

        print("Training enviroment initialized...")

    def _normalize_data(self):
        self.price_scaler = 1000
        self.volume_scaler = 1000
        
        # self.price_scaler = self.df['close'].std()
        # self.volume_scaler = self.df['volume'].std()
        
        price_columns = ['open', 'high', 'low', 'close', 'ema5', 'ema20', 'bb_upper', 'bb_lower']
        self.df[price_columns] = self.df[price_columns].divide(self.price_scaler)
        
        self.df['volume'] = self.df['volume'] / self.volume_scaler
        
        self.df['rsi'] = self.df['rsi'] / 100
        
        #macd_std = self.df['macd'].std()
        macd_std = 1
        self.df['macd'] = self.df['macd'] / macd_std
        self.df['macd_signal'] = self.df['macd_signal'] / macd_std

    def augment_day_data(self, current_date: datetime.date) -> pd.DataFrame:
        """"""
        day_mask = pd.to_datetime(self.df['begin']).dt.date == current_date
        day_data = self.df[day_mask]
        
        mean_price = day_data['close'].mean()
                
        augmented_full_day = day_data.copy(deep=True)
        
        price_columns = ['open', 'high', 'low', 'close']
        augmented_full_day[price_columns] = 2 * mean_price - augmented_full_day[price_columns]

        
        augmented_full_day['ema5'] = ta.trend.ema_indicator(augmented_full_day['close'], window=5)
        augmented_full_day['ema20'] = ta.trend.ema_indicator(augmented_full_day['close'], window=20)
        
        augmented_full_day['rsi'] = ta.momentum.rsi(augmented_full_day['close'], window=14)
        augmented_full_day['rsi'] = augmented_full_day['rsi'] / 100
        
        macd = ta.trend.MACD(augmented_full_day['close'])
        augmented_full_day['macd'] = macd.macd()
        augmented_full_day['macd_signal'] = macd.macd_signal()
        
        macd_std = augmented_full_day['macd'].std()
        if macd_std != 0:
            augmented_full_day['macd'] = augmented_full_day['macd'] / macd_std
            augmented_full_day['macd_signal'] = augmented_full_day['macd_signal'] / macd_std
        
        bollinger = ta.volatility.BollingerBands(augmented_full_day['close'])
        augmented_full_day['bb_upper'] = bollinger.bollinger_hband()
        
        augmented_full_day = augmented_full_day.fillna(method='bfill')
        
        return augmented_full_day

    def get_action_mask(self):
        mask = np.zeros(5, dtype=np.float32)
        
        if self.position == "none":
            mask[1] = 1  # open long
            mask[3] = 1  # open short
            mask[0] = 1  # hold
        elif self.position == "long":
            mask[2] = 1  # close long
            mask[0] = 1  # hold
        elif self.position == "short":
            mask[4] = 1  # close short
            mask[0] = 1  # hold
            
        return mask


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

    def _force_close_position(self):
        if self.position == 'long':
            profit = (self.current_price - self.entry_price)
        else:
            profit = (self.entry_price - self.current_price)
        self.profits.append(profit)
        self.position = "none"
        self.entry_price = 0

    def _get_observation(self):
        """
        Creates an observation window of market data.
        
        Returns:
            numpy.ndarray: Flattened array of market features for the last window_size candles
            
        Notes:
            - Ensures all observations are from the same trading day
            - Adjusts current_step if day boundary is crossed
        """

        last_minute = False
        datetime_format = '%Y-%m-%d %H:%M:%S' 


        try:
            action_mask = self.get_action_mask()
            start = self.current_step - self.window_size + 1

            current_date = pd.to_datetime(self.df.loc[self.current_step]['begin'], format=datetime_format).date()
            dates = pd.to_datetime(self.df.loc[start:self.current_step + 1]['begin'], format=datetime_format).dt.date
            self.transfer_day_flag = False

            try:
                next_date = pd.to_datetime(self.df.loc[self.current_step + 2]['begin'], format=datetime_format).date()
            except:
                print("Dataframe has ended can't get next date")
                next_date = current_date

            if current_date != next_date:
                last_minute = True

            if hasattr(self, 'current_augmented_day') and self.current_augmented_date != current_date:
                del self.current_augmented_day
                self.current_augmented_date = None

            """ If new day has started --> skip first window size of steps for stability """
            if not all(dates == current_date):
                if self.position != "none":
                    self.transfer_day_flag = True
                day_start = dates[dates == current_date].index[0]

                start = day_start
                self.current_step = day_start + self.window_size - 1

                if np.random.random() < self.aug_prob and not hasattr(self, 'current_augmented_day'):
                    self.current_augmented_day = self.augment_day_data(current_date)
                    self.current_augmented_date = current_date


            if hasattr(self, 'current_augmented_day'):
                obs_data = self.current_augmented_day.loc[start:self.current_step][
                    ['open', 'high', 'low', 'close', 'volume', 
                    'ema5', 'ema20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'begin']
                ]

            else:
                obs_data = self.df.loc[start:self.current_step][
                    ['open', 'high', 'low', 'close', 'volume', 
                    'ema5', 'ema20', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'begin']
                ]

                
            obs_data['hour'] = pd.to_datetime(obs_data['begin'], format=datetime_format).dt.hour
            obs_data.drop(["begin"], inplace=True, axis=1)
            obs_data = obs_data.reset_index(drop=True)
            
            self.current_price = obs_data.iloc[self.window_size - 1]['close'] * self.price_scaler

            """ Concat current prices with reward for right action (for easy vf learning) and entry_price for stock (zero if none)"""
            if self.position == "long":
                self.probable_reward = self.scale_reward_function(self.current_price - self.entry_price)
            if self.position == "short":
                self.probable_reward = -self.scale_reward_function(self.current_price - self.entry_price)
            if self.position == "none":
                self.probable_reward = self.action_reward
            if self.transfer_day_flag:
                self.probable_reward -= self.day_penalty

            obs = np.concatenate((obs_data.values.flatten(), np.array([self.pos_dict[self.position], self.row, self.probable_reward, self.entry_price, int(last_minute)]), action_mask))
        except Exception as e:
            print("Error in _get_observation:")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10)
            raise

        return obs.astype(np.float32)

    def _calculate_reward(self, action):
        reward = 0.0

        assert self.current_price is not None 
        self.wrong_action_flag = False

        if action == 1:  # open long
            if self.position == "none":
                self.position = 'long'
                self.entry_price =self.current_price
                self.entry_step = self.current_step
                reward += self.action_reward
                self.row = 0
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True
            
        elif action == 2:  # close long
            if self.position == 'long':
                profit = (self.current_price - self.entry_price)
                reward = self.scale_reward_function(profit)
                returns = profit / self.entry_price
                self.returns.append(returns)
                self.profits.append(profit)
                self.longs += 1
                self.entry_price = 0
                self.position = "none"
                self.row = 0
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True

            
        elif action == 3:  # open short
            if self.position == "none":
                self.position = 'short'
                self.entry_price = self.current_price
                self.entry_step = self.current_step
                reward += self.action_reward
                self.row = 0

            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True

        elif action == 4:  # close short
            if self.position == 'short':
                profit = (self.entry_price - self.current_price)
                reward = self.scale_reward_function(profit)
                returns = profit / self.entry_price
                self.returns.append(returns)
                self.profits.append(profit)
                self.shorts += 1
                self.entry_price = 0
                self.position = "none"
                self.row = 0

            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True
            
        else:
            
            self.row += 1

            if self.row > self.max_holding:
                reward -= self.eta

        if self.transfer_day_flag:
            reward -= self.day_penalty
            
           
        return reward
    
    def calculate_sharpe_ratio(self, risk_free_rate=0.0):

        
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        excess_returns = returns_array - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) 


    def calculate_metrics(self):
        metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'total_profit': sum(self.profits) if self.profits else 0,
            'total_trades': self.shorts + self.longs,
            'win_rate': len([p for p in self.profits if p > 0]) / len(self.profits) if self.profits else 0,
            'average_return': np.mean(self.returns) if self.returns else 0,
            'return_std': np.std(self.returns) if self.returns else 0,
            'max_return': max(self.returns) if self.returns else 0,
            'min_return': min(self.returns) if self.returns else 0,
            'average_profit': np.mean(self.profits) if self.profits else 0,
            'shorts' : self.shorts if self.shorts else 0,
            'longs' : self.longs if self.longs else 0,
        }
        return metrics

    def step(self, action):
        reward = self._calculate_reward(action)

        self.current_step += 1
        done = False
        if self.current_step >= len(self.df) - 1:
            done = True

        info = {
            "Action": f"{action}",
            "Reward": f"{reward}",
            "Wrong Action" : self.wrong_action_flag,
        }

        obs = self._get_observation()

        if done:
            return obs, 0, done, done, info
        
        
        return obs, reward, done, done, info
    
    def is_reset(self):
        return self.was_reset

    def reset(self, seed=42, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = self.window_size
        self.position = "none"
        self.entry_price = 0.0
        self.entry_step = 0
        self.row = 0
        self.current_augmented_date = None
        self.was_reset += 1
        if hasattr(self, 'current_augmented_day'):
            del self.current_augmented_day

        print("Enviroment reset successfully...")
    
        return self._get_observation(), {}

    def get_current_step(self):
        return self.current_step
    
class TestingTradingEnv(MoexTradingEnv):
    def __init__(self, df, window_size = 10, alpha = 0.0001, volume=1, eta=1, action_reward=0, wrong_action_reward=-0.2, scale_reward=10, max_holding=40, 
                 override_scale_function: dict = {"override" : False, "function" : lambda x: x}):
        super().__init__(df, window_size, alpha, volume, eta, action_reward, wrong_action_reward, scale_reward, max_holding, override_scale_function)
        self.aug_prob = 0

        print("Testing enviroment initialized...")
        

    def _get_observation(self):
        return super()._get_observation()
    
    def step(self, action):
        return super().step(action)
    
    def reset(self, seed=42, options=None):
        return super().reset(seed=seed, options=options)
    
    def calculate_sharpe_ratio(self, risk_free_rate=0):
        return super().calculate_sharpe_ratio(risk_free_rate)
    
    def get_action_mask(self):
        return super().get_action_mask()
    
    def augment_data(self, obs_data, current_date):
        return super().augment_data(obs_data, current_date)
    
    def is_reset(self):
        return super().is_reset()
    

    def calculate_metrics(self):
        return super().calculate_metrics()
    
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
                returns = reward / self.entry_price 
                self.returns.append(returns)
                self.profits.append(reward)
                self.longs += 1 
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
                returns = reward / self.entry_price
                self.returns.append(returns)
                self.profits.append(reward)
                self.shorts += 1
                self.position = "none"
                self.entry_price = 0
            else:
                reward += self.wrong_action_reward
                self.wrong_action_flag = True            
            
        return reward
    

def create_masked_env(base_env=MoexTradingEnv, **args):
    env = base_env(**args)
    masked_env = ActionMasker(env, action_mask_fn=lambda env: env.get_action_mask())
    dummy_env = DummyVecEnv([lambda: masked_env])
    normalized_env = VecNormalize(dummy_env, norm_obs=False, norm_reward=True)
    return normalized_env
