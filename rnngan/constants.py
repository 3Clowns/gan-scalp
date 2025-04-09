import torch

###################################################################################
# Stocks universe definition
###################################################################################

# Define number of tickers in universe
N_ASSETS = 1
# Data sampling interval (1 minutes)
INTERVAL_MINUTES = 1
# Define number of trading days in a year
WORK_DAYS_PER_YEAR = 252
# Stock Market Hours (10:00 AM - 6:40 PM MSK)
STOCK_MARKET_HOURS = 8 * 60 + 40 
# Calculate 1-minute intervals per day/week/year
STOCK_INTERVALS_PER_DAY = STOCK_MARKET_HOURS // INTERVAL_MINUTES  # 520
STOCK_INTERVALS_PER_WEEK = STOCK_INTERVALS_PER_DAY * 5  # 2600
STOCK_INTERVALS_PER_YEAR = STOCK_INTERVALS_PER_DAY * WORK_DAYS_PER_YEAR # 131 040 

# Initial price
INIT_PRICE = 7574.0

###################################################################################
# Constants to train GAN
###################################################################################

# Dimentions for GAN
INPUT_DIM = 5 # Output also
FEATURES = [
    'lrl',
    'log_dh',
    'delta_c',
    'delta_o',
    'log_volume'
]

LATENT_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2

# Number of time steps
WINDOW_SIZE = STOCK_INTERVALS_PER_DAY

BATCH_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Penalty for incorrect ratio between open, high, low, close
LAMBDA = 0.5

# Number of training sessions for Discriminator and Generator in one epoch
TRAIN_G = 1
TRAIN_D = 1


