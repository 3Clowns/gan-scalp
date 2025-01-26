import pandas as pd
import torch

###################################################################################
# Stocks universe definition
###################################################################################

# Define number of tickers in universe
N_ASSETS = 5
# Data sampling interval (15 minutes)
INTERVAL_MINUTES = 15
# Define number of trading days in a year
WORK_DAYS_PER_YEAR = 252
# Stock Market Hours (10:00 AM - 6:40 PM MSK)
STOCK_MARKET_HOURS = 8 * 60 + 40 
# Calculate 15-minute intervals per day/week/year
STOCK_INTERVALS_PER_DAY = STOCK_MARKET_HOURS // INTERVAL_MINUTES  # 35
STOCK_INTERVALS_PER_WEEK = STOCK_INTERVALS_PER_DAY * 5  # 175
STOCK_INTERVALS_PER_YEAR = STOCK_INTERVALS_PER_DAY * WORK_DAYS_PER_YEAR  # 8,750

###################################################################################
# Constants to train GAN
###################################################################################

LATENT_DIM = 100
OUTPUT_DIM = 1

WINDOW_SIZE = STOCK_INTERVALS_PER_WEEK

BATCH_SIZE = 256

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')