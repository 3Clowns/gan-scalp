import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_len: int, features: list):
        self.data = data[features].values
        self.features = features
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        return torch.FloatTensor(x)