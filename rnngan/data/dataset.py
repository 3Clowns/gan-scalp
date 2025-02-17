import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        return torch.FloatTensor(x)