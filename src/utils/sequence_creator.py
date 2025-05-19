import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np

class SequenceCreator:
    @staticmethod
    def create_sequences(data, targets, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = targets[i+seq_length]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)