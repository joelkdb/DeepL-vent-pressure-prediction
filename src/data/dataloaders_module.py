import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.data.breath_sampler import StratifiedBreathSampler
from src.data.dataset2 import VentilatorDataset
from src.data.normalizer import VentilatorDataNormalizer

class VentilatorDataModule:
    def __init__(self, train_df, val_df, test_df, features, target_column, seq_length, batch_size):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.features = features
        self.target_column = target_column
        self.seq_length = seq_length
        self.batch_size = batch_size
    
    def setup(self):
        # Normalisation sur le train uniquement
        self.normalizer = VentilatorDataNormalizer(self.train_df, self.features)
        train_df_norm = self.normalizer.normalize(self.train_df)
        val_df_norm = self.normalizer.normalize(self.val_df)
        test_df_norm = self.normalizer.normalize(self.test_df)
        
        # Création des datasets
        self.train_dataset = VentilatorDataset(train_df_norm, self.seq_length, self.features, self.target_column)
        self.val_dataset = VentilatorDataset(val_df_norm, self.seq_length, self.features, self.target_column)
        self.test_dataset = VentilatorDataset(test_df_norm, self.seq_length, self.features, self.target_column)
    
    def get_dataloaders(self, use_stratified_sampler=False):
        sampler = StratifiedBreathSampler.create(self.train_dataset) if use_stratified_sampler else None
        
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=not bool(sampler), 
            sampler=sampler,
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader