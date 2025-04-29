import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class VentilatorFeatureEngineer:
    def __init__(self):
        self.minmax_scalers = {}
        self.std_scalers = {}
        
    def normalize_features(self, df: pd.DataFrame, columns: list, fit: bool = False) -> pd.DataFrame:
        """
        Normalisation Min-Max des colonnes spécifiées
        Appliquer fit=True uniquement sur le training set
        """
        df = df.copy()
        for col in columns:
            if fit:
                self.minmax_scalers[col] = MinMaxScaler().fit(df[[col]])
            if col in self.minmax_scalers:
                df[col] = self.minmax_scalers[col].transform(df[[col]]).flatten()
        return df

    def standardize_features(self, df: pd.DataFrame, columns: list, fit: bool = False) -> pd.DataFrame:
        """
        Standardisation (Z-score) des colonnes spécifiées
        Appliquer fit=True uniquement sur le training set
        """
        df = df.copy()
        for col in columns:
            if fit:
                self.std_scalers[col] = StandardScaler().fit(df[[col]])
            if col in self.std_scalers:
                df[col] = self.std_scalers[col].transform(df[[col]]).flatten()
        return df