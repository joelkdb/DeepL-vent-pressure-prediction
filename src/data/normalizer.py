from sklearn.preprocessing import MinMaxScaler, StandardScaler

class VentilatorDataNormalizer:
    def __init__(self, df, features):
        self.features = features
        # Calculer statistiques sur l'ensemble d'entraînement uniquement
        self.mean = df[features].mean()
        self.std = df[features].std()
        
    def normalize(self, df):
        """Normalise les features du dataframe"""
        df_norm = df.copy()
        df_norm[self.features] = (df_norm[self.features] - self.mean) / self.std
        return df_norm
        
    def denormalize(self, df):
        """Dénormalise les données pour revenir à l'échelle d'origine"""
        df_denorm = df.copy()
        df_denorm[self.features] = df_denorm[self.features] * self.std + self.mean
        return df_denorm