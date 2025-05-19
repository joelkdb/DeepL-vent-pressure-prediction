import torch
from torch.utils.data import Dataset
from src.utils.sequence_creator import SequenceCreator

class VentilatorDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, sequence_length, features, target_column):
        """
        Args:
            dataframe: Le pandas DataFrame contenant les données
            sequence_length: Longueur des séquences temporelles
            features: Liste des colonnes à utiliser comme features
            target_column: Nom de la colonne cible (pression)
        """
        self.df = dataframe
        self.sequence_length = sequence_length
        self.features = features
        self.target_column = target_column
        
        # Identifier les identifiants uniques de séries temporelles
        self.unique_series = self.df['breath_id'].unique()
        # Préparation de la séquence
        self.X, self.y = self.prepare_sequence_data()
    
    # Application aux données du ventilateur
    def prepare_sequence_data(self):
        sequence_features, sequence_targets = [], []
        
        for breath_id in self.unique_series:
            breath_data = self.df[self.df['breath_id'] == breath_id]
            features = breath_data[self.features].values
            targets = breath_data[self.target_column].values
            
            feature_tensor = torch.FloatTensor(features)
            target_tensor = torch.FloatTensor(targets)
            
            X_seq, y_seq = SequenceCreator.create_sequences(feature_tensor, target_tensor, self.seq_length)
            sequence_features.append(X_seq)
            sequence_targets.append(y_seq)
        
        return torch.cat(sequence_features), torch.cat(sequence_targets) 
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]