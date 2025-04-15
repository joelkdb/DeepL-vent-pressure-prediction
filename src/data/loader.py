import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

class VentilatorDataLoader:
    """
    Classe pour charger les données du challenge Ventilator Pressure Prediction
    """

    def __init__(self, data_dir='data'):
        """
        Initialise le chargeur de données
        
        Args:
            data_dir: Répertoire où stocker/chercher les données
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_data(self):
        """
        Charge les données depuis Kaggle ou les télécharge si nécessaire
        
        Returns:
            tuple: (train_df, test_df, submission_df)
        """
        
        # Télécharger les données
        self._download_data()

        # Charger les données téléchargées
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
        submission_df = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))
        print("Données téléchargées et chargées avec succès.")
        
        return train_df, test_df, submission_df

    def _download_data(self):
        print("Téléchargement des données avec l'API Kaggle...")
        zip_path = os.path.join(self.data_dir, 'ventilator-pressure-prediction.zip')
        train_csv_path = os.path.join(self.data_dir, 'train.csv')
        test_csv_path = os.path.join(self.data_dir, 'test.csv')

        if os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
            print("Les fichiers de données existent déjà, pas de téléchargement.")
            return

        # Initier l’API
        api = KaggleApi()
        api.authenticate()

        # Télécharger le dataset dans le dossier data
        api.competition_download_files('ventilator-pressure-prediction', path=self.data_dir)

        # Décompresser le fichier
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        print("Téléchargement et extraction terminés.")
