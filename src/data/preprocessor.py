import pandas as pd
import numpy as np
from tqdm import tqdm

class VentilatorDataPreprocessor:
    """
    Classe pour le prétraitement des données
    """
    
    def analyze_dataset(self, df, title):
        """
        Fournit une analyse statistique d'un dataframe
        
        Args:
            df: DataFrame à analyser
            title: Titre de l'analyse
            
        Returns:
            dict: Dictionnaire contenant les statistiques principales
        """
        stats = {
            'shape': df.shape,
            'head': df.head(),
            'info': df.info(),
            'describe': df.describe(),
            'null_counts': df.isnull().sum(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        
        print(f"\n{title} - Aperçu des données:")
        print(df.head())
        
        print(f"\n{title} - Informations sur les types de données:")
        print(df.info())
        
        print(f"\n{title} - Statistiques descriptives:")
        print(df.describe())
        
        print(f"\n{title} - Valeurs manquantes:")
        print(df.isnull().sum())
        
        print(f"\n{title} - Nombre d'échantillons uniques par colonne:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} valeurs uniques")
            
        return stats
    
    def analyze_time_series_structure(self, df):
        """
        Analyse la structure des séries temporelles
        
        Args:
            df: DataFrame contenant les données de séries temporelles
            
        Returns:
            dict: Statistiques sur la structure des séries temporelles
        """
        breath_counts = df.groupby('breath_id').size()
        stats = {
            'num_series': df['id'].nunique(),
            'num_breaths': df['breath_id'].nunique(),
            'observations_per_breath': {
                'min': breath_counts.min(),
                'max': breath_counts.max(),
                'mean': breath_counts.mean(),
                'median': breath_counts.median()
            }
        }
        
        print("\nAnalyse des identifiants uniques:")
        print(f"Nombre de séries temporelles uniques: {stats['num_series']}")
        print(f"Nombre de respirations uniques: {stats['num_breaths']}")
        
        print("\nDistribution du nombre d'observations par respiration:")
        print(f"Min: {stats['observations_per_breath']['min']}")
        print(f"Max: {stats['observations_per_breath']['max']}")
        print(f"Moyenne: {stats['observations_per_breath']['mean']:.2f}")
        print(f"Médiane: {stats['observations_per_breath']['median']}")
        
        return stats
        
    def analyze_categorical_vars(self, df):
        """
        Analyse les variables catégorielles
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            dict: Distribution des variables catégorielles
        """
        cat_dist = {}
        for col in ['R', 'C']:
            dist = df[col].value_counts(normalize=True) * 100
            cat_dist[col] = dist
            
            print(f"\nDistribution de {col}:")
            print(dist)
            
        return cat_dist
    
    def add_derived_features(self, df):
        """
        Ajoute des caractéristiques dérivées aux données
        
        Args:
            df: DataFrame contenant les données brutes
            
        Returns:
            DataFrame: DataFrame avec caractéristiques dérivées ajoutées
        """
        result_df = df.copy()
        
        # Calculer les dérivées par respiration
        for breath_id in tqdm(df['breath_id'].unique(), desc="Calcul des caractéristiques dérivées"):
            mask = df['breath_id'] == breath_id
            subset = df[mask].copy()
            
            # Dérivée première de la pression (taux de changement) -> difference consécutive
            result_df.loc[mask, 'pressure_delta'] = subset['pressure'].diff().fillna(0)
            
            # Dérivée seconde (accélération)
            result_df.loc[mask, 'pressure_delta2'] = result_df['pressure_delta'].diff().fillna(0)
            
            # Cumul de u_in pour estimer le volume total d'air insufflé au fil du temps
            result_df.loc[mask, 'u_in_cumsum'] = subset['u_in'].cumsum()
        
        return result_df
    
    def create_ventilator_features(self, df):
        """
            Crée des caractéristiques supplémentaires pour la prédiction
        Args:
            df (_type_): DataFrame contenant la données brutes
        """
        df_new = df.copy()
        
        # Taux de changement -> différence consécutive
        df_new['pressure_delta'] = df_new.grouby('breath_id')['pressure'].diff().fillna(0)
    
        # Calcul du temps écoulé depuis le début du cycle de respiration
        df_new['time_step_percent'] = df_new['time_step'] / df_new.groupby('breath_id')['time_step'].transform('max')
        
        # Caractéristiques de la dynamique de respiration
        df_new['u_in_cumsum'] = df_new.groupby('breath_id')['u_in'].cumsum()
        df_new['u_in_diff'] = df_new.groupby('breath_id')['u_in'].diff().fillna(0)
        
        # Caractéristiques d'interaction
        df_new['u_in_times_R'] = df_new['u_in'] * df_new['R']
        df_new['u_in_times_C'] = df_new['u_in'] * df_new['C']
        
        return df_new
    
    def aggregate_features(self, df):
        """
        Calcule des statistiques agrégées pour chaque cycle respiratoire
        
        Args:
            df: DataFrame avec les données par pas de temps
            
        Returns:
            DataFrame: DataFrame avec caractéristiques agrégées par cycle respiratoire
        """
        agg_df = df.groupby('breath_id').agg({
            'pressure': ['mean', 'std', 'min', 'max', 'median'],
            'u_in': ['mean', 'std', 'min', 'max', 'sum'],
            'R': 'first',
            'C': 'first'
        })
        
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        return agg_df.reset_index()
    
    def aggregate_enhanced_features(self, df):
        """
        Calcule des statistiques agrégées incluant les caractéristiques dérivées
        
        Args:
            df: DataFrame avec les caractéristiques dérivées
            
        Returns:
            DataFrame: DataFrame avec caractéristiques améliorées agrégées
        """
        agg_df = df.groupby('breath_id').agg({
            'pressure': ['mean', 'std', 'min', 'max'],
            'pressure_delta': ['mean', 'std', 'min', 'max'],
            'pressure_delta2': ['mean', 'std', 'min', 'max'],
            'u_in_cumsum': ['max'],
            'R': 'first',
            'C': 'first'
        })
        
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        return agg_df.reset_index()
    
    def prepare_data_for_dim_reduction(self, df, sample_size=1000):
        """
        Prépare les données pour la réduction de dimension
        
        Args:
            df: DataFrame contenant les données complètes
            sample_size: Nombre de cycles respiratoires à échantillonner
            
        Returns:
            tuple: (sequences, labels_r, labels_c)
        """
        # Échantillonner des cycles respiratoires
        breath_ids = df['breath_id'].unique()
        selected_ids = np.random.choice(breath_ids, min(sample_size, len(breath_ids)), replace=False)
        
        # Pour chaque cycle, extraire les séquences de pression
        sequences = []
        labels_r = []
        labels_c = []
        
        for breath_id in tqdm(selected_ids, desc="Préparation des données pour réduction de dimension"):
            subset = df[df['breath_id'] == breath_id]
            sequence = subset['pressure'].values
            
            # Enregistrer les séquences et étiquettes
            sequences.append(sequence)
            labels_r.append(subset['R'].iloc[0])
            labels_c.append(subset['C'].iloc[0])
        
        # Convertir en array numpy
        return np.array(sequences), np.array(labels_r), np.array(labels_c)