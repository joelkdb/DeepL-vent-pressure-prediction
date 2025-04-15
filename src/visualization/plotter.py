import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

class VentilatorVisualizer:
    """
    Classe pour visualiser les données
    """
    
    def __init__(self, output_dir='output'):
        """
        Initialise le visualiseur
        
        Args:
            output_dir: Répertoire où sauvegarder les visualisations
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_series_by_feature(self, df, feature, n_samples=5, save=True):
        """
        Visualise les séries temporelles regroupées par une caractéristique
        
        Args:
            df: DataFrame contenant les données
            feature: Caractéristique pour le regroupement (ex: 'R', 'C')
            n_samples: Nombre d'exemples à visualiser par valeur
            save: Sauvegarder la figure ou non
        """
        unique_values = df[feature].unique()
        plt.figure(figsize=(15, 10))
        
        for i, value in enumerate(unique_values):
            # Sélectionner quelques exemples aléatoires pour chaque valeur du paramètre
            breath_ids = df[df[feature] == value]['breath_id'].unique()
            selected_ids = np.random.choice(breath_ids, min(n_samples, len(breath_ids)), replace=False)
            
            plt.subplot(len(unique_values), 1, i+1)
            for breath_id in selected_ids:
                subset = df[df['breath_id'] == breath_id]
                plt.plot(subset['time_step'], subset['pressure'], label=f'breath_id={breath_id}')
            
            plt.title(f'{feature} = {value} (Échantillons aléatoires)')
            plt.xlabel('Time Step')
            plt.ylabel('Pressure')
            plt.grid(True)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/pressure_by_{feature}.png')
        plt.show()
    
    def plot_input_output_signals(self, df, n_samples=5, save=True):
        """
        Visualise les signaux d'entrée et de sortie
        
        Args:
            df: DataFrame contenant les données
            n_samples: Nombre d'exemples à visualiser
            save: Sauvegarder la figure ou non
        """
        breath_ids = df['breath_id'].unique()
        selected_ids = np.random.choice(breath_ids, min(n_samples, len(breath_ids)), replace=False)
        
        plt.figure(figsize=(15, 12))
        for i, breath_id in enumerate(selected_ids):
            subset = df[df['breath_id'] == breath_id]
            
            plt.subplot(n_samples, 1, i+1)
            plt.plot(subset['time_step'], subset['u_in'], label='u_in (entrée)', color='blue')
            plt.plot(subset['time_step'], subset['pressure'], label='pressure (sortie)', color='red')
            plt.title(f'ID: {breath_id}, R={subset["R"].iloc[0]}, C={subset["C"].iloc[0]}')
            plt.ylabel('Valeur')
            plt.grid(True)
            plt.legend()
        
        plt.xlabel('Time Step')
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/input_output_signals.png')
        plt.show()
    
    def plot_breathing_phases(self, df, n_samples=3, save=True):
        """
        Visualise les points de pression selon u_in
        
        Args:
            df: DataFrame contenant les données
            n_samples: Nombre d'exemples à visualiser
            save: Sauvegarder la figure ou non
        """
        breath_ids = df['breath_id'].unique()
        selected_ids = np.random.choice(breath_ids, min(n_samples, len(breath_ids)), replace=False)
        
        plt.figure(figsize=(15, 10))
        for i, breath_id in enumerate(selected_ids):
            subset = df[df['breath_id'] == breath_id]
            
            plt.subplot(n_samples, 1, i+1)
            plt.plot(subset['time_step'], subset['pressure'], label='Pressure')
            
            # Colorer les phases d'inhalation et d'exhalation
            inhalation = subset[subset['u_in'] > 0]
            exhalation = subset[subset['u_in'] == 0]
            
            plt.scatter(inhalation['time_step'], inhalation['pressure'], 
                       color='green', label='u_in > 0', alpha=0.5)
            plt.scatter(exhalation['time_step'], exhalation['pressure'], 
                       color='red', label='u_in = 0', alpha=0.5)
            
            plt.title(f'Phases respiratoires - ID: {breath_id}, R={subset["R"].iloc[0]}, C={subset["C"].iloc[0]}')
            plt.xlabel('Time Step')
            plt.ylabel('Pressure')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/breathing_phases.png')
        plt.show()
    
    def plot_breathing_phases_with_uout(self, df, n_samples=3, save=True):
        """
        Visualise les phases de respiration (inhalation, plateau, exhalation) 
        en utilisant à la fois u_in et u_out.

        Args:
            df: DataFrame contenant les données
            n_samples: Nombre d'exemples à visualiser
            save: Sauvegarder la figure ou non
        """
        breath_ids = df['breath_id'].unique()
        selected_ids = np.random.choice(breath_ids, min(n_samples, len(breath_ids)), replace=False)

        plt.figure(figsize=(15, 4 * n_samples))
        
        for i, breath_id in enumerate(selected_ids):
            subset = df[df['breath_id'] == breath_id].copy()
            time = subset['time_step']
            pressure = subset['pressure']

            # Déterminer les phases
            conditions = [
                (subset['u_in'] > 0) & (subset['u_out'] == 0),
                (subset['u_in'] == 0) & (subset['u_out'] == 0),
                (subset['u_out'] == 1)
            ]
            choices = ['Inhalation', 'Plateau', 'Exhalation']
            subset['phase'] = np.select(conditions, choices, default='Unknown')

            plt.subplot(n_samples, 1, i + 1)
            plt.plot(time, pressure, color='gray', label='Pressure')

            # Affichage des phases en couleur
            colors = {'Inhalation': 'green', 'Plateau': 'blue', 'Exhalation': 'red'}
            for phase, color in colors.items():
                phase_data = subset[subset['phase'] == phase]
                plt.scatter(phase_data['time_step'], phase_data['pressure'],
                            color=color, label=phase, alpha=0.5)

            plt.title(f'Respiratory Phases - ID: {breath_id}, R={subset["R"].iloc[0]}, C={subset["C"].iloc[0]}')
            plt.xlabel('Time Step')
            plt.ylabel('Pressure')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/breathing_phases_with_uout.png')
        plt.show()

    
    def plot_correlation_matrix(self, df, title='Matrice de corrélation', save=True):
        """
        Visualise la matrice de corrélation
        
        Args:
            df: DataFrame contenant les données
            title: Titre du graphique
            save: Sauvegarder la figure ou non
        """
        plt.figure(figsize=(12, 10))
        corr = df.select_dtypes(include=[np.number]).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/{title.lower().replace(" ", "_")}.png')
        plt.show()
        
    def plot_correlation_matrix_pearson(self, df, title='Matrice de corrélation', save=True):
        """
        Visualise la matrice de corrélation
        
        Args:
            df: DataFrame contenant les données
            title: Titre du graphique
            save: Sauvegarder la figure ou non
        """
        plt.figure(figsize=(12, 10))
        corr_matrix = df.corr(method='pearson')
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/{title.lower().replace(" ", "_")}.png')
        plt.show()
    
    def plot_feature_relationships(self, agg_df, save=True):
        """
        Visualise les relations entre caractéristiques
        
        Args:
            agg_df: DataFrame agrégé contenant les données
            save: Sauvegarder la figure ou non
        """
        plt.figure(figsize=(15, 10))

        # Relation entre R et les statistiques de pression
        plt.subplot(2, 2, 1)
        sns.boxplot(x='R_first', y='pressure_max', data=agg_df)
        plt.title('Impact de R sur la pression maximale')
        plt.xlabel('Résistance (R)')
        plt.ylabel('Pression maximale')

        plt.subplot(2, 2, 2)
        sns.boxplot(x='C_first', y='pressure_max', data=agg_df)
        plt.title('Impact de C sur la pression maximale')
        plt.xlabel('Compliance (C)')
        plt.ylabel('Pression maximale')

        plt.subplot(2, 2, 3)
        sns.scatterplot(x='u_in_sum', y='pressure_mean', hue='R_first', data=agg_df)
        plt.title('Relation entre volume d\'air total et pression moyenne par R')
        plt.xlabel('Somme des entrées u_in')
        plt.ylabel('Pression moyenne')

        plt.subplot(2, 2, 4)
        sns.scatterplot(x='u_in_sum', y='pressure_mean', hue='C_first', data=agg_df)
        plt.title('Relation entre volume d\'air total et pression moyenne par C')
        plt.xlabel('Somme des entrées u_in')
        plt.ylabel('Pression moyenne')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/feature_relationships.png')
        plt.show()
        
    def plot_feature_relationships_raw(self, df, save=True):
        """
        Visualise les relations entre caractéristiques à partir des données brutes.

        Args:
            df: DataFrame brut contenant les données par time_step
            save: Sauvegarder la figure ou non
        """
        plt.figure(figsize=(15, 10))

        # 1. Impact de R sur la pression (distribution)
        plt.subplot(2, 2, 1)
        sns.boxplot(x='R', y='pressure', data=df)
        plt.title('Impact de R sur la pression')
        plt.xlabel('Résistance (R)')
        plt.ylabel('Pression')

        # 2. Impact de C sur la pression
        plt.subplot(2, 2, 2)
        sns.boxplot(x='C', y='pressure', data=df)
        plt.title('Impact de C sur la pression')
        plt.xlabel('Compliance (C)')
        plt.ylabel('Pression')

        # 3. u_in vs pressure par R
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='u_in', y='pressure', hue='R', data=df, alpha=0.3)
        plt.title('Relation entre u_in et pression (par R)')
        plt.xlabel('u_in')
        plt.ylabel('Pression')

        # 4. u_in vs pressure par C
        plt.subplot(2, 2, 4)
        sns.scatterplot(x='u_in', y='pressure', hue='C', data=df, alpha=0.3)
        plt.title('Relation entre u_in et pression (par C)')
        plt.xlabel('u_in')
        plt.ylabel('Pression')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/feature_relationships_raw.png')
        plt.show()
        
    def plot_feature_relationships_filtered(self, df, save=True):
        """
        Visualise les relations entre caractéristiques à partir des données brutes
        en filtrant les lignes où u_out == 1 (valve fermée).
        
        Args:
            df: DataFrame brut contenant les données par time_step
            save: Sauvegarder la figure ou non
        """
        # Ne garder que les lignes où u_out == 1
        df_filtered = df[df["u_out"] == 1]

        plt.figure(figsize=(15, 10))

        # 1. Impact de R sur la pression
        plt.subplot(2, 2, 1)
        sns.boxplot(x='R', y='pressure', data=df_filtered)
        plt.title('Impact de R sur la pression (u_out == 1)')
        plt.xlabel('Résistance (R)')
        plt.ylabel('Pression')

        # 2. Impact de C sur la pression
        plt.subplot(2, 2, 2)
        sns.boxplot(x='C', y='pressure', data=df_filtered)
        plt.title('Impact de C sur la pression (u_out == 1)')
        plt.xlabel('Compliance (C)')
        plt.ylabel('Pression')

        # 3. u_in vs pressure par R
        #plt.subplot(2, 2, 3)
        #sns.scatterplot(x='u_in', y='pressure', hue='R', data=df_filtered, alpha=0.3)
        #plt.title('u_in vs pression (par R, u_out == 1)')
        #plt.xlabel('u_in')
        #plt.ylabel('Pression')

        # 4. u_in vs pressure par C
        #plt.subplot(2, 2, 4)
        #sns.scatterplot(x='u_in', y='pressure', hue='C', data=df_filtered, alpha=0.3)
        #plt.title('u_in vs pression (par C, u_out == 1)')
        #plt.xlabel('u_in')
        #plt.ylabel('Pression')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/feature_relationships_filtered.png')
        plt.show()


    
    def plot_dim_reduction(self, results, labels_r, labels_c, method='t-SNE', save=True):
        """
        Visualise les résultats de réduction de dimension
        
        Args:
            results: Résultat de la réduction de dimension
            labels_r: Étiquettes R correspondantes
            labels_c: Étiquettes C correspondantes
            method: Méthode utilisée ('t-SNE' ou 'UMAP')
            save: Sauvegarder la figure ou non
        """
        plt.figure(figsize=(20, 8))

        plt.subplot(1, 2, 1)
        scatter = plt.scatter(results[:, 0], results[:, 1], c=labels_r, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Résistance (R)')
        plt.title(f'{method} des séquences de pression colorées par R')
        plt.xlabel(f'{method} 1')
        plt.ylabel(f'{method} 2')

        plt.subplot(1, 2, 2)
        scatter = plt.scatter(results[:, 0], results[:, 1], c=labels_c, cmap='plasma', alpha=0.7)
        plt.colorbar(scatter, label='Compliance (C)')
        plt.title(f'{method} des séquences de pression colorées par C')
        plt.xlabel(f'{method} 1')
        plt.ylabel(f'{method} 2')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/{method.lower()}_visualization.png')
        plt.show()
        
    def plot_enhanced_features(self, df, n_samples=3, save=True):
        """
        Visualise les caractéristiques dérivées pour quelques exemples
        
        Args:
            df: DataFrame avec caractéristiques dérivées
            n_samples: Nombre d'exemples à visualiser
            save: Sauvegarder la figure ou non
        """
        breath_ids = df['breath_id'].unique()
        selected_ids = np.random.choice(breath_ids, min(n_samples, len(breath_ids)), replace=False)
        
        for breath_id in selected_ids:
            subset = df[df['breath_id'] == breath_id]
            
            plt.figure(figsize=(15, 12))
            
            # Visualiser la pression et son entrée
            plt.subplot(4, 1, 1)
            plt.plot(subset['time_step'], subset['pressure'], label='Pressure')
            plt.plot(subset['time_step'], subset['u_in'], label='u_in')
            plt.title(f'Caractéristiques du cycle - ID: {breath_id}, R={subset["R"].iloc[0]}, C={subset["C"].iloc[0]}')
            plt.grid(True)
            plt.legend()
            
            # Visualiser la dérivée première
            plt.subplot(4, 1, 2)
            plt.plot(subset['time_step'], subset['pressure_delta'], color='orange')
            plt.title('Dérivée première de la pression (taux de changement)')
            plt.grid(True)
            
            # Visualiser la dérivée seconde
            plt.subplot(4, 1, 3)
            plt.plot(subset['time_step'], subset['pressure_delta2'], color='green')
            plt.title('Dérivée seconde de la pression (accélération)')
            plt.grid(True)
            
            # Visualiser le volume cumulatif d'air
            plt.subplot(4, 1, 4)
            plt.plot(subset['time_step'], subset['u_in_cumsum'], color='purple')
            plt.title('Volume d\'air cumulatif (cumsum de u_in)')
            plt.grid(True)
            
            plt.xlabel('Time Step')
            plt.tight_layout()
            if save:
                plt.savefig(f'{self.output_dir}/enhanced_features_breath_{breath_id}.png')
            plt.show()
            
    def plot_multivariate_analysis(self, enhanced_agg, save=True):
        """
        Visualise l'analyse multivariée des caractéristiques dérivées
        
        Args:
            enhanced_agg: DataFrame avec caractéristiques agrégées dérivées
            save: Sauvegarder la figure ou non
        """
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        sns.scatterplot(x='pressure_max', y='pressure_delta_max', hue='R_first', 
                        size='C_first', sizes=(20, 200), data=enhanced_agg)
        plt.title('Relation entre pression maximale et taux de changement maximal')
        plt.xlabel('Pression maximale')
        plt.ylabel('Taux de changement maximal')

        plt.subplot(2, 2, 2)
        sns.scatterplot(x='u_in_cumsum_max', y='pressure_max', 
                        hue='R_first', size='C_first', sizes=(20, 200), data=enhanced_agg)
        plt.title('Volume d\'air total vs pression maximale')
        plt.xlabel('Volume d\'air total')
        plt.ylabel('Pression maximale')

        plt.subplot(2, 2, 3)
        sns.scatterplot(x='pressure_delta_std', y='pressure_delta2_std', 
                        hue='R_first', size='C_first', sizes=(20, 200), data=enhanced_agg)
        plt.title('Variabilité des dérivées première et seconde')
        plt.xlabel('Écart-type de la dérivée première')
        plt.ylabel('Écart-type de la dérivée seconde')

        plt.subplot(2, 2, 4)
        sns.scatterplot(x='pressure_mean', y='pressure_std', 
                        hue='R_first', size='C_first', sizes=(20, 200), data=enhanced_agg)
        plt.title('Moyenne vs écart-type de la pression')
        plt.xlabel('Pression moyenne')
        plt.ylabel('Écart-type de la pression')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/multivariate_analysis.png')
        plt.show()