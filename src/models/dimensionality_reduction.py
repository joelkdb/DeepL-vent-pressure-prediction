from sklearn.manifold import TSNE
import umap
import numpy as np
from tqdm import tqdm

class DimensionalityReducer:
    """
    Classe pour réduire la dimensionnalité des données
    """
    
    def __init__(self):
        """
        Initialise le réducteur de dimensionnalité
        """
        pass
    
    def apply_tsne(self, sequences, random_state=42, perplexity=30):
        """
        Applique t-SNE pour réduire la dimensionnalité
        
        Args:
            sequences: Séquences de données à réduire
            random_state: État aléatoire pour reproduire les résultats
            perplexity: Paramètre de perplexité pour t-SNE
            
        Returns:
            numpy.ndarray: Résultats de t-SNE en 2D
        """
        print("Exécution de t-SNE...")
        tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
        return tsne.fit_transform(sequences)
    
    def apply_umap(self, sequences, n_neighbors=30, random_state=42):
        """
        Applique UMAP pour réduire la dimensionnalité
        
        Args:
            sequences: Séquences de données à réduire
            random_state: État aléatoire pour reproduire les résultats
            
        Returns:
            numpy.ndarray: Résultats de UMAP en 2D
        """
        print("Exécution de UMAP...")
        reducer = umap.UMAP(n_neighbors, random_state=random_state)
        return reducer.fit_transform(sequences)