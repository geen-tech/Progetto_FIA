import pandas as pd
import numpy as np
from collections import Counter
import random

class CustomKNN:
    def __init__(self, k=3):
        """
        Inizializza il classificatore KNN con un valore di k.
        """
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Addestra il classificatore memorizzando i dati di addestramento.

        Args:
            X (pd.DataFrame): I dati di addestramento.
            y (pd.Series): Le etichette di addestramento.
        """
        self.data = X
        self.labels = y

    def _euclidean_distance(self, point1, point2):
        """
        Calcola la distanza euclidea tra due punti nel loro spazio n-dimensionale.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, points):
        """
        Prevede la classe per uno o più punti basandosi sui dati di addestramento.

        Args:
            points (pd.Series or pd.DataFrame): Punto o DataFrame di punti da classificare.

        Returns:
            list: Lista delle etichette predette per i punti.
        """
        if self.data is None or self.labels is None:
            raise ValueError("Il classificatore non è stato addestrato. Esegui 'fit' prima di usare 'predict'.")
        
        # Assicurati che l'input sia un DataFrame o una Series
        if isinstance(points, pd.DataFrame):
            # Se l'input è un DataFrame, itera su ogni riga
            predictions = points.apply(self._predict_single, axis=1)
        elif isinstance(points, pd.Series):
            # Se l'input è una Series, predici per quel singolo punto
            predictions = [self._predict_single(points)]
        else:
            raise ValueError("L'input deve essere un pd.Series o pd.DataFrame.")
        
        return predictions.tolist()

    def _predict_single(self, point: pd.Series) -> int:
        """
        Predice la classe di un singolo punto basato sui dati di addestramento.

        Args:
            point (pd.Series): Un singolo punto da classificare.

        Returns:
            int: L'etichetta predetta per il punto.
        """
        # Calcola le distanze tra il punto e tutti gli altri dati registrati
        distances = self.data.apply(lambda row: self._euclidean_distance(row.values, point.values), axis=1)
        
        # Seleziona gli indici dei k vicini più prossimi
        nearest_neighbors = distances.nsmallest(self.k).index
        
        # Conta le occorrenze delle etichette dei vicini più vicini
        nearest_labels = self.labels.loc[nearest_neighbors]
        label_count = Counter(nearest_labels)
        most_common = label_count.most_common()
        max_count = most_common[0][1]  # Frequenza maggiore tra le etichette

        # Se ci sono più etichette con la stessa frequenza, sceglie casualmente
        tied_classes = [label for label, count in most_common if count == max_count]
        if len(tied_classes) > 1:
            return random.choice(tied_classes)
        else:
            return tied_classes[0]


