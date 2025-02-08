import random
import pandas as pd
import numpy as np
from collections import Counter

class CustomKNN:
    def __init__(self, k:int):
        """
        Costruttore della classe che imposta il numero di vicini da considerare.
        """
        self.k = k
        self.data = None
        self.labels = None

    def fit(self, data: pd.DataFrame, labels: pd.Series) -> None:
        """
        Salva i dati di riferimento per la classificazione.

        Args:
            data (pd.DataFrame): Il dataset che contiene le caratteristiche.
            labels (pd.Series): Le etichette associate ai dati.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("I dati devono essere sotto forma di DataFrame di Pandas.")
        if not isinstance(labels, pd.Series):
            raise ValueError("Le etichette devono essere fornite come Serie di Pandas.")
        
        self.data = data
        self.labels = labels

    def _euclidean_distance(self, point1, point2):
        """
        Calcola la distanza tra due punti nello spazio n-dimensionale.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, point: pd.Series) -> int:
        """
        Determina la categoria di un nuovo punto basandosi sui dati di riferimento.

        Args:
            point (pd.Series): Punto da classificare.

        Returns:
            int: Etichetta predetta per il punto.
        """
        if self.data is None or self.labels is None:
            raise ValueError("Il classificatore non è stato addestrato. Esegui 'fit' prima di usare 'predict'.")
        
        if not isinstance(point, pd.Series):
            raise ValueError("Il punto da classificare deve essere una Serie di Pandas.")
        
        # Calcola le distanze tra il punto e tutti gli altri dati registrati
        distances = self.data.apply(lambda row: self._euclidean_distance(row.values, point.values), axis=1)
        
        # Seleziona gli indici dei punti più vicini
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

    def predict_batch(self, points: pd.DataFrame) -> pd.Series:
        """
        Classifica un insieme di punti contemporaneamente.

        Args:
            points (pd.DataFrame): Un insieme di punti da classificare.

        Returns:
            pd.Series: Etichette predette per ciascun punto del dataset.
        """
        if not isinstance(points, pd.DataFrame):
            raise ValueError("I dati in ingresso devono essere un DataFrame di Pandas.")
        
        predictions = points.apply(self.predict, axis=1)
        return predictions



