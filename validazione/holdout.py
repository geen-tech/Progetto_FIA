import numpy as np
import pandas as pd
from .validation import ValidationProcess
from models.classifier import CustomKNN


class Holdout(ValidationProcess):
    
    # Classe che gestisce il processo di validazione Holdout per il modello KNN
    
    def __init__(self, test_size):
        """
        Inizializza la strategia Holdout con una dimensione del set di test.

        Args:
            test_size (float): Percentuale del dataset da utilizzare come test (compreso tra 0 e 1).

        Raises:
            ValueError: Se `test_size` non è compreso tra 0 e 1.
        """
        # Verifica che 'test_size' sia compreso tra 0 e 1
        if not (0 < test_size <= 1):
            raise ValueError("Il test size deve essere compreso tra 0 e 1")
        
        self.test_size = test_size

    def split_data(self, data: pd.DataFrame, labels: pd.Series, k_vicini: int) -> list[tuple[list[int], list[int], list[float]]]:
        n_samples = len(data)
        n_test = int(n_samples * self.test_size)  # Calcola il numero di campioni nel test set

        # Verifica che il numero di campioni nel set di test non sia uguale al totale o maggiore del totale
        if n_test == 0:
            raise ValueError("Il set di test è vuoto. Aumenta il valore di test_size")
        if n_test == n_samples:
            raise ValueError("Il set di test è troppo grande. Riduci il valore di test_size per avere un set di training valido") 

        # Mescolare in modo casuale l'ordine degli esempi nel dataset (shuffle)
        shuffled_indices = np.random.permutation(n_samples)
        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]
        
        # Divisione del dataframe in base alle percentuali richieste
        train_data, test_data = data.iloc[train_indices], data.iloc[test_indices]
        train_labels, test_labels = labels.iloc[train_indices], labels.iloc[test_indices]
        
        # Addestramento e predizione
        KNN = CustomKNN(k_vicini) 
        KNN.fit(train_data, train_labels)
        
        # Predizioni delle etichette e probabilità per ogni esempio nel test set
        y_pred = KNN.predict_batch(test_data)
        
        # Supponiamo che la classe positiva sia 4.0, estraiamo la probabilità per la classe '2.0'
        probabilities = [KNN.predict_proba(pd.Series(point))[4.0] for point in test_data.itertuples(index=False)]

        # Costruisce la lista di tuple (y_real, y_pred, probabilità)
        risultati = [(test_labels.tolist(), y_pred.tolist(), probabilities)]

        return risultati