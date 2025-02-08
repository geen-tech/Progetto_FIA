import numpy as np
import pandas as pd
from .validation import ValidationProcess
from models.classifier import CustomKNN


class RandomSubsampling(ValidationProcess):
    
    # Classe che gestisce il processo di validazione Random Subsampling per il modello KNN.

    def __init__(self, test_size, iterazioni):
        """
        Inizializza la strategia Random Subsampling con una dimensione del set di test.

        Args:
            test_size (float): Percentuale del dataset da utilizzare come test (compreso tra 0 e 1).
            iterazioni (int): Numero di iterazioni K da svolgere sul dataframe (deve essere un numero positivo).

        Raises:
            ValueError: Se `test_size` non è compreso tra 0 e 1.
            ValueError: Se 'iterazioni' non è positivo.
        """

        # Verifica che 'iterazioni' sia un intero positivo
        if not (0 < iterazioni):
            raise ValueError("Il numero d'iterazioni deve essere un intero positivo")

        # Verifica che 'test_size' sia compreso tra 0 e 1
        if not (0 < test_size <= 1):
            raise ValueError("Il test size deve essere compreso tra 0 e 1")
        
        self.n_iterazioni = iterazioni
        self.test_size = test_size


    def split_data(self, data: pd.DataFrame, labels: pd.Series, k_vicini: int) -> list[tuple[list[int], list[int]]]:
        # Divide i dati casualmente in train e test
        risultati=[]
        n_campioni=len(data)
        n_test=int(n_campioni*self.test_size)

        for _ in range(self.n_iterazioni):
            n_campioni=len(data)
            n_test=int(n_campioni*self.test_size)

            # Verifica che il numero di campioni nel set di test non sia uguale al totale o maggiore del totale
            if n_test == 0:
                raise ValueError("Il set di test è vuoto. Aumenta il valore di test_size")
            if n_test == n_campioni:
                raise ValueError("Il set di test è troppo grande. Riduci il valore di test_size per avere un set di training valido") 

            # Mescolare in modo casuale l'ordine degli esempi nel dataset (shuffle)
            shuffled_indices = np.random.permutation(n_campioni)
            test_indici = shuffled_indices[:n_test]
            train_indici = shuffled_indices[n_test:]
            # Divisione dataframe
            train_data, test_data = data.iloc[train_indici], data.iloc[test_indici]
            train_labels, test_labels = labels.iloc[train_indici], labels.iloc[test_indici]
            # Processo/Predizione
            KNN = CustomKNN(k_vicini)
            KNN.fit(train_data, train_labels)
            pred = KNN.predict_batch(test_data) # Valori ottenuti
             # Costruisce la lista di tuple (y_real, y_pred)
            risultati.append((test_labels.tolist(), pred.tolist()))

        return risultati

