import math
import numpy as np
import pandas as pd
import random
from .validation import ValidationProcess
from models.classifier import CustomKNN


class StratifiedValidation(ValidationProcess):

    def _init_(self, iterazioni, test_size):
        """
        :param iterazioni: Numero di split/training-testing da eseguire.
        :param test_size: Frazione di campioni da destinare al test (0 < test_size < 1).
        """
        # Verifica che 'iterazioni' sia un intero positivo
        if not (iterazioni > 0):
            raise ValueError("Il numero di iterazioni deve essere un intero positivo.")
        # Verifica che 'test_size' sia compreso fra 0 e 1 (esclusi)
        if not (0 < test_size < 1):
            raise ValueError("Il test size deve essere compreso tra 0 e 1 (esclusi).")

        self.n_iterazioni = iterazioni
        self.test_size = test_size

def split_data(self, data: pd.DataFrame, labels: pd.Series, k_vicini: int) -> list[tuple[list[int], list[int], list[float]]]:
    """
    Ritorna una lista di tuple (y_test, y_pred, probabilità).
    """
    n_samples = len(data)
    
    # Numero totale di campioni che andranno nel test set
    test_count = int(round(n_samples * self.test_size))
    
    # Se non possiamo creare train e test non vuoti, solleviamo eccezione
    if test_count == 0 or test_count == n_samples:
        raise ValueError("Impossibile creare train e test set non vuoti con questi parametri")
    
    risultati = []
    # Classi uniche presenti nelle label
    classes = labels.unique()

    for _ in range(self.n_iterazioni):
        # Raggruppiamo gli indici per classe in un dict {classe: [lista di indici]}
        class_indices = {c: [] for c in classes}
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        # Mescoliamo casualmente gli indici per ogni classe (per randomizzare il campionamento)
        for c in classes:
            random.shuffle(class_indices[c])
        
        # Calcoliamo quanti campioni per classe mettere nel test set (proporzione)
        # usando un approccio "fractions" + leftover
        freq = {c: len(class_indices[c]) for c in classes}  # numero di campioni per ciascuna classe
        class_test_counts = {}
        sum_base = 0
        fractions = []

        for c in classes:
            # test_c_float: numero "ideale" di campioni di classe c da mettere in test
            test_c_float = freq[c] * (test_count / float(n_samples))
            base = int(math.floor(test_c_float))  # parte intera
            class_test_counts[c] = base
            sum_base += base
            frac = test_c_float - base  # parte frazionaria
            fractions.append((c, frac))

        # leftover: quanti campioni aggiuntivi (o in meno) dobbiamo distribuire
        leftover = test_count - sum_base
        # Ordiniamo le classi in base alla parte frazionaria discendente
        fractions.sort(key=lambda x: x[1], reverse=True)

        # Assegniamo +1 ai primi leftover in fractions (se leftover > 0)
        # oppure togliamo 1 se leftover < 0
        i_fraction = 0
        while leftover != 0 and i_fraction < len(fractions):
            c, _ = fractions[i_fraction]
            if leftover > 0:
                # Aggiungiamo 1 finché leftover > 0
                if class_test_counts[c] < freq[c]:
                    class_test_counts[c] += 1
                    leftover -= 1
            else:
                # Togliamo 1 finché leftover < 0
                if class_test_counts[c] > 0:
                    class_test_counts[c] -= 1
                    leftover += 1
            i_fraction += 1
        
        # Ora class_test_counts[c] ci dice quanti campioni di classe c vanno nel test
        test_idx = []
        train_idx = []
        for c in classes:
            needed_for_test = class_test_counts[c]
            if needed_for_test > freq[c]:
                raise ValueError(
                    f"Richiesti {needed_for_test} campioni di classe {c} "
                    f"ma ne esistono solo {freq[c]}"
                )
            
            # I primi needed_for_test indici vanno in test, il resto in train
            test_idx_c = class_indices[c][:needed_for_test]
            train_idx_c = class_indices[c][needed_for_test:]
            
            test_idx.extend(test_idx_c)
            train_idx.extend(train_idx_c)
        
        # Controlliamo che non sia test set o train set vuoto
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise ValueError("Non è possibile creare train e test set non vuoti con i parametri specificati.")
        
        # Creazione dei set di train e test
        X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
        
        # Addestramento KNN
        knn = CustomKNN(k_vicini)
        knn.fit(X_train, y_train)
        
        # Predizione
        y_pred = knn.predict_batch(X_test)
        
        # Calcolare le probabilità per ciascun esempio nel test set
        probabilities = [knn.predict_proba(pd.Series(point)) [4.0] for point in X_test.itertuples(index=False)]

        
        # Aggiungiamo la tupla (y_test, y_pred, probabilità) ai risultati
        risultati.append((y_test.tolist(), y_pred.tolist(), probabilities))

    return risultati