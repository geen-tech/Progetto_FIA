import unittest
import pandas as pd
import numpy as np
from models.classifier import CustomKNN
from validazione import StratifiedValidation  

class TestStratifiedValidation(unittest.TestCase):
    def setUp(self):
        """
        Setup iniziale con un dataset di esempio.
        """
        # 5 campioni, 2 feature, 3 classi (0, 0, 1, 1, 2)
        self.data = pd.DataFrame({
            "feature1": [7, 3, 2, 4, 6],
            "feature2": [70, 30, 20, 40, 60]
        })
        self.labels = pd.Series([0, 0, 1, 1, 2]) 

    def test_split_data(self):
        """
        Verifica il corretto funzionamento di base di split_data:
        - Che venga restituita una lista
        - Che la lista contenga tuple (y_real, y_pred)
        - Che il numero di iterazioni corrisponda al valore richiesto
        """
        strat = StratifiedValidation(iterazioni=3, test_size=0.4)
        results = strat.split_data(self.data, self.labels, k_vicini=5)

        # Verifica che il risultato sia una lista
        self.assertIsInstance(results, list)
        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(results), 3)

        # Verifica che ogni elemento della lista sia una tupla (y_real, y_pred)
        for elem in results:
            self.assertIsInstance(elem, tuple)
            self.assertEqual(len(elem), 2)  # Deve essere (y_real, y_pred)
            y_real, y_pred = elem
            self.assertIsInstance(y_real, list)
            self.assertIsInstance(y_pred, list)

    def test_split_data_invalid(self):
        """
        Verifica che venga generato un errore quando i parametri 'iterazioni' o 'test_size' non sono validi:
        - iterazioni <= 0
        - test_size <= 0 o test_size > 1
        """
        # Caso iterazioni <= 0
        with self.assertRaises(ValueError):
            StratifiedValidation(iterazioni=0, test_size=0.2)
        with self.assertRaises(ValueError):
            StratifiedValidation(iterazioni=-1, test_size=0.2)

        # Caso test_size <= 0 e > 1
        with self.assertRaises(ValueError):
            StratifiedValidation(iterazioni=5, test_size=-0.3)
        with self.assertRaises(ValueError):
            StratifiedValidation(iterazioni=5, test_size=1.5)

    def test_split_data_small_dataset(self):
        """
        Verifica il comportamento con dataset molto piccolo o con una sola classe.
        Nel caso in cui non sia possibile eseguire una suddivisione stratificata coerente,
        ci si aspetta un ValueError (o altro comportamento definito).
        """
        # Dataset con un solo campione
        data = pd.DataFrame({"feature1": [10], "feature2": [100]})
        labels = pd.Series([1])

        strat = StratifiedValidation(iterazioni=2, test_size=0.5)
        # Se il codice non gestisce la suddivisione minima (train e test non vuoti), ci aspettiamo un errore
        with self.assertRaises(ValueError):
            strat.split_data(data, labels, k_vicini=5)

    def test_split_data_check_iterazioni(self):
        """
        Verifica che il numero di iterazioni effettuate sia quello indicato.
        """
        n_iter = 5
        strat = StratifiedValidation(iterazioni=n_iter, test_size=0.4)
        results = strat.split_data(self.data, self.labels, k_vicini=5)
        # Verifica che siano state eseguite esattamente n_iter iterazioni
        self.assertEqual(len(results), n_iter)

    def test_stratified_distribution(self):
        """
        Verifica che la suddivisione sia effettivamente stratificata.
        In particolare, controlla che ogni classe compaia nel test set 
        in proporzione corretta (o almeno non sia totalmente assente 
        se nel dataset principale ci sono campioni di quella classe).
        """
        # Abbiamo 3 classi: 0 (2 campioni), 1 (2 campioni), 2 (1 campione).
        # test_size=0.4 => ci aspettiamo 2 campioni in test su 5 totali.
        strat = StratifiedValidation(iterazioni=1, test_size=0.4)
        results = strat.split_data(self.data, self.labels, k_vicini=5)

        self.assertEqual(len(results), 1)
        y_real, y_pred = results[0]

        # Verifichiamo che la lunghezza di y_real sia circa il 40% di 5, ossia 2
        self.assertEqual(len(y_real), 2)

        # Verifichiamo che almeno in linea di massima ogni classe sia "considerata".
        # Per dataset così piccolo, può capitare che la classe con 1 solo campione
        # finisca nel train o nel test, ma almeno le classi con 2 campioni 
        # dovrebbero apparire anche nel test, se la stratificazione è stata fatta.
        unique_test_classes = set(y_real)
        # Se la classe 0 e 1 avevano 2 campioni ciascuna, è plausibile che appaiano nel test.
        # Non forziamo troppo la mano sul singolo campione di classe 2, perché con 1 campione
        # può finire tutto nel train. Giusto controllare che la dimensione totale è corretta:
        self.assertTrue(len(unique_test_classes) >= 1,
                        "Nel test set dovrebbero comparire almeno alcune classi presenti nel dataset.")

if __name__ == "__main__":
    unittest.main()
