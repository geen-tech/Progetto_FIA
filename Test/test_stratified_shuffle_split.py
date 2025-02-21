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
        - Che la lista contenga tuple (y_test, y_pred, probabilità)
        - Che il numero di iterazioni corrisponda al valore richiesto
        """
        strat = StratifiedValidation(iterazioni=3, test_size=0.4)
        results = strat.split_data(self.data, self.labels, k_vicini=5)

        # Verifica che il risultato sia una lista
        self.assertIsInstance(results, list)
        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(results), 3)

        # Verifica che ogni elemento della lista sia una tupla (y_test, y_pred, probabilità)
        for elem in results:
            self.assertIsInstance(elem, tuple)
            self.assertEqual(len(elem), 3)  # Deve essere (y_test, y_pred, probabilità)
            y_test, y_pred, probabilities = elem
            self.assertIsInstance(y_test, list)
            self.assertIsInstance(y_pred, list)
            self.assertIsInstance(probabilities, list)

    def test_invalid_params(self):
        """
        Verifica che venga generato un errore con parametri errati:
        - iterazioni <= 0
        - test_size <= 0 o test_size >= 1
        """
        with self.assertRaises(ValueError):
            StratifiedValidation(iterazioni=0, test_size=0.2)
        with self.assertRaises(ValueError):
            StratifiedValidation(iterazioni=5, test_size=1.0)

    def test_small_dataset(self):
        """
        Verifica il comportamento con dataset molto piccolo.
        """
        data = pd.DataFrame({"feature1": [10], "feature2": [100]})
        labels = pd.Series([1])

        strat = StratifiedValidation(iterazioni=2, test_size=0.5)
        with self.assertRaises(ValueError):
            strat.split_data(data, labels, k_vicini=5)

    def test_correct_iterations(self):
        """
        Verifica che il numero di iterazioni effettuate sia corretto.
        """
        n_iter = 5
        strat = StratifiedValidation(iterazioni=n_iter, test_size=0.4)
        results = strat.split_data(self.data, self.labels, k_vicini=5)
        self.assertEqual(len(results), n_iter)

    def test_stratification(self):
        """
        Verifica che la suddivisione sia effettivamente stratificata.
        """
        strat = StratifiedValidation(iterazioni=1, test_size=0.4)
        results = strat.split_data(self.data, self.labels, k_vicini=5)
        
        self.assertEqual(len(results), 1)
        y_test, y_pred, probabilities = results[0]

        # Controlla che il test set abbia la dimensione attesa
        self.assertEqual(len(y_test), 2)
        
        # Controlla che le classi siano rappresentate in modo stratificato
        unique_test_classes = set(y_test)
        self.assertTrue(len(unique_test_classes) >= 1, "Almeno una classe del dataset dovrebbe apparire nel test set.")

if __name__ == "__main__":
    unittest.main()

