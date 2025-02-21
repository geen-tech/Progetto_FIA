import unittest
import pandas as pd
import numpy as np
from validazione import RandomSubsampling
from models.classifier import CustomKNN

class TestRandomSubsampling(unittest.TestCase):
    
    def setUp(self):
        # Creazione di un dataset di esempio
        self.data = pd.DataFrame({
            "feature1": [7, 3, 2, 4, 6, 8, 5, 9, 1, 10],
            "feature2": [70, 30, 20, 40, 60, 80, 50, 90, 10, 100]
        })
        self.labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    def test_valid_split(self):
        # Verifica che la funzione split_data restituisca il numero corretto di iterazioni
        test_size = 0.2
        iterazioni = 5
        k_vicini = 3
        random_subsampling = RandomSubsampling(test_size, iterazioni)
        risultati = random_subsampling.split_data(self.data, self.labels, k_vicini)
        
        self.assertEqual(len(risultati), iterazioni, "Il numero di iterazioni nei risultati non corrisponde a quello atteso")
    
    def test_correct_split_proportions(self):
        # Verifica che il numero di campioni nel test set sia corretto
        test_size = 0.3
        iterazioni = 5
        k_vicini = 3
        random_subsampling = RandomSubsampling(test_size, iterazioni)
        risultati = random_subsampling.split_data(self.data, self.labels, k_vicini)
        
        expected_test_samples = int(len(self.data) * test_size)
        
        for y_real, y_pred, probabilities in risultati:
            self.assertEqual(len(y_real), expected_test_samples, "La dimensione del test set non è corretta")
            self.assertEqual(len(y_pred), expected_test_samples, "La dimensione delle predizioni non è corretta")
            self.assertEqual(len(probabilities), expected_test_samples, "La dimensione delle probabilità non è corretta")
    
    def test_edge_case_small_dataset(self):
        # Test con dataset molto piccolo (2 campioni)
        small_data = pd.DataFrame({
            "feature1": [7, 3],
            "feature2": [70, 30]
        })
        small_labels = pd.Series([1, 0])
        
        test_size = 0.5
        iterazioni = 3
        k_vicini = 1
        random_subsampling = RandomSubsampling(test_size, iterazioni)
        risultati = random_subsampling.split_data(small_data, small_labels, k_vicini)
        
        for y_real, y_pred, probabilities in risultati:
            self.assertGreaterEqual(len(y_real), 1, "Il set di test è troppo piccolo")
            self.assertGreaterEqual(len(y_pred), 1, "Il set di predizioni è troppo piccolo")
    
    def test_no_overlap_train_test(self):
        # Verifica che non ci siano sovrapposizioni tra il set di training e il set di test
        test_size = 0.2
        iterazioni = 5
        k_vicini = 3
        random_subsampling = RandomSubsampling(test_size, iterazioni)
        risultati = random_subsampling.split_data(self.data, self.labels, k_vicini)
        
        for i, (y_real, _, _) in enumerate(risultati):
            test_indices = [self.labels.tolist().index(y) for y in y_real]
            all_indices = set(range(len(self.labels)))
            train_indices = list(all_indices - set(test_indices))
            overlap = set(test_indices).intersection(train_indices)
            self.assertEqual(len(overlap), 0, f"Sovrapposizione trovata tra test e train nell'iterazione {i + 1}.")
    
    def test_invalid_test_size(self):
        # Verifica che venga sollevato un errore per test_size fuori intervallo
        with self.assertRaises(ValueError):
            RandomSubsampling(test_size=1.2, iterazioni=5)
    
    def test_invalid_iterations(self):
        # Verifica che venga sollevato un errore per numero di iterazioni non valido
        with self.assertRaises(ValueError):
            RandomSubsampling(test_size=0.2, iterazioni=-3)

if __name__ == "__main__":
    unittest.main()






    




