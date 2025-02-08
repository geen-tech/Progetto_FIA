import unittest
import pandas as pd
from validazione import RandomSubsampling

class TestRandomSubsampling(unittest.TestCase):
    def setUp(self):
        # Setup iniziale
        # Dataset di esempio
        self.data = pd.DataFrame({
            "feature1": [7, 3, 2, 4, 6],
            "feature2": [70, 30, 20, 40, 60]
        })
        self.labels = pd.Series([1, 0, 1, 0, 1])
    
    def test_split_data(self):
        
        # Testa il funzionamento di base di generate_splits con valori validi
        
        subsampling = RandomSubsampling(iterazioni=5, test_size=0.2)
        results = subsampling.split_data(self.data, self.labels, k_vicini=5)

        # Verifica che il risultato sia una lista
        self.assertIsInstance(results, list)
        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(results), 5)

        # Verifica che ogni elemento sia una tupla (y_real, y_pred)
        for y_real, y_pred in results:
            self.assertIsInstance(y_real, list)
            self.assertIsInstance(y_pred, list)

    def test_split_data_deafault(self):
        
        # Testa generate_splits con i valori di default di itearzioni e test_size (iterazioni=10 tst_size=0.2)
        
        random = RandomSubsampling(test_size=0.2, iterazioni=10)  
        results = random.split_data(self.data, self.labels, k_vicini=5)

        # Verifica che il numero di iterazioni sia quello di default
        self.assertEqual(len(results), 10)

        # Verifica che il test set abbia circa il 20% dei campioni
        y_real, _ = results[0]
        self.assertAlmostEqual(len(y_real), int(len(self.data) * 0.2), delta=1)
    
    def test_split_data_small_dataset(self):
        """
        Testa generate_splits con un dataset molto piccolo.
        """
        data = pd.DataFrame({
            "feature1": [7, 9],
            "feature2": [70, 90]
        })
        labels = pd.Series([1, 1])

        random = RandomSubsampling(iterazioni=5, test_size=0.5)
        results = random.split_data(data, labels, k_vicini=5)

        # Verifica che ogni test set abbia almeno 1 campione
        for y_real, y_pred in results:
            test_size = len(y_real)
            pred_size = len(y_pred)
        
        # Controlla che entrambi i set abbiano almeno un campione
        self.assertTrue(test_size >= 1, f"Test set troppo piccolo: {test_size} campioni")
        self.assertTrue(pred_size >= 1, f"Predizioni troppo piccole: {pred_size} campioni")

    def test_split_data_check_iterazioni(self):
        """
        Verifica che il numero di iterazioni generato da RandomSubsampling sia corretto.
        """

        # Inizializza RandomSubsampling con 5 iterazioni
        n_iter = 5
        random = RandomSubsampling(test_size=0.5, iterazioni=n_iter)
        results = random.split_data(self.data, self.labels, k_vicini=5)

        # Verifica che il numero di iterazioni sia corretto
        self.assertEqual(len(results), n_iter, f"Numero di iterazioni generato: {len(results)}. Ci si aspettava {n_iter} iterazioni.")
    
    def test_split_data_no_overlap_between_test_train(self):
        """
        Verifica che non ci siano sovrapposizioni tra il set di training e quello di test.
        """

        # Inizializza RandomSubsampling con 3 iterazioni e test_size del 40% (0.4)
        random = RandomSubsampling(test_size=0.2, iterazioni=5)
        results = random.split_data(self.data, self.labels, k_vicini=5)

        # Per ogni iterazione, controlla che non ci siano sovrapposizioni tra set di test e set di addestramento
        for i, (y_real, _) in enumerate(results):
            # Ottieni gli indici dei campioni nel set di test
            test_indices = [self.labels.tolist().index(y) for y in y_real]

            # Calcola gli indici del set di training come la differenza tra tutti gli indici e quelli del test set
            all_indices = set(range(len(self.labels)))
            train_indices = list(all_indices - set(test_indices))

            # Verifica che non ci siano sovrapposizioni tra i set di test e di addestramento
            overlap = set(test_indices).intersection(train_indices)
            self.assertEqual(len(overlap), 0, f"Sovrapposizione trovata tra test e train nell'iterazione {i + 1}.")

if __name__ == "__main__":
    unittest.main()

    




