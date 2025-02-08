import unittest
import pandas as pd
import numpy as np
from models.classifier import  CustomKNN
from validazione import Holdout

class TestHoldout(unittest.TestCase): 
    def setUp(self):
        # Setup iniziale
        # Dataset di esempio
        self.data = pd.DataFrame({
            "feature1": [7, 3, 2, 4, 6],
            "feature2": [70, 30, 20, 40, 60]
        })
        self.labels = pd.Series([1, 0, 1, 0, 1])

    def test_split_data(self):
        """
        Si verifica il corretto funzionamento di base di generate_splits con valori validi,  
        assicurandosi che restituisca una lista contenente tuple nella forma (y_real, y_pred).  
        Inoltre, si controlla che il numero di campioni nei set di training e test sia corretto.  

        """
        holdout = Holdout(test_size=0.4)
        results = holdout.split_data(self.data, self.labels, k_vicini=5)
        '''
        - Verifica che il risultato sia una lista
        - Verifica che il risultato sia una lista con una sola tupla
        - Verifica che l'elemento della lista sia una tupla
        '''
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1) 
        self.assertIsInstance(results[0], tuple)

        # Verifica che y_real e y_pred abbiano la dimensione corretta
        y_real, y_pred = results[0]
        self.assertEqual(len(y_real), 2)  # 40% di 5 campioni ≈ 2
        self.assertEqual(len(y_pred), 2)

    def test_split_data_default(self): # Facciamo il controllo con il valore di default del test size, ossia 20%
    
        holdout = Holdout(test_size=0.2)  
        results = holdout.split_data(self.data, self.labels, k_vicini=5)
        
        y_real, y_pred = results[0]
        self.assertEqual(len(y_real), 1)  # 20% di 5 campioni ≈ 1
        self.assertEqual(len(y_pred), 1) 

    def test_split_data_invalid(self):  
        """  
        Controlla che venga generato un errore quando test_size ha un valore non valido 
        """  
        # Caso in cui test_size è maggiore di 1  
        with self.assertRaises(ValueError):  
            Holdout(test_size=1.8)  

        # Caso in cui test_size è negativo  
        with self.assertRaises(ValueError):  
            Holdout(test_size=-0.8)  

    def test_split_data_small_dataset(self):  
        """  
        Verifica il comportamento di generate_splits con un dataset molto ridotto.  
        Con test_size=0.5 e un solo campione, entrambe le liste (y_real e y_pred)  
        dovrebbero risultare vuote.  
        """  
        data = pd.DataFrame({  
            "feature1": [6],  
            "feature2": [60]  
        })  
        labels = pd.Series([1])  

        holdout = Holdout(test_size=0.5)    

        with self.assertRaises(ValueError): # Controllo se split_data solleva un errore. Se questo avviene il test non fallisce 
            holdout.split_data(data, labels, k_vicini=5)

    def test_split_data_full(self):
        """
        Testa generate_splits con test_size = 1.0 (tutti i campioni come test) -> Dovrebbe generare un errore (manacanza dati nel training set)
        """
        holdout = Holdout(test_size=1.0)
        with self.assertRaises(ValueError): # Controllo se split_data solleva un errore. Se questo avviene il test non fallisce
            holdout.split_data(self.data, self.labels, k_vicini=5) 
    
    def test_split_data_no_shuffle(self):  
        """  
        Verifica che il dataset venga suddiviso correttamente senza mescolamento.  
        Controlla che i dati di training e test siano disgiunti.  
        """  
        holdout = Holdout(test_size=0.2)  
        results = holdout.split_data(self.data, self.labels, k_vicini=5)  

        y_real_test = results[0][0]  # Estrai le etichette reali del test set  
        
        # Ottieni gli indici originali corrispondenti alle etichette del test set  
        test_indices = [self.labels.tolist().index(y) for y in y_real_test]  

        # Calcola gli indici del training set come differenza tra tutti gli indici e quelli del test set  
        all_indices = set(range(len(self.labels)))  
        train_indices = list(all_indices - set(test_indices))  

        # Verifica che non ci siano sovrapposizioni tra i due insiemi di indici  
        self.assertTrue(set(test_indices).isdisjoint(set(train_indices)),  
                    "Gli indici di test e training non dovrebbero sovrapporsi.")  


if __name__ == "__main__":
    unittest.main()


