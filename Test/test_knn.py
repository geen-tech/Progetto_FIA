import unittest
import pandas as pd
from models.classifier import CustomKNN

class TestCustomKNN(unittest.TestCase):
    def setUp(self):
        # Setup iniziale
        # Dataset di esempio
        self.training_data = pd.DataFrame({
            'Feature1': [9, 5, 7, 6],
            'Feature2': [1, 4, 2, 5]
        })
        self.training_labels = pd.Series([0, 1, 1, 0])

        self.test_data = pd.DataFrame({
            'Feature1': [3, 2.5],
            'Feature2': [5, 2.5]
        })

        self.expected_predictions = [1, 0]  # Risultati attesi con k=3

        self.knn = CustomKNN(k=5)

    def test_fit(self):
        """
        Verifica che il metodo fit memorizzi correttamente i dati di training.
        """
        self.knn.fit(self.training_data, self.training_labels)
        pd.testing.assert_frame_equal(self.knn.data, self.training_data)
        pd.testing.assert_series_equal(self.knn.labels, self.training_labels)
    
    def test_predict(self):
        """
        Verifica che il metodo predict fornisca il risultato atteso.
        """
        self.knn.fit(self.training_data, self.training_labels)
        prediction = self.knn.predict(self.test_data.iloc[0])
        self.assertEqual(prediction, self.expected_predictions[0])
    
    def test_invalid_fit_input(self):
        """
        Verifica che venga sollevata un'eccezione se i dati forniti a fit non sono validi.
        """
        with self.assertRaises(ValueError):
            self.knn.fit([1, 2, 3], self.training_labels)  # Non è un DataFrame
    
    def test_invalid_predict_input(self):  
        """  
        Verifica che predict sollevi un'eccezione se l'input non è nel formato corretto.  
        """  
        self.knn.fit(self.training_data, self.training_labels)  

        with self.assertRaises(ValueError, msg="Predict dovrebbe sollevare un ValueError per input non valido"):  
            self.knn.predict([2.5, 2.5])  # Input non conforme (non è un pandas Series)  

    def tearDown(self):  
        """  
        Dealloca le risorse utilizzate nei test.  
        """  
        del self.knn  

if __name__ == '__main__':  
    unittest.main()  
    

