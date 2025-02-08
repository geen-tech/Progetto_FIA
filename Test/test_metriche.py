import unittest
from metriche.metrics import Metrics

class TestMetriche(unittest.TestCase):
    def setUp(self):
        # Setup iniziale: crea un'istanza della classe Metrics 
        self.calculator = Metrics()
        # Dati d'esempio
        self.sample_data = [
            ([1, 0, 1, 1, 0], [1, 0, 1, 0, 0]),
            ([0, 1, 1, 0, 0], [0, 1, 0, 0, 1]),
            ([1, 1, 0, 1, 0], [1, 1, 0, 0, 0]),
            ([0, 0, 1, 1, 1], [0, 0, 1, 1, 0]),
            ([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]),  
            ([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]),  
            ([0, 1, 1, 0, 0, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1, 1, 0, 1, 1]),  
            ([1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]),  
            ([0, 0, 1, 1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 0, 1, 1, 1]),  
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 1, 1, 0, 0]),  
            ([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1, 1, 1]),  
        ]
    
    def test_accuracy_rate(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._matrix_confusion(y_true, y_pred)
            accuracy_rate = self.calculator._accuracy_rate(tp, tn, fp, fn)

            # Calcolo manuale dell'accuracy per confronto
            total = tp + tn + fp + fn
            expected_accuracy = (tp + tn) / total if total > 0 else 0.0

            # Verifica che i due valori siano quasi uguali (considerando piccole imprecisioni)
            self.assertAlmostEqual(accuracy_rate, expected_accuracy, places=5)

    def test_area_under_curve(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._matrix_confusion(y_true, y_pred)
            a_u_c = self.calculator._area_under_curve(tp, tn, fp, fn)
            sensitivity = self.calculator._sensitivity(tp, fn)
            specificity = self.calculator._specificity(tn, fp)
            expected_auc = (sensitivity + specificity) / 2
            self.assertAlmostEqual(a_u_c, expected_auc)


    def test_sensitivity(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._matrix_confusion(y_true, y_pred)
            sensitivity = self.calculator._sensitivity(tp, fn)

            actual_positive = tp + fn
            expected_sensitivity = tp / actual_positive if actual_positive > 0 else 0.0

            self.assertAlmostEqual(sensitivity, expected_sensitivity, places=5)

    def test_geometric_mean(self):
        for y_true, y_pred in self.sample_data:
            tp, tn, fp, fn = self.calculator._matrix_confusion(y_true, y_pred)
            geometric_mean = self.calculator._geometric_mean(tp, tn, fp, fn)
            # Calcolo manuale della geometric mean per confronto
            sensitivity = self.calculator._sensitivity(tp, fn)
            specificity = self.calculator._specificity(tn, fp)
            expected_gmean = (sensitivity * specificity) ** 0.5
            self.assertAlmostEqual(geometric_mean, expected_gmean)

if __name__ == '__main__':
    unittest.main() 
    

    



