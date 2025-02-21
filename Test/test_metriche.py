import unittest
import numpy as np
from metriche.metrics import Metrics 

class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.metrics = Metrics()
        self.y_real = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        self.y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
        self.predicted_proba = [0.9, 0.2, 0.85, 0.4, 0.1, 0.75, 0.7, 0.3, 0.95, 0.05]
    
    def test_confusion_matrix(self):
        tp, tn, fp, fn = self.metrics._matrix_confusion(self.y_real, self.y_pred)
        self.assertEqual(tp, 4)
        self.assertEqual(tn, 4)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
    
    def test_accuracy_rate(self):
        tp, tn, fp, fn = 4, 4, 1, 1
        accuracy = self.metrics._accuracy_rate(tp, tn, fp, fn)
        self.assertAlmostEqual(accuracy, 0.8, places=2)
    
    def test_sensitivity(self):
        tp, fn = 4, 1
        sensitivity = self.metrics._sensitivity(tp, fn)
        self.assertAlmostEqual(sensitivity, 0.8, places=2)
    
    def test_specificity(self):
        tn, fp = 4, 1
        specificity = self.metrics._specificity(tn, fp)
        self.assertAlmostEqual(specificity, 0.8, places=2)
    
    def test_geometric_mean(self):
        tp, tn, fp, fn = 4, 4, 1, 1
        gmean = self.metrics._geometric_mean(tp, tn, fp, fn)
        self.assertAlmostEqual(gmean, np.sqrt(0.8 * 0.8), places=2)
    
    def test_error_rate(self):
        tp, tn, fp, fn = 4, 4, 1, 1
        error_rate = self.metrics._error_rate(tp, tn, fp, fn)
        self.assertAlmostEqual(error_rate, 0.2, places=2)
    
    def test_auc(self):
        auc = self.metrics._area_under_curve(self.y_real, self.predicted_proba)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)
    
    
if __name__ == '__main__':
    unittest.main()

    

    



