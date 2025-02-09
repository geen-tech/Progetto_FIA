import numpy as np
from typing import List, Tuple, Dict

class Metrics:
    def __init__(self):
        '''
        Inizializza il costruttore della classe Metrics
        '''
        pass
    
    def calcolo_metriche(self, input: List[Tuple[List[int], List[int]]]) -> Dict[str, float]:
        print("Input ricevuto:", input)
        
        aggregated_metrics = {
            "Accuracy Rate": [],
            "Area Under Curve": [],
            "Sensitivity": [],
            "Geometric Mean": [],
            "Specificity": [],
        }

        # Calcola metriche per ogni coppia e aggiungi i valori alle liste aggregate
        for i, (y_real, y_pred) in enumerate(input):
            
            tp, tn, fp, fn = self._matrix_confusion(y_real, y_pred)
            
            try:
                accuracy = self._accuracy_rate(tp, tn, fp, fn)
                auc = self._area_under_curve(tp, tn, fp, fn)
                sensitivity = self._sensitivity(tp, fn)
                gmean = self._geometric_mean(tp, tn, fp, fn)
                specificity = self._specificity(tn, fp)

                aggregated_metrics["Accuracy Rate"].append(accuracy)
                aggregated_metrics["Area Under Curve"].append(auc)
                aggregated_metrics["Sensitivity"].append(sensitivity)
                aggregated_metrics["Geometric Mean"].append(gmean)
                aggregated_metrics["Specificity"].append(specificity)
            except Exception as e:
                print("Errore durante il calcolo delle metriche:", e)
        
        # Calcola la media solo se ci sono valori, altrimenti restituisce 0.0
        return {key: (sum(values) / len(values) if values else 0.0) for key, values in aggregated_metrics.items()}
    
    def _matrix_confusion(self, y_real: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
        tn = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 0)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 1)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 0)
        return tp, tn, fp, fn
    '''
    Essendo il dataset utilizzato sbilanciato l'accuracy rate non è molto affidabile, però la consideriamo perché ci dà 
    una visone completa dell'accuartezza dell'algorirmo di validazione (considera sia casi positivi che negativi) 
    '''
    def _accuracy_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        return (tp + tn) / total if total > 0 else 0.0
    '''
    Essendo il dataset in utilizzo sbilanciato l'AUC è una metrica fondamentale e affidabile, poiché indica la capacità del modello di validazione
    di distinguere tra classi postive  e negative
    '''
    
    def _area_under_curve(self, tp: int, tn: int, fp: int, fn: int) -> float:
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return (sensitivity + specificity) / 2
    '''
    Indica la capacità del modello di identificare i veri positivi (celluele veramente tumorali 4->1). 
    Importantissimo in un modello che vuole identificare le cellule tumurali, poiché se questa metrica risulta alta, non rischiamo di identificare cellule tumurali come benigne
    '''
    
    def _sensitivity(self, tp: int, fn: int) -> float:
        actual_positive = tp + fn
        return tp / actual_positive if actual_positive > 0 else 0.0
    '''
    Equilibrio tra Sensitivity e Specificity, quindi evita che il modello favorisca una delle due classi 
    '''
    
    def _geometric_mean(self, tp: int, tn: int, fp: int, fn: int) -> float:
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        return np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0.0
    '''
    Misura la capacità del modello di identificare i veri negativi (cellule sane 2->0)
    '''
    
    def _specificity(self, tn: int, fp: int) -> float:
        actual_negative = tn + fp
        return tn / actual_negative if actual_negative > 0 else 0.0
    

