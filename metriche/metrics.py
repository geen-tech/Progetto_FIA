import numpy as np
from typing import List, Tuple, Dict

class Metrics:
    def __init__(self):
        '''
        Inizializza il costruttore della classe Metrics
        '''
        pass
    
    def calcolo_metriche(self, input: list[tuple[list[int], list[int], list[float]]]) -> Dict[str, float]:
        print("Input ricevuto:", input)

        aggregated_metrics = {
            "Accuracy Rate": [],
            "Area Under Curve": [],
            "Sensitivity": [],
            "Geometric Mean": [],
            "Specificity": [],
            "Error Rate": [],
        }

        # Calcola metriche per ogni coppia di y_real e predicted_proba
        for i, (y_real, y_pred, predicted_proba) in enumerate(input):
            print(f"\nElaborazione del gruppo {i+1}:")
            print(f"y_real: {y_real}")
            print(f"y_pred: {y_pred}")
            print(f"predicted_proba: {predicted_proba}")

            tp, tn, fp, fn = self._matrix_confusion(y_real, y_pred)
            print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

            try:
                accuracy = self._accuracy_rate(tp, tn, fp, fn)
                print(f"Accuracy: {accuracy}")

                auc = self._area_under_curve(y_real, predicted_proba)  # Passa le probabilità
                print(f"AUC: {auc}")

                sensitivity = self._sensitivity(tp, fn)
                print(f"Sensitivity: {sensitivity}")

                gmean = self._geometric_mean(tp, tn, fp, fn)
                print(f"Geometric Mean: {gmean}")

                specificity = self._specificity(tn, fp)
                print(f"Specificity: {specificity}")

                error_rate = self._error_rate(tp, tn, fp, fn)
                print(f"Error Rate: {error_rate}")

                # Aggiungi le metriche agli aggregati
                aggregated_metrics["Accuracy Rate"].append(accuracy)
                aggregated_metrics["Area Under Curve"].append(auc)
                aggregated_metrics["Sensitivity"].append(sensitivity)
                aggregated_metrics["Geometric Mean"].append(gmean)
                aggregated_metrics["Specificity"].append(specificity)
                aggregated_metrics["Error Rate"].append(error_rate)

            except Exception as e:
                print("Errore durante il calcolo delle metriche:", e)

        # Calcola la media solo se ci sono valori, altrimenti restituisce 0.0
        print("\nMetriche aggregate:", aggregated_metrics)
        return {key: (sum(values) / len(values) if values else 0.0) for key, values in aggregated_metrics.items()}
    
    def _matrix_confusion(self, y_real: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        print(f"Calcolando la matrice di confusione tra:\n y_real: {y_real} \n y_pred: {y_pred}")
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
        tn = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 0)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 1)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 0)
        print(f"Matrix Confusion -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        return tp, tn, fp, fn

    @staticmethod
    def scegli_metriche() -> List[str]:
        print("Scegli le metriche da calcolare:")
        print("1. Accuracy Rate")
        print("2. Error Rate")
        print("3. Sensitivity")
        print("4. Specificity")
        print("5. Geometric Mean")
        print("6. Area Under Curve")
        print("7. Calcola tutte le metriche")

        while True:
            metrics_choice = input("Inserisci i numeri delle metriche da calcolare separati da virgola (es. 1,2,3): ").strip()
            
            if metrics_choice == "7":
                return ["Accuracy Rate", "Error Rate", "Sensitivity", "Specificity", "Geometric Mean", "Area Under Curve"]

            # Split dell'input e rimozione degli spazi extra
            choices = metrics_choice.split(',')
            
            # Validazione delle scelte
            valid_choices = set(["1", "2", "3", "4", "5", "6"])
            selected_metrics = []
            invalid_choices = []

            for choice in choices:
                choice = choice.strip()  # Rimuove gli spazi extra
                if choice in valid_choices:
                    metrics_map = {
                        "1": "Accuracy Rate",
                        "2": "Error Rate",
                        "3": "Sensitivity",
                        "4": "Specificity",
                        "5": "Geometric Mean",
                        "6": "Area Under Curve"
                    }
                    selected_metrics.append(metrics_map[choice])
                else:
                    invalid_choices.append(choice)

            if not selected_metrics:
                print("Nessuna metrica valida selezionata. Riprova.")
            else:
                if invalid_choices:
                    print(f"Le seguenti scelte non sono valide: {', '.join(invalid_choices)}. Per favore, correggi l'input.")
                else:
                    return selected_metrics

    def _accuracy_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        print(f"Calcolato Accuracy: {accuracy}")
        return accuracy

    def _area_under_curve(self, y_test: np.ndarray, predicted_proba: np.ndarray) -> float:
        y_test = np.asarray(y_test)
        predicted_proba = np.asarray(predicted_proba)
        # Debug: Verifica il tipo di y_test e predicted_proba
        print(f"Tipo di y_test: {type(y_test)}, Tipo di predicted_proba: {type(predicted_proba)}")
        print(f"y_test: {y_test}")
        print(f"predicted_proba: {predicted_proba}")

        # Controllo per verificare se le lunghezze di y_test e predicted_proba corrispondono
        if len(y_test) != len(predicted_proba):
            print(f"Errore: le lunghezze di y_test e predicted_proba non corrispondono. "
                f"y_test: {len(y_test)}, predicted_proba: {len(predicted_proba)}")
            return 0.0

        print("Lunghezza di y_test e predicted_proba sono corrette.")

        # Verifica se ci sono almeno due classi
        unique_classes = np.unique(y_test)
        
        if len(unique_classes) < 2:
            print("Errore: Actual values contiene solo una classe, AUC non può essere calcolata!")
            return 0.0

        positive_label = 1  # Definito come classe positiva
        if len(np.unique(y_test)) < 2:
            print("Errore: Actual values contiene solo una classe, AUC non può essere calcolata!")
            return 0.0

        # Ordinamento per ottenere le probabilità predette
        sorted_indices = np.argsort(predicted_proba)
        y_true_sorted = y_test[sorted_indices]
        predicted_proba_sorted = predicted_proba[sorted_indices]

        total_positives = np.sum(y_true_sorted == positive_label)
        total_negatives = np.sum(y_true_sorted != positive_label)

        print(f"Totale positivi: {total_positives}, Totale negativi: {total_negatives}")

        if total_positives == 0 or total_negatives == 0:
            print("Errore: non ci sono abbastanza esempi per calcolare la curva ROC.")
            return 0.0

        # Calcolo di True Positive Rate (TPR) e False Positive Rate (FPR)
        TPR = np.cumsum(y_true_sorted == positive_label) / total_positives
        FPR = np.cumsum(y_true_sorted != positive_label) / total_negatives

        print(f"TPR (prime 5): {TPR[:5]} ...")
        print(f"FPR (prime 5): {FPR[:5]} ...")

        # Calcolo dell'AUC con la formula dell'integrazione numerica (metodo del trapezio)
        auc = np.trapz(TPR, FPR)
        print(f"AUC calcolato: {auc}")

        return float(auc)

    def _sensitivity(self, tp: int, fn: int) -> float:
        actual_positive = tp + fn
        sensitivity = tp / actual_positive if actual_positive > 0 else 0.0
        print(f"Sensitivity: {sensitivity}")
        return sensitivity
    
    def _geometric_mean(self, tp: int, tn: int, fp: int, fn: int) -> float:
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        gmean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0.0
        print(f"Geometric Mean: {gmean}")
        return gmean
    
    def _specificity(self, tn: int, fp: int) -> float:
        actual_negative = tn + fp
        specificity = tn / actual_negative if actual_negative > 0 else 0.0
        print(f"Specificity: {specificity}")
        return specificity
    
    def _error_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        error_rate = (fp + fn) / total if total > 0 else 0.0
        print(f"Error Rate: {error_rate}")
        return error_rate
