import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class Metrics:
    def __init__(self):
        '''
        Inizializza il costruttore della classe Metrics
        '''
        pass

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
    
    def calcolo_metriche(self, y_real: np.ndarray, y_pred: np.ndarray, predicted_proba: np.ndarray , metriche_selezionate) -> Dict[str, float]:

        aggregated_metrics={metrica: [] for metrica in metriche_selezionate}

        tp, tn, fp, fn = self._matrix_confusion(y_real, y_pred)
        print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        try:
            if "Accuracy Rate" in metriche_selezionate:
                accuracy = self._accuracy_rate(tp, tn, fp, fn)
                print(f"Accuracy: {accuracy}")
                aggregated_metrics["Accuracy Rate"].append(accuracy)

            if  "Area Under Curve" in metriche_selezionate:
                auc = self._area_under_curve(y_real, predicted_proba)  
                print(f"AUC: {auc}")
                aggregated_metrics["Area Under Curve"].append(auc)

            if  "Sensitivity" in metriche_selezionate:
                sensitivity = self._sensitivity(tp, fn)
                print(f"Sensitivity: {sensitivity}")
                aggregated_metrics["Sensitivity"].append(sensitivity)

            if  "Geometric Mean" in metriche_selezionate:
                gmean = self._geometric_mean(tp, tn, fp, fn)
                print(f"Geometric Mean: {gmean}")
                aggregated_metrics["Geometric Mean"].append(gmean)

            if  "Specificity" in metriche_selezionate:
                specificity = self._specificity(tn, fp)
                print(f"Specificity: {specificity}")
                aggregated_metrics["Specificity"].append(specificity)

            if  "Error Rate" in metriche_selezionate:
                error_rate = self._error_rate(tp, tn, fp, fn)
                print(f"Error Rate: {error_rate}")
                aggregated_metrics["Error Rate"].append(error_rate)

        except Exception as e:
            print("Errore durante il calcolo delle metriche:", e)

        # Calcola la media delle metriche aggregate
        print("\nMetriche aggregate:", aggregated_metrics)
        return {
            key: (sum(values) / len(values) if values else 0.0) 
            for key, values in aggregated_metrics.items()
        }
    
    def compute_batch_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, predicted_probs: np.ndarray, num_folds: int, batch_size: int, metriche_selezionate: List[str]) -> Dict[str, float]:
        """
        Calcola le metriche sui diversi gruppi e salva la distribuzione sulle folds
        """
        print("DENTRO REALI")
        print(y_true)
        print("DENTRO PRED")
        print(y_pred)
        metrics_calc = Metrics()  # Creiamo una nuova istanza della classe Metrics per calcolare le metriche
        metrics_data = {
            "fold": [],
            "Accuracy Rate": [],
            "Error Rate": [],
            "Sensitivity": [],
            "Specificity": [],
            "Geometric Mean": [],
            "Area Under Curve": []
        }

        # Itera su ogni fold per creare i batch
        for fold in range(num_folds):
            start_idx = fold * batch_size
            end_idx = start_idx + batch_size

            
            y_true[start_idx:end_idx], 
            y_pred[start_idx:end_idx], 
            predicted_probs[start_idx:end_idx]

            # Calcola le metriche per questo batch
            metrics = metrics_calc.calcolo_metriche(y_true, y_pred, predicted_probs, metriche_selezionate)

            # Salva i risultati per il fold corrente
            metrics_data["fold"].append(fold + 1)
            metrics_data["Accuracy Rate"].append(metrics["Accuracy Rate"])
            metrics_data["Error Rate"].append(metrics["Error Rate"])
            metrics_data["Sensitivity"].append(metrics["Sensitivity"])
            metrics_data["Specificity"].append(metrics["Specificity"])
            metrics_data["Geometric Mean"].append(metrics["Geometric Mean"])
            metrics_data["Area Under Curve"].append(metrics["Area Under Curve"])

        # Converte in un DataFrame per calcolare la media
        df_metrics = pd.DataFrame(metrics_data)

        # Calcola la media delle metriche
        avg_metrics = df_metrics.drop(columns=["fold"]).mean().to_dict()

        return avg_metrics

    def _matrix_confusion(self, y_real: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        tp = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 1)
        tn = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 0)
        fp = sum(1 for r, p in zip(y_real, y_pred) if r == 0 and p == 1)
        fn = sum(1 for r, p in zip(y_real, y_pred) if r == 1 and p == 0)
        print(f"Matrix Confusion -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        return tp, tn, fp, fn

    def _accuracy_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        return accuracy


    def _compute_roc_points(self, y_test: np.ndarray, predicted_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola i punti (TPR, FPR) della curva ROC iterando su soglie da 1 a 0.
        Classe positiva = 1.
        """
        
        thresholds = np.linspace(1, 0, num=50)  
        tpr_list = [0.0]   
        fpr_list = [0.0]  

        for thr in thresholds:
            # binarizzo le predizioni in base a thr
            pred_pos = (predicted_proba >= thr)

            # Calcolo della conf matrix su questa soglia
            TP = np.sum((y_test == 1) & (pred_pos == True))
            FP = np.sum((y_test == 0) & (pred_pos == True))
            TN = np.sum((y_test == 0) & (pred_pos == False))
            FN = np.sum((y_test == 1) & (pred_pos == False))

            # TPR & FPR
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

            tpr_list.append(TPR)
            fpr_list.append(FPR)

        return np.array(tpr_list), np.array(fpr_list)

    def _area_under_curve(self, y_test: np.ndarray, predicted_proba: np.ndarray) -> float:
        """
        Calcola AUC-ROC iterando su più soglie e integrando la curva TPR(FPR).
        """
        y_test = np.asarray(y_test)
        predicted_proba = np.asarray(predicted_proba)
        
        # 1) Verifica dimensioni e almeno due classi
        if len(y_test) != len(predicted_proba):
            print("Errore: le lunghezze di y_test e predicted_proba non coincidono.")
            return 0.0
        if len(np.unique(y_test)) < 2:
            print("Errore: y_test contiene una sola classe, impossibile calcolare AUC.")
            return 0.0
        
        # 2) Calcolo dei punti ROC
        tpr, fpr = self._compute_roc_points(y_test, predicted_proba)

        # 3) Ordino i punti in base a FPR (crescente)
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # 4) Integro con metodo dei trapezi
        auc_value = np.trapz(tpr_sorted, fpr_sorted)
        return float(auc_value)

    # Metodo pubblico per mostrare la curva ROC
    def plot_roc_curve(self, y_test: List[int], predicted_proba: List[float]) -> None:
        """
        Traccia la curva ROC (TPR vs FPR) basata su soglie multiple.
        """
        y_test = np.asarray(y_test)
        predicted_proba = np.asarray(predicted_proba)

        # Calcolo TPR, FPR
        tpr, fpr = self._compute_roc_points(y_test, predicted_proba)

        # Ordino i punti
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # Plot
        plt.figure(figsize=(6,5))
        plt.plot(fpr_sorted, tpr_sorted, label='ROC curve', color='blue')
        plt.plot([0,1], [0,1], color='red', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def _sensitivity(self, tp: int, fn: int) -> float:
        actual_positive = tp + fn
        sensitivity = tp / actual_positive if actual_positive > 0 else 0.0
        return sensitivity
    
    def _specificity(self, tn: int, fp: int) -> float:
        actual_negative = tn + fp
        specificity = tn / actual_negative if actual_negative > 0 else 0.0
        return specificity
    
    def _geometric_mean(self, tp: int, tn: int, fp: int, fn: int) -> float:
        sensitivity = self._sensitivity(tp, fn)
        specificity = self._specificity(tn, fp)
        gmean = np.sqrt(sensitivity * specificity) if (sensitivity > 0 and specificity > 0) else 0.0
        return float(gmean)
    
    def _error_rate(self, tp: int, tn: int, fp: int, fn: int) -> float:
        total = tp + tn + fp + fn
        error_rate = (fp + fn) / total if total > 0 else 0.0
        return error_rate
    
    def plot_conf_matrix(self, confusion_dict: Dict[str, int]):
        """Plotta la matrice di confusione"""
        confusion_matrix = np.array([
            [confusion_dict["TN"], confusion_dict["FP"]],
            [confusion_dict["FN"], confusion_dict["TP"]]
        ])

        fig, ax = plt.subplots()
        cax = ax.matshow(confusion_matrix, cmap='Blues')
        fig.colorbar(cax)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'])
        ax.set_yticklabels(['Actual Negative', 'Actual Positive'])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(confusion_matrix[i, j]), va='center', ha='center', color='black')

        plt.title("Matrice di Confusione")
        plt.show()
    
