import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from .metrics import Metrics

class Visualizer:
    def __init__(self, input: list[tuple[list[int], list[int], list[float]]], metriche_selezionate):
        """
        Inizializza l'oggetto per visualizzare le metriche.

        Args:
            input_data (List[Tuple[List[int], List[int]]]): Una lista di tuple (y_real, y_pred).
        """
        self.input = input
        self.calculator = Metrics()
        self.metriche_selezionate=metriche_selezionate
        self.metrics = {}
        self.y_real = [row[0] for row in input] 
        self.y_real = [item for sublist in self.y_real for item in sublist] 
        self.y_pred = [row[1] for row in input]
        self.y_pred = [item for sublist in self.y_pred for item in sublist]
        self.pred_proba = [row[2] for row in input] 
        self.pred_proba = [item for sublist in self.pred_proba for item in sublist] 
        
    def visualize_metrics(self) -> None:
        """
        Calcola e visualizza tutte le metriche disponibili
        """
        # Calcola le metriche
        self.metrics = self.calculator.calcolo_metriche(self.y_real, self.y_pred, self.pred_proba, self.metriche_selezionate)
        
        # Calcolo della matrice di confusione
        tp, tn, fp, fn = self.calculator._matrix_confusion(self.y_real, self.y_pred)
        confusion_dict = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }
        self.calculator.plot_conf_matrix(confusion_dict)

        # Mostra Curva ROC
        self.calculator.plot_roc_curve(self.y_real, self.pred_proba)
        
        # Grafico a barre delle metriche aggregate
        self.plot(self.metrics)

    def media(self, K = 1) -> dict:
        """
        Calcola le metriche medie sui vari gruppi per Random e Stratified
        """
        # calcoliamo le metriche
        sample_size = int(len(self.y_real)/K)
        print(len(self.y_real))
        self.metrics = self.calculator.compute_batch_metrics(self.y_real, self.y_pred, self.pred_proba, K, sample_size, self.metriche_selezionate)
        self.plot(self.metrics)

        return
    
    def save(self, filename: str = "metriche.xlsx") -> None:
        """
        Salva quello calcolato in un file Excel
        """
        if not self.metrics:
            print("Nessuna metrica da salvare. Esegui prima 'visualize_metrics()'.")
            return

        # Salva le metriche in un file Excel
        metrics_df = pd.DataFrame(self.metrics.items(), columns=["Metric", "Value"])
        metrics_df.to_excel(filename, index=False, engine="openpyxl")
        print(f"Metriche salvate in {filename}")

        """
        Filename: Nome del file in cui salvare i dati,
        Non include l'indice del DataFrame nel file Excel,
        Specifica che viene utilizzato il motore openpyxl per scrivere il file
        """
    
    def plot(self, metrics: Dict[str, float]) -> None:
        """
        Grafica le metriche in un grafico a barre
        """
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
        plt.title("Performance Metrics")
        plt.ylabel("Value")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

