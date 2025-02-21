import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from metriche.metrics import Metrics

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

    def visualize_metrics(self) -> None:
        """
        Calcola e visualizza tutte le metriche disponibili.
        """
        # Calcola le metriche
        self.metrics = self.calculator.calcolo_metriche(self.input, self.metriche_selezionate)
        
        # Per ogni gruppo nel tuo input, mostriamo la Confusion Matrix e la curva ROC
        for i, (y_real, y_pred, predicted_proba) in enumerate(self.input, start=1):
            print(f"\n--- Plot del gruppo n. {i} ---")

           
            # Mostra Curva ROC
            self.calculator.plot_roc_curve(y_real, predicted_proba)
        
        # Grafico a barre delle metriche aggregate
        self.plot(self.metrics)

    
    
    def save(self, filename: str = "metriche.xlsx") -> None:
        """
        Salva quello calcolato in un file Excel

        Args:
            filename (str): Nome del file Excel "metriche.xlsx"
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

        Args:
            metrics (dict): Dizionario contenente i valori delle metriche
        """
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
        plt.title("Performance Metrics")
        plt.ylabel("Value")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
