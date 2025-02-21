import traceback
from validazione.validazione_main import setup_knn_validation
from preprocesso.preprocesso_main import preprocess_data
from preprocesso.mapped_dati import ValidationMapper
from metriche import Metrics
from metriche import Visualizer

def main():
    print("Benvenuto! Questo programma ti aiuterà ad analizzare un dataset di cellule attraverso diverse tecniche di validazione")
    
    try:
        # 1 Acquisizione del percorso del file
        file_path = input("Inserisci il percorso del file del dataset: ").strip()
        if not file_path:
            print("Percorso non fornito. Utilizzo del file di default: 'Data/version_1.csv'")
            file_path = 'Data/version_1.csv'
        
        # 2 Preprocessing dei dati (lettura, pulizia, scaling)
        print("Sto analizzando il dataset...")
        success, features, labels, scaled_data = preprocess_data(file_path)

        # Se il preprocessing non ha avuto successo, solleva un'eccezione
        if not success:
            raise ValueError("Preprocessing fallito: il dataset non è valido.")
        
        # 3 Configurazione della validazione KNN
        k, strategy = setup_knn_validation(scaled_data)
        
        # 4 Suddivisione del dataset per la validazione
        print(f"Generazione delle divisioni utilizzando la strategia: {strategy.__class__.__name__}...")
        validation_data = strategy.split_data(features, labels, k)
        
        # 5 Mappatura delle etichette per il calcolo delle metriche
        mapped_data = ValidationMapper.map_static(validation_data)

        # 6 Creazione dell'oggetto per la visualizzazione delle metriche
        metriche_selezionate = Metrics.scegli_metriche()
        
    
        # 7 Creazione dell'oggetto per la visualizzazione delle metriche
        visualizzatore = Visualizer(mapped_data, metriche_selezionate)

        # 8 Calcolo e visualizzazione delle metriche
        visualizzatore.visualize_metrics()

        # 9 Salvataggio delle metriche in un file Excel
        visualizzatore.save("metrics_output.xlsx")


    except FileNotFoundError:
        print("Errore: Il file specificato non esiste. Controlla il percorso e riprova.")
    except ValueError as e:
        print(f"Errore di valore: {e}")
        traceback.print_exc()  
    except Exception as e:
        print(f"Si è verificato un errore imprevisto: {e}")
        traceback.print_exc()  

if __name__ == "__main__":
    main()