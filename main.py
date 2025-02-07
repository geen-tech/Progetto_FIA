import pandas as pd
from preprocesso.file_parser import ParserDispatcher
from preprocesso.missing_data_manager import MissingDataStrategyManager
from preprocesso.feature_transformer import FeatureTransformationManager
from validazione import Holdout, RandomSubsampling, StratifiedValidation
from metriche import Visualizer

def map_validation(validation_data, mapping={2: 0, 4: 1}):
    """
    Trasforma validation_data mappando i valori secondo il dizionario fornito

    Args:
        validation_data (list of tuples): Lista di tuple ([y_real], [y_pred])
        mapping (dict): Dizionario che definisce la mappatura delle classi

    Returns:
        list of tuples: Nuova lista con i valori mappati.

    Esempio:
        -> validation_data = [([2, 4, 2, 4], [4, 4, 2, 2])]
        -> map_validation(validation_data) = [([0, 1, 0, 1], [1, 1, 0, 0])]
    """
    mapped_validation_data = [
        (
            [mapping.get(x, x) for x in y_real],  # Mappa y_real
            [mapping.get(x, x) for x in y_pred]   # Mappa y_pred
        )
        for y_real, y_pred in validation_data
    ]
    
    return mapped_validation_data

def main():
    print("Benvenuto! Questo programma ti aiuterà ad analizzare un dataset di cellule attraverso diverse tecniche di validazione")

    # Step 1: Inserimento percorso del file
    file_path = input("Inserisci il percorso del file del dataset: ").strip()
    if not file_path:
        print("Percorso non fornito. Utilizzo del file di default: 'Data/version_1.csv'")
        file_path = 'Data/version_1.csv'
        print(file_path)
    
    print("Sto analizzando il dataset...")

    try:
        parser = ParserDispatcher.get_parser(file_path)
        data = parser.parse_file(file_path) # Lettura del file
    # Se c'è un errore nella lettura del file, viene mostrato un messaggio all'utente
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}. Verrà utilizzato un dataset vuoto.") # Il codice termina al passo successivo
        # Invece di terminare subito, si assegna un dataset vuoto a 'data'
        data = pd.DataFrame()
    # Dopo il parsing, controlliamo esplicitamente se il dataset è vuoto
    if data.empty:
        print("Il dataset è vuoto. Termino l'esecuzione.")
        return # Termina il programma qui
    
    # Se il dataset non è vuoto, il programma continua con altre operazioni...
    print("Dati iniziali:")
    # Mostra le prime righe del dataset, così da verificare che il caricamento del dataset sia corretto
    print(data.head())

    # Step 2: L'utente decide come comprtarsi con i valori mancanti
    print("Come vuoi gestire i valori mancanti?")
    print("Hai quattro possibilità: remove | mode | mean | median ")
    attempts=0
    max_attempts=2

    while attempts < max_attempts:
        missing_strategy = input("Inserisci la tua scelta ").strip().lower()
        if missing_strategy in ['remove', 'mode', 'mean', 'median']:
            print(f"Hai scelto: {missing_strategy}")  # Messaggio di conferma
            break # Input valido, vai avanti 
        attempts += 1
        if attempts != max_attempts:
            print(f"Scelta non valida. Hai {max_attempts - attempts} tentativi rimanenti.")
    
    # Se l'utente ha superato i tentativi, assegna 'remove' di default
    if attempts == max_attempts:
        print("Hai superato il numero massimo di tentativi. Verrà utilizzata la strategia 'remove' per default.")
        missing_strategy = 'remove'

    try:
        data = MissingDataStrategyManager.handle_missing_data(strategy=missing_strategy, data=data)
    except Exception as e:
        print(f"Errore durante la gestione dei valori mancanti: {e}. Procedo con i dati originali.")

    # Step 3: L'utente sceglie il Feature Scaling
    attempts = 0
    max_attempts = 2  # La richiesta di inserimento della tecnica può essere fatta massimo due volte
    print("Come vuoi svolgere lo scaling delle fearures: normalize | standardize? ")

    while attempts < max_attempts:
        feature_scaling = input("Inserisci la tua scelta ").strip().lower()
        if feature_scaling in ['standardize', 'normalize']:
            scaling_strategy = feature_scaling  # Assegna la scelta finale
            print(f"Hai scelto: {scaling_strategy}")  # Messaggio di conferma
            break  # Input valido, esce dal loop
        attempts += 1
        if attempts != max_attempts:
            print(f"Scelta non valida. Hai {max_attempts - attempts} tentativi rimanenti.")

    # Se l'utente ha superato i tentativi, assegna 'normalize' di default
    if attempts == max_attempts:
        print("Hai superato il numero massimo di tentativi. Verrà utilizzata la strategia 'normalize' per default.")
        feature_scaling = 'normalize'
        scaling_strategy = feature_scaling

    ignored_columns = ['Sample code number', 'classtype_v1']
    try:
        scaled_data = FeatureTransformationManager.apply_transformation(strategy=scaling_strategy, data=data, skip_columns=ignored_columns)
    except Exception as e:
        print(f"Errore durante lo scaling delle feature: {e}. Utilizzo dei dati senza scaling")
        scaled_data = data

    print("Dati dopo lo scaling:")
    print(scaled_data.head()) # Mostra le prime righe del dataset, per verificare che il procedimento sia eseguito correttamente

    # Separazione delle feature e delle etichette
    kind_cell_column = 'classtype_v1'  
    if kind_cell_column not in scaled_data.columns:
        print(f"La colonna delle etichette '{kind_cell_column}' non è presente nel dataset. Termino l'esecuzione.")
        return

    labels = scaled_data[kind_cell_column]
    features = scaled_data.drop(columns=ignored_columns, errors='ignore')

    # Step 4: Scelta della strategia di validazione e KNN
    attempts=0
    max_attempts=2

    while attempts < max_attempts:
        k_utente = input("Inserisci il valore di k per il KNN (numero di vicini da utilizzare): ").strip()
        # Verifica se l'input è un numero
        try:
            k = int(k_utente)
            # Controlla che k sia maggiore di 0 e minore di 100
            if k <= 0:
                attempts += 1
                if attempts != max_attempts:
                    print(f"k deve essere maggiore di 0. Hai {max_attempts - attempts} tentativi rimanenti. Riprova.")
                    continue
            elif k >= len(scaled_data):
                attempts += 1
                if attempts != max_attempts:
                    print(f"k deve essere maggiore di 0. Hai {max_attempts - attempts} tentativi rimanenti. Riprova.")
                    continue
            
            # Se il valore di k è valido, esci dal ciclo e assegna il valore
            break
            
        except ValueError:
            attempts += 1
            if attempts != max_attempts:
                print(f"Valore di k non valido. Devi inserire un numero intero valido. Hai {max_attempts - attempts} tentativi rimanenti.")
                    
            
    if attempts == max_attempts:
        print("Hai superato il numero massimo di tentativi. Il numero di vicin sarà assegnato di default uguale  a 5.")
        k=5
    
    # Step 5: Decidere la tecnica di validazione
    print("Scegli la strategia di validazione:")
    print("A. Holdout")
    print("B. Random Subsampling")
    print("C. Stratified Validation")
    method = input("Inserisci la lettera corrispondente al metodo desiderato (A per Hold-Out, B per Random Subsampling, C per Stratified Shuffle Split): ").upper()
    strategy = None

    while True:
        try:

            if method == 'A':  # Holdout
                training_size = input("Inserisci la percentuale di training (il valore deve essere compreso tra 0 e 1): ").strip()
                training_size = float(training_size) if training_size else 0.8
                test_size = 1 - training_size
                strategy = Holdout(test_size=test_size)

            elif method == 'B':  # Random Subsampling
                iterazioni = input("Inserisci il numero di iterazioni (default 5): ").strip()
                iterazioni = int(iterazioni) if iterazioni else 5

                training_size = input("Inserisci la percentuale di training (il valore deve essere compreso tra 0 e 1): ").strip()
                training_size = float(training_size) if training_size else 0.8
                test_size = 1 - training_size
                strategy = RandomSubsampling(iterazioni=iterazioni, test_size=test_size)

            elif method == 'C':  # Stratified Validation
                iterazioni = input("Inserisci il numero di iterazioni (default 5): ").strip()
                iterazioni = int(iterazioni) if iterazioni else 5

                training_size = input("Inserisci la percentuale di training (il valore deve essere compreso tra 0 e 1): ").strip()
                training_size = float(training_size) if training_size else 0.8
                test_size = 1 - training_size
                strategy = StratifiedValidation(iterazioni=iterazioni, test_size=test_size)
            
            else:  # Scelta non valida
                print("Scelta non valida. Uso Holdout come default con percentuale di test: 0.2.")
                strategy = Holdout(test_size=0.2)

            break  
        except ValueError as e:
            print(f"Errore: {e}. Riprova con valori validi.")

    # Generazione delle divisioni
    print(f"Generazione delle divisioni utilizzando la strategia: {strategy.__class__.__name__}...")
    validation_data = strategy.split_data(features, labels, k)

    # Mappa i valori 2 -> 0 e 4 -> 1 per le etichette reali e predette per calcolare le metriche
    mapped_validation_data = map_validation(validation_data)

    
    # Creazione dell'oggetto PerformanceMetricsVisualizer
    visualizzatore = Visualizer(mapped_validation_data)

    # Calcolo e visualizzazione delle metriche
    visualizzatore.visualize_metrics()

    # Salvataggio delle metriche in un file Excel
    visualizzatore.save("metrics_output.xlsx")

if __name__ == "__main__":
    main()  
        

