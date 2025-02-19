import pandas as pd
from .file_parser import ParserDispatcher
from .missing_data_manager import MissingDataStrategyManager
from .feature_transformer import FeatureTransformationManager


class DataPreprocessor:
    def _init_(self, file_path):
        self.file_path = file_path
        self.data = pd.DataFrame()
        self.scaled_data = pd.DataFrame()
        self.labels = None
        self.features = None
        self.ignored_columns = ['Sample code number', 'classtype_v1']
        self.kind_cell_column = 'classtype_v1'

    def load_data(self):
        """Carica i dati dal file specificato."""
        try:
            parser = ParserDispatcher.get_parser(self.file_path)
            self.data = parser.parse_file(self.file_path)  # Lettura del file
        except Exception as e:
            print(f"Errore durante la lettura del file: {e}. Verrà utilizzato un dataset vuoto.")
            self.data = pd.DataFrame()

        if self.data.empty:
            print("Il dataset è vuoto. Termino l'esecuzione.")
            return False  # Indica che il caricamento è fallito

        print("Dati iniziali:")
        print(self.data.head())  # Mostra le prime righe per verifica
        return True  # Indica che il caricamento è riuscito

    def handle_missing_values(self):
        """Gestisce i valori mancanti in base alla scelta dell'utente."""
        print("Come vuoi gestire i valori mancanti?")
        print("Hai quattro possibilità: remove | mode | mean | median")
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            missing_strategy = input("Inserisci la tua scelta ").strip().lower()
            if missing_strategy in ['remove', 'mode', 'mean', 'median']:
                print(f"Hai scelto: {missing_strategy}")
                break
            attempts += 1
            if attempts != max_attempts:
                print(f"Scelta non valida. Hai {max_attempts - attempts} tentativi rimanenti.")

        if attempts == max_attempts:
            print("Hai superato il numero massimo di tentativi. Verrà usata la strategia 'remove' per default.")
            missing_strategy = 'remove'

        try:
            self.data = MissingDataStrategyManager.handle_missing_data(strategy=missing_strategy, data=self.data)
        except Exception as e:
            print(f"Errore durante la gestione dei valori mancanti: {e}. Procedo con i dati originali.")

    def apply_feature_scaling(self):
        """Applica la normalizzazione o la standardizzazione ai dati."""
        print("Come vuoi svolgere lo scaling delle feature: normalize | standardize?")
        attempts = 0
        max_attempts = 2

        while attempts < max_attempts:
            feature_scaling = input("Inserisci la tua scelta ").strip().lower()
            if feature_scaling in ['standardize', 'normalize']:
                scaling_strategy = feature_scaling
                print(f"Hai scelto: {scaling_strategy}")
                break
            attempts += 1
            if attempts != max_attempts:
                print(f"Scelta non valida. Hai {max_attempts - attempts} tentativi rimanenti.")

        if attempts == max_attempts:
            print("Hai superato il numero massimo di tentativi. Verrà usata la strategia 'normalize' per default.")
            scaling_strategy = 'normalize'

        try:
            self.scaled_data = FeatureTransformationManager.apply_transformation(
                strategy=scaling_strategy, data=self.data, skip_columns=self.ignored_columns
            )
        except Exception as e:
            print(f"Errore durante lo scaling delle feature: {e}. Utilizzo dei dati senza scaling.")
            self.scaled_data = self.data

        print("Dati dopo lo scaling:")
        print(self.scaled_data.head())

        if self.kind_cell_column not in self.scaled_data.columns:
            print(f"Attenzione: la colonna '{self.kind_cell_column}' è stata persa durante lo scaling!")
            
    def prepare_features_and_labels(self):
        """Separa le feature dalle etichette e restituisce i dati scalati."""
        if self.kind_cell_column not in self.scaled_data.columns:
            print(f"La colonna delle etichette '{self.kind_cell_column}' non è presente. Termino l'esecuzione.")
            return False, None, None  

        self.labels = self.scaled_data[self.kind_cell_column]
        self.features = self.scaled_data.drop(columns=self.ignored_columns, errors='ignore')
            
        return True, self.features, self.labels, self.scaled_data  # Aggiunto scaled_data


def preprocess_data(file_path):
    """Carica, pulisce e scala i dati restituendo le feature, le etichette e il dataset scalato."""
    preprocessor = DataPreprocessor(file_path)

    if not preprocessor.load_data():
        return None  # Termina il processo in caso di errore

    preprocessor.handle_missing_values()
    preprocessor.apply_feature_scaling()
    
    return preprocessor.prepare_features_and_labels()