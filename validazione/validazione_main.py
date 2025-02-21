import pandas as pd
from validazione import Holdout, RandomSubsampling, StratifiedValidation

class KNNValidation_main:
    def __init__(self):
        self.k = 5  # Valore di default
        self.strategy = None
        self.K=None

    def get_k_value(self, scaled_data,  max_attempts=2):
        attempts = 0
        while attempts < max_attempts:
            k_utente = input("Inserisci il valore di k per il KNN (numero di vicini da utilizzare): ").strip()
            try:
                k = int(k_utente)
                if k <= 0 or k >= len(scaled_data):
                    attempts += 1
                    if attempts != max_attempts:
                        print(f"k deve essere maggiore di 0 e minore della lunghezza del data frame. Hai {max_attempts - attempts} tentativi rimanenti. Riprova.")
                        continue
                else:
                    self.k = k
                    return
            except ValueError:
                attempts += 1
                if attempts != max_attempts:
                    print(f"Valore di k non valido. Devi inserire un numero intero valido. Hai {max_attempts - attempts} tentativi rimanenti.")
                    
        if attempts == max_attempts:
            print("Hai superato il numero massimo di tentativi. Il valore di k sarà impostato a 5.")
            self.k = 5

    def get_validation_strategy(self):
        print("Scegli la strategia di validazione:")
        print("A. Holdout")
        print("B. Random Subsampling")
        print("C. Stratified Validation")
        method = input("Inserisci la lettera corrispondente al metodo: ").upper()
        
        try:
            if method == 'A':
                training_size = float(input("Inserisci la percentuale di training (0-1, default 0.8): ") or 0.8)
                self.strategy = Holdout(test_size=1 - training_size)
            elif method == 'B':
                iterazioni = int(input("Inserisci il numero di iterazioni (default 5): ") or 5)
                training_size = float(input("Inserisci la percentuale di training (0-1, default 0.8): ") or 0.8)
                self.strategy = RandomSubsampling(iterazioni=iterazioni, test_size=1 - training_size)
            elif method == 'C':
                iterazioni = int(input("Inserisci il numero di iterazioni (default 5): ") or 5)
                training_size = float(input("Inserisci la percentuale di training (0-1, default 0.8): ") or 0.8)
                self.strategy = StratifiedValidation(iterazioni=iterazioni, test_size=1 - training_size)
            else:
                print("Scelta non valida. Uso Holdout con test_size=0.2 di default.")
                self.strategy = Holdout(test_size=0.2)
        except ValueError:
            print("Errore nell'inserimento dei parametri. Uso Holdout con test_size=0.2 di default.")
            self.strategy = Holdout(test_size=0.2)
    
    def run(self):
        self.get_k_value()
        self.get_validation_strategy()
        return self.k, self.strategy
    

def setup_knn_validation(scaled_data):
    """
    Imposta la validazione KNN e restituisce l'istanza di KNNValidation_main con i valori selezionati.
    """
    selector = KNNValidation_main()
    selector.get_k_value(scaled_data)
    selector.get_validation_strategy()
    return selector.k, selector.strategy