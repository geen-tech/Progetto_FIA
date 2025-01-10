def main():
    print("Benvenuto! Questo programma ti aiuter\u00e0 a selezionare un metodo di cross-validation.")

    # Chiedi all'utente di selezionare un numero per k piccolo
    while True:
        try:
            k = int(input("Per favore, inserisci un numero per k piccolo: "))
            if k > 0:
                break
            else:
                print("Inserisci un numero positivo.")
        except ValueError:
            print("Valore non valido. Inserisci un numero intero.")

    print("\nSeleziona il metodo di cross-validation:")
    print("A - Hold-Out")
    print("B - Random Subsampling")
    print("C - Stratified Shuffle Split")

    # Chiedi all'utente di selezionare il metodo di cross-validation
    while True:
        method = input("Inserisci la lettera corrispondente al metodo desiderato (A per Hold-Out, B per Random Subsampling, C per Stratified Shuffle Split): ").upper()
        if method in ['A', 'B', 'C']:
            break
        else:
            print("Valore non valido. Inserisci A, B, o C.")

    # Stampa un messaggio in base al metodo selezionato
    if method == 'A':
        print("Hai selezionato il metodo Hold-Out.")
    elif method == 'B':
        print("Hai selezionato il metodo Random Subsampling.")
        from subsampling import random_subsampling  # Importa la funzione da un altro file
        random_subsampling(k)  # Richiama la funzione con il valore di k
    elif method == 'C':
        print("Hai selezionato il metodo Stratified Shuffle Split.")

if __name__ == "__main__":
    main()
