# **Classificazione delle Cellule con k-NN**

## **Indice**
- [Panoramica del Progetto](#panoramica-del-progetto)
- [Dataset Utilizzato](#dataset-utilizzato)
- [Struttura del Codice](#struttura-del-codice)
- [Istruzioni per l'Esecuzione](#istruzioni-per-lesecuzione)
- [Interazione con lo Script](#interazione-con-lo-script)
  - [Gestione dei Valori Mancanti](#gestione-dei-valori-mancanti)
  - [Scaling delle Feature](#scaling-delle-feature)
  - [Selezione di k per il Classificatore k-NN](#selezione-di--k--per-il-classificatore-k-nn)
  - [Scelta della Strategia di Validazione](#scelta-della-strategia-di-validazione)
- [Metriche di Valutazione](#metriche-di-valutazione)
- [Output e Visualizzazione](#output-e-visualizzazione)
- [Conclusioni](#conclusioni-e-contributi)

---

## **Panoramica del Progetto**
L’obiettivo principale di questo progetto è offrire una **pipeline completa** per la classificazione di cellule, mirando a distinguere tra due possibili stati: benigno e maligno.
Attraverso l’uso di tecniche di validazione diversificate (Holdout, Random Subsampling e Stratified Shuffle Split) e del classificatore **k-Nearest Neighbors (k-NN)**, il codice fornisce un’analisi flessibile e dettagliata delle prestazioni del modello.

Nel contesto dell’oncologia, la distinzione tra cellule benigne e maligne è cruciale per una diagnosi tempestiva e accurata.
Per questo motivo, il progetto si focalizza su un dataset di riferimento che include misurazioni e indici relativi a cellule tumorali, con l’obiettivo di fornire un supporto decisionale solido ai professionisti.

---

## **Dataset Utilizzato**
Il file **`version_1.csv`** (presente nella cartella `Data/`) contiene i dati da analizzare:
- **Dimensione**: Circa 693 campioni.
- **Features Riporatte**:
  - `Sample code number`: Identificatore univoco della cellula.
  - `Clump Thickness`: Spessore del gruppo di cellule, densità delle cellule.
  - `Uniformity of Cell Size` / `Uniformity of Cell Shape`: Indicatori di regolarità cellulare.
  - `Bare Nucleix_wrong`: Numero di nuclei “scoperti” (potrebbe indicare un refuso).
  - `Blood Pressure`, `Heart Rate`: Variabili supplementari, non sempre direttamente correlate al tumore.
  - `classtype_v1`: Indica la classe di tumore (2 = benigno, 4 = maligno).
  - `Mitoses`: Indica il grado di proliferazione cellulare.
  - `Normal Nucleoli`: Rappresenta il nuemro di nucleoli standard nelle cellule.
  - `Single Epithelial Cell Size`: Indicatore della regolarità della cellula, basato sulla cellula epiteliale.
  - `Marginal Adhesion`: Capacità delle singole cellule di aderire tra loro.
  - `Bland Chromatin`: Rappresenta l'omogeneità della cromatina della cellula.

Nel codice, la colonna `Sample code number` viene utilizzata per:
1. Eliminare potenziali duplicati.
2. Impostare l’indice del `DataFrame`.
3. Individuare in modo univoco ogni riga del dataset.



---

## **Struttura del Codice**
Il file **`main.py`** rappresenta il **nucleo** dell’applicazione. All’interno, troviamo le seguenti operazioni principali:

1. **Caricamento del Dataset**:
   - Utilizza la classe `ParserDispatcher` per leggere il file nei formati supportati (`.csv`, `.xlsx`, `.json`, `.txt`, `.tsv`).
   - Se non viene fornito un percorso valido, il programma carica in automatico `Data/version_1.csv`.

2. **Pulizia dei Dati**:
   - Gestione dei valori mancanti (tramite `MissingDataStrategyManager`).

3. **Trasformazione delle Feature**:
   - Applicazione di tecniche di **Scaling** (normalizzazione o standardizzazione) mediante `FeatureTransformationManager`.

4. **Selezione del Classificatore e Validazione**:
   - L’utente specifica k per il metodo **k-NN**.
   - Scelta del tipo di validazione (Holdout, Random Subsampling o Stratified Validation).

5. **Calcolo e Visualizzazione delle Metriche**:
   - Utilizzo di `metrics` per calcolare e `visualizer` per mostrare le metriche come Accuracy Rate, Area Under Curve, Sensitivity, Geometric Mean, Specificity.
   - Salvataggio dei risultati in un file Excel (`metrics_output.xlsx`).

---

## **Istruzioni per l'Esecuzione**
1. **Clona o scarica** questa repository sul tuo computer.
2. Assicurati di avere **Python 3** installato (es. Python 3.8+).
3. Installa le dipendenze con:
   ```bash
   pip install -r requirements.txt 
   (Oppure, se preferisci, attiva un ambiente virtuale prima di installare.) 
4. Esegui il file principale:
   main.py
5. Interagisci con lo script seguendo i prompt testuali, che ti guideranno nella scelta delle strategie di pulizia del dataset, nella tecnica di scaling, nel valore di k per k-NN e nella metodologia di validazione.

## **Interazione con lo Script**

### **Gestione dei Valori Mancanti**
All’avvio, il programma chiederà come gestire eventuali valori mancanti. Potrai scegliere tra:
- **`remove`**: Elimina le righe contenenti valori NaN.
- **`mode`**: Rimpiazza i valori mancanti con quello più frequente nella colonna.
- **`mean`**: Sostituisce i valori mancanti con la media della colonna.
- **`median`**: Sostituisce i valori mancanti con la mediana della colonna.

Se le risposte non sono valide e vengono superati i tentativi concessi, la modalità di default diventerà `remove`.

### **Scaling delle Feature**
In seguito, il programma ti consente di scegliere tra:
- **`normalize`**: Ridimensiona i valori in un intervallo compreso tra 0 e 1.
- **`standardize`**: Normalizza i dati in base alla media 0 e deviazione standard 1.

Se la scelta non è valida, verrà applicato `normalize` come default.

### **Selezione di k per il Classificatore k-NN**
Verrà richiesto un valore di k (numero di vicini considerati dal classificatore).

- Deve essere un intero positivo e inferiore al numero di righe del dataset.
- Se inserito correttamente, lo script procederà; in caso contrario, dopo un certo numero di tentativi, verrà impostato un valore di default (ad es. k = 5).

### **Scelta della Strategia di Validazione**
Lo script propone tre approcci per la validazione del modello:

1. **Holdout**:
   - L’utente può indicare la percentuale di training (ad esempio 0.8), e di conseguenza il resto (0.2) sarà utilizzato come test.

2. **Random Subsampling**:
   - Si eseguono più divisioni casuali del dataset. L’utente sceglie quante iterazioni effettuare (ad es. 5) e la percentuale di training/test.

3. **Stratified Validation**:
   - Variante che preserva la proporzione delle classi (benigno/maligno) nelle suddivisioni, utile per dataset sbilanciati.
   - Anche in questo caso, si specifica il numero di iterazioni e la percentuale di training/test.

---

## **Metriche di Valutazione**

Una volta completata l’analisi, il programma calcola in automatico diverse metriche, tra cui:

- **Accuracy Rate**: Percentuale di campioni correttamente classificati.
- **Error Rate**: Percentuale di campioni classificati erroneamente.
- **Sensitivity** (o Recall): Misura la capacità di individuare correttamente i positivi (maligni).
- **Specificity**: Indica la capacità di classificare correttamente i negativi (benigni).
- **Geometric Mean**: Rappresenta l’equilibrio tra Sensitivity e Specificity.

Queste metriche forniscono una panoramica completa delle prestazioni, evidenziando punti di forza e debolezza del modello.

---

## **Output e Visualizzazione**

- **File Excel**: I risultati metrici (come Accuracy, Sensitivity, Specificity e Geometric Mean) vengono salvati in un file chiamato `metrics_output.xlsx`. Ogni riga del file corrisponde a una delle metriche chiave.
- **Grafici**: Viene generato un grafico a barre (o più grafici) grazie all’oggetto `Visualizer`. Tale grafico mostra visivamente i punteggi ottenuti, facilitando il confronto tra più esperimenti o strategie di validazione diverse.

> **Nota**: Se hai bisogno di conservare diversi scenari di test, è consigliabile rinominare o spostare il file `metrics_output.xlsx` generato prima di eseguire nuove prove, così da evitare sovrascritture.

---

## **Conclusioni**

Lo script `main.py` costituisce una soluzione **interattiva** e **configurabile** per l’analisi e la classificazione di cellule tumorali, coprendo l’intero flusso di lavoro: dalla pulizia dei dati alla validazione del modello, fino alla valutazione delle prestazioni. La modularità del codice permette di aggiungere facilmente nuovi metodi di validazione, strategie di preprocessing o classificatori differenti.
**Buone analisi!**
