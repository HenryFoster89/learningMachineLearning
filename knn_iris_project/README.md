# Progetto KNN Iris Classifier

## Introduzione
Questo progetto implementa un classificatore basato sull'algoritmo **K-Nearest Neighbors (KNN)** per classificare i fiori del dataset **Iris**. Il codice è strutturato in più moduli per garantire chiarezza e modularità.

---

## Struttura del Progetto
Il progetto è organizzato come segue:

```
knn_iris_project/
│── main.py              # Entry point per eseguire l'intero workflow
│── README.md            # Documentazione del progetto
│── requirements.txt     # Librerie necessarie
│
├───data/                # Contiene il dataset (se necessario)
│
├───models/              # Contiene il modello addestrato e lo scaler
│   ├── knn_model.pkl    # Modello KNN salvato
│   ├── scaler.pkl       # Standardizzatore salvato
│
├───src/                 # Codice sorgente
│   ├── train.py         # Addestramento del modello
│   ├── evaluate.py      # Valutazione del modello
│   ├── predict.py       # Predizione su nuovi dati
│   ├── utils.py         # Funzioni di supporto (caricamento dati, plotting)
│
├───notebooks/           # Jupyter Notebook per test ed esplorazione
│
└───tests/               # Script di test per verificare il codice
```

---

## Workflow Informatico
Il flusso di lavoro del progetto è gestito da `main.py`, che funge da punto di ingresso per il sistema e permette all'utente di interagire con il modello KNN attraverso un'interfaccia testuale. Ogni modulo esegue un compito specifico nel pipeline di machine learning. Di seguito è descritto il funzionamento di ogni file:

### **`train.py` – Addestramento del Modello**
1. **Caricamento del dataset**: Il dataset Iris viene caricato usando `load_iris()` dalla libreria `sklearn.datasets`.
2. **Suddivisione in training e test set**: `train_test_split()` divide i dati in 80% training e 20% test per garantire una valutazione realistica del modello.
3. **Standardizzazione delle feature**: `StandardScaler()` normalizza i dati per evitare che feature con scale diverse influiscano negativamente sulle distanze calcolate dal KNN.
4. **Ricerca del miglior valore di K**: `GridSearchCV()` testa diversi valori di K con validazione incrociata per trovare il valore ottimale.
5. **Addestramento del modello KNN**: Viene istanziato `KNeighborsClassifier()` con il miglior valore di K e addestrato sui dati di training.
6. **Salvataggio del modello**: Il modello e lo scaler vengono salvati in `models/` utilizzando `joblib.dump()` per essere riutilizzati in fase di inferenza.

### **`evaluate.py` – Valutazione del Modello**
1. **Caricamento del dataset**: Lo stesso dataset viene ricaricato per garantire coerenza con l'addestramento.
2. **Caricamento del modello addestrato**: `joblib.load()` recupera il modello KNN precedentemente salvato.
3. **Standardizzazione del test set**: I dati di test vengono normalizzati con lo stesso scaler usato in fase di training.
4. **Predizione sul test set**: `knn.predict(X_test)` restituisce le etichette predette per i dati di test.
5. **Calcolo dell'accuratezza**: `accuracy_score(y_test, y_pred)` misura la performance del modello confrontando le predizioni con le etichette reali.

### **`predict.py` – Predizione su Nuovi Dati**
1. **Caricamento del modello e dello scaler**: `joblib.load()` recupera il modello e lo scaler precedentemente salvati.
2. **Definizione della funzione di inferenza**: `predict(sample)` prende in input una nuova osservazione, la standardizza e restituisce la classe predetta.
3. **Input utente**: L'utente può inserire manualmente i valori per effettuare una previsione.
4. **Esecuzione della predizione**: La classe predetta viene restituita e stampata a schermo.

### **`utils.py` – Funzioni di Supporto**
1. **Caricamento dei dati**: `load_data()` fornisce una funzione per caricare e suddividere il dataset.
2. **Visualizzazione delle performance del modello**: `plot_accuracy()` genera un grafico che mostra l'accuratezza del modello per diversi valori di K, utile per l'analisi del comportamento del classificatore.
3. **Visualizzazione della distribuzione dei dati**: `plot_data_distribution()` genera un grafico che mostra la distribuzione delle classi nel dataset originale, utile per analizzare la bilanciatura delle classi.

### **`main.py` – Interfaccia Utente e Coordinazione**
1. **Menu interattivo**: Presenta un menu testuale che permette all'utente di selezionare quale operazione eseguire (addestramento, valutazione, predizione o workflow completo).
2. **Gestione dell'input utente**: L'utente può digitare un'opzione e il programma eseguirà il corrispondente modulo.
3. **Chiamata ai moduli specifici**: Quando l'utente seleziona un'operazione, `main.py` richiama la funzione corrispondente importata dagli altri file.

**Esecuzione del progetto:**
```bash
python main.py
```

Questo flusso garantisce modularità e riusabilità, separando le fasi di addestramento, valutazione e inferenza in moduli distinti, facilitando la manutenzione del codice e l'espandibilità futura.

---

## Conclusioni
Questo progetto fornisce un'implementazione completa e modulare di KNN applicato al dataset Iris. Grazie alla sua struttura, è facilmente estendibile per altri dataset e ottimizzabile per migliorare le prestazioni.

Se hai domande o vuoi miglioramenti, sentiti libero di contribuire.

