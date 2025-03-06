import os  # Libreria per la gestione dei percorsi e dei file
import numpy as np  # Libreria per la manipolazione di array numerici
import joblib  # Per salvare e caricare il modello addestrato
from sklearn.datasets import load_iris  # Dataset per ottenere i nomi delle classi

# =============================
# Definizione delle directory per il modello salvato
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory base dello script
MODELS_DIR = os.path.join(BASE_DIR, '../models')  # Directory dei modelli salvati

# =============================
# Caricamento del modello e dello scaler
# =============================
knn = joblib.load(os.path.join(MODELS_DIR, 'knn_model.pkl'))  # Carica il modello KNN
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))  # Carica lo scaler per normalizzare i dati

# Caricamento dei nomi delle classi
target_names = load_iris().target_names  # Ottiene i nomi delle classi dal dataset Iris

# =============================
# Funzione per effettuare una previsione
# =============================
def predict(sample):
    """Effettua una previsione sulla base di un nuovo campione di dati."""
    sample = np.array(sample).reshape(1, -1)  # Converte il campione in un array numpy e lo ridimensiona
    sample_scaled = scaler.transform(sample)  # Standardizza i dati per allinearli con il modello
    prediction = knn.predict(sample_scaled)  # Effettua la previsione con il modello KNN
    return target_names[prediction][0]  # Restituisce il nome della classe predetta

# =============================
# Punto di ingresso dello script
# =============================
if __name__ == "__main__":
    print("🔹 Inserisci i valori per la previsione (4 valori separati da spazio):")
    try:
        user_input = list(map(float, input("Valori: ").split()))  # Legge e converte i valori di input
        if len(user_input) != 4:
            raise ValueError("Devono essere forniti esattamente 4 valori.")  # Controlla il numero di input
        result = predict([user_input])  # Effettua la previsione
        print(f'Predizione per il nuovo campione: {result}')  # Stampa il risultato
    except ValueError as e:
        print(f"Errore di input: {e}. Riprova.")  # Gestisce eventuali errori di input