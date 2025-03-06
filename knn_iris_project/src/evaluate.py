import os  # Libreria per la gestione dei percorsi e dei file
import joblib  # Per salvare e caricare il modello addestrato
from sklearn.metrics import accuracy_score  # Per calcolare l'accuratezza del modello
from sklearn.datasets import load_iris  # Dataset di esempio
from sklearn.model_selection import train_test_split  # Divisione del dataset in training e test

# =============================
# Definizione delle directory per il modello salvato
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory base dello script
MODELS_DIR = os.path.join(BASE_DIR, '../models')  # Directory dei modelli

# =============================
# Funzione per valutare il modello salvato
# =============================
def evaluate_model():
    """Carica il modello salvato e calcola l'accuratezza sul test set."""
    # Caricamento del dataset Iris
    iris = load_iris()  # Carica il dataset
    X, y = iris.data, iris.target  # Divide i dati dalle etichette

    # Suddivisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratifica per bilanciare le classi

    # Caricamento del modello e dello scaler
    knn = joblib.load(os.path.join(MODELS_DIR, 'knn_model.pkl'))  # Carica il modello KNN
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))  # Carica lo scaler

    # Standardizzazione del test set
    X_test = scaler.transform(X_test)  # Applica la stessa trasformazione del training set

    # Valutazione del modello
    y_pred = knn.predict(X_test)  # Effettua la previsione sui dati di test
    accuracy = accuracy_score(y_test, y_pred)  # Calcola l'accuratezza
    print(f'Accuratezza sul test set: {accuracy:.2f}')  # Stampa il risultato

# =============================
# Punto di ingresso dello script
# =============================
if __name__ == "__main__":
    evaluate_model()  # Esegue la valutazione del modello