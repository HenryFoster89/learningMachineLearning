import numpy as np  # Libreria per la manipolazione di array numerici
import os  # Libreria per la gestione dei percorsi e dei file
import joblib  # Per salvare e caricare il modello addestrato
from sklearn.datasets import load_iris  # Dataset di esempio
from sklearn.model_selection import train_test_split, GridSearchCV  # Per suddivisione e ricerca iperparametri
from sklearn.preprocessing import StandardScaler  # Per standardizzare i dati
from sklearn.neighbors import KNeighborsClassifier  # Modello KNN
from sklearn.metrics import accuracy_score  # Per valutare il modello

# =============================
# Definizione delle directory per il salvataggio del modello
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory base dello script
MODELS_DIR = os.path.join(BASE_DIR, '../models')  # Directory per salvare i modelli
os.makedirs(MODELS_DIR, exist_ok=True)  # Crea la directory se non esiste

# =============================
# Funzione per addestrare il modello
# =============================
def train_model():
    """Addestra il modello KNN e lo salva su disco."""
    # Caricamento del dataset
    iris = load_iris()  # Carica il dataset Iris
    X, y = iris.data, iris.target  # Divide i dati dalle etichette

    # Suddivisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratifica per bilanciare le classi

    # Standardizzazione delle feature
    scaler = StandardScaler()  # Inizializza lo standardizzatore
    X_train = scaler.fit_transform(X_train)  # Adatta e trasforma il training set
    X_test = scaler.transform(X_test)  # Trasforma il test set con gli stessi parametri

    # =============================
    # Ricerca del miglior valore di K
    # =============================
    param_grid = {'n_neighbors': np.arange(1, 21)}  # Definisce il range di K da testare
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')  # Esegue la ricerca con validazione incrociata
    grid_search.fit(X_train, y_train)  # Addestra il modello con tutti i valori di K
    best_k = grid_search.best_params_['n_neighbors']  # Ottiene il valore di K ottimale
    print(f'Miglior valore di K: {best_k}')  # Stampa il miglior valore di K

    # =============================
    # Addestramento del modello con il miglior K
    # =============================
    knn = KNeighborsClassifier(n_neighbors=best_k)  # Inizializza il modello con il miglior K
    knn.fit(X_train, y_train)  # Addestra il modello

    # =============================
    # Salvataggio del modello e dello scaler
    # =============================
    joblib.dump(knn, os.path.join(MODELS_DIR, 'knn_model.pkl'))  # Salva il modello addestrato
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))  # Salva lo scaler per future previsioni

    print("Modello e scaler salvati con successo!")  # Messaggio di conferma

# =============================
# Punto di ingresso dello script
# =============================
if __name__ == "__main__":
    train_model()  # Esegue l'addestramento del modello