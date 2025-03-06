import os  # Libreria per la gestione dei percorsi e dei file
import numpy as np  # Libreria per la manipolazione di array numerici
import matplotlib.pyplot as plt  # Libreria per la visualizzazione dei dati
import joblib  # Per salvare e caricare il modello addestrato
from datetime import datetime  # Per ottenere la data e l'orario attuali
from sklearn.datasets import load_iris  # Dataset di esempio
from sklearn.model_selection import train_test_split  # Per suddividere i dati
from sklearn.metrics import accuracy_score  # Per calcolare l'accuratezza del modello
from sklearn.neighbors import KNeighborsClassifier  # Modello KNN

# =============================
# Definizione delle directory per il modello salvato e i grafici
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory base dello script
MODELS_DIR = os.path.join(BASE_DIR, '../models')  # Directory dei modelli salvati
PLOTS_DIR = os.path.join(BASE_DIR, '../plots')  # Directory per salvare i grafici
os.makedirs(PLOTS_DIR, exist_ok=True)  # Crea la directory se non esiste

# =============================
# Funzione per caricare e suddividere i dati
# =============================
def load_data():
    """Carica il dataset Iris e lo suddivide in training e test set."""
    iris = load_iris()  # Carica il dataset
    X, y = iris.data, iris.target  # Divide i dati dalle etichette
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Suddivide il dataset

# =============================
# Funzione per generare e salvare il grafico dell'accuratezza rispetto a K
# =============================
def plot_accuracy(X_train, X_test, y_train, y_test):
    """Genera un grafico dell'accuratezza per diversi valori di K e lo salva."""
    k_values = np.arange(1, 21)  # Definisce il range di K da testare
    train_accuracies = []  # Lista per le accuratezze sul training set
    test_accuracies = []  # Lista per le accuratezze sul test set

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)  # Inizializza il modello KNN con valore K
        knn.fit(X_train, y_train)  # Addestra il modello
        train_accuracies.append(accuracy_score(y_train, knn.predict(X_train)))  # Calcola accuratezza sul training set
        test_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))  # Calcola accuratezza sul test set

    # Creazione del grafico
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_accuracies, marker='o', linestyle='dashed', color='g', label='Training Accuracy')  # Grafico per il training set
    plt.plot(k_values, test_accuracies, marker='s', linestyle='dashed', color='r', label='Test Accuracy')  # Grafico per il test set
    plt.xlabel('Numero di vicini (K)')  # Etichetta asse X
    plt.ylabel('Accuratezza')  # Etichetta asse Y
    plt.title('Performance del modello KNN per differenti valori di K')  # Titolo del grafico
    plt.xticks(k_values)  # Imposta i tick dell'asse X
    plt.legend()  # Aggiunge la legenda
    plt.grid()  # Mostra la griglia
    
    # Salvataggio del grafico prima di chiudere la figura
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Ottiene la data e l'orario attuale
    filename = os.path.join(PLOTS_DIR, f"accuracy_plot_{timestamp}.png")  # Nome del file con timestamp
    plt.savefig(filename)  # Salva il grafico
    print(f"Grafico salvato in: {filename}")  # Stampa il percorso del file salvato
    plt.close()  # Chiude la figura per evitare che si sovrappongano

# =============================
# Funzione per mostrare e salvare la distribuzione dei dati
# =============================
def plot_data_distribution():
    """Genera una matrice di scatter plot per mostrare la distribuzione di tutte le feature del dataset Iris e la salva."""
    iris = load_iris()  # Carica il dataset
    X, y = iris.data, iris.target  # Divide i dati dalle etichette
    feature_names = iris.feature_names  # Nomi delle feature
    target_names = iris.target_names  # Nomi delle classi
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))  # Crea una griglia di sottografici
    fig.suptitle('Distribuzione delle feature del dataset Iris', fontsize=16)  # Titolo generale
    
    for i in range(4):
        for j in range(4):
            if i == j:
                axes[i, j].hist(X[:, i], bins=20, color='gray', alpha=0.7)  # Istogramma della feature
                axes[i, j].set_xlabel(feature_names[i])  # Etichetta dell'asse X
            else:
                for k, label in enumerate(target_names):
                    axes[i, j].scatter(X[y == k, j], X[y == k, i], label=label, alpha=0.6)  # Scatter plot tra due feature
                if j == 0:
                    axes[i, j].set_ylabel(feature_names[i])  # Etichetta dell'asse Y
                if i == 3:
                    axes[i, j].set_xlabel(feature_names[j])  # Etichetta dell'asse X
    
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')  # Aggiunge la legenda globale
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adatta il layout
    
    # Salvataggio del grafico prima di chiudere la figura
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Ottiene la data e l'orario attuale
    filename = os.path.join(PLOTS_DIR, f"data_distribution_{timestamp}.png")  # Nome del file con timestamp
    plt.savefig(filename)  # Salva il grafico
    print(f"Grafico salvato in: {filename}")  # Stampa il percorso del file salvato
    plt.close()  # Chiude la figura per evitare che si sovrappongano

# =============================
# Punto di ingresso dello script (solo per test)
# =============================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()  # Carica i dati
    plot_accuracy(X_train, X_test, y_train, y_test)  # Mostra e salva il grafico dell'accuratezza
    plot_data_distribution()  # Mostra e salva la distribuzione dei dati