import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Definizione delle directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODELS_DIR = os.path.join(BASE_DIR, '../models')

# Creazione delle cartelle se non esistono
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Caricamento del dataset
iris = load_iris()  # Carica il dataset Iris da sklearn
X = iris.data        # Matrice delle feature (4 caratteristiche per fiore)
y = iris.target      # Vettore delle etichette (classi delle specie di fiori)

# 2. Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# train_test_split suddivide i dati in 80% train e 20% test   <-- Informatica: suddivisione dataset
# stratify=y mantiene la distribuzione delle classi           <-- Statistica: garantisce equa rappresentazione delle classi

# 3. Standardizzazione delle feature
scaler = StandardScaler()  # Inizializza lo standardizzatore
X_train = scaler.fit_transform(X_train)                          # Calcola media e deviazione standard sui dati di training
X_test = scaler.transform(X_test)                                # Applica la trasformazione ai dati di test
# StandardScaler normalizza i dati (media 0, varianza 1)      <-- Statistica: normalizzazione migliora performance KNN

# 4. Ricerca del miglior valore di K
param_grid = {'n_neighbors': np.arange(1, 21)}  # Testiamo K da 1 a 20
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
# GridSearchCV testa diversi K con validazione incrociata     <-- Informatica: ottimizzazione iperparametri
# cv=5: suddivide il training set in 5 fold                   <-- Statistica: k-fold cross-validation migliora generalizzazione

best_k = grid_search.best_params_['n_neighbors']
print(f'Miglior valore di K: {best_k}')

# 5. Visualizzazione dell'accuratezza per diversi valori di K
k_values = np.arange(1, 21)
train_accuracies = []
test_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(k_values, train_accuracies, marker='o', linestyle='dashed', color='g', label='Training Accuracy')
plt.plot(k_values, test_accuracies, marker='s', linestyle='dashed', color='r', label='Test Accuracy')
plt.xlabel('Numero di vicini (K)')
plt.ylabel('Accuratezza')
plt.title('Performance del modello KNN per differenti valori di K')
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()
# Il grafico mostra l'accuratezza su training e test set      <-- Statistica: aiuta a rilevare overfitting o underfitting

# 6. Addestramento del modello con il miglior K
knn = KNeighborsClassifier(n_neighbors=best_k)  # Modello KNN con miglior valore di K
knn.fit(X_train, y_train)  # Addestramento del modello
# KNN memorizza i dati di training per le previsioni          <-- Informatica: algoritmo basato su memoria

# 7. Valutazione del modello
y_pred = knn.predict(X_test)  # Predizione sulle osservazioni di test
accuracy = accuracy_score(y_test, y_pred)  # Calcolo accuratezza
print(f'Accuratezza sul test set: {accuracy:.2f}')
# Accuracy = (predizioni corrette / totale test set)         <-- Statistica: metrica di valutazione

# 8. Salvataggio del modello
joblib.dump(knn, os.path.join(MODELS_DIR, 'knn_model.pkl'))  # Salva il modello KNN
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))  # Salva lo standardizzatore
# joblib permette di salvare e ricaricare modelli Python      <-- Informatica: serializzazione modelli ML

# 9. Caricamento del modello salvato e test su nuovi dati
knn_loaded = joblib.load(os.path.join(MODELS_DIR, 'knn_model.pkl'))  # Carica modello salvato
scaler_loaded = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))  # Carica scaler salvato

# Esempio di previsione su un nuovo dato
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Una nuova osservazione
sample_scaled = scaler_loaded.transform(sample)  # Standardizzazione
prediction = knn_loaded.predict(sample_scaled)  # Predizione
print(f'Predizione per il nuovo campione: {iris.target_names[prediction][0]}')
# Il modello predice la classe più vicina ai dati forniti      <-- Matematica: distanza euclidea tra punti
