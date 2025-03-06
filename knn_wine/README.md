# 📌 Esercizio: Classificazione del Vino con KNN 🍷

## ✅ Perché scegliere il **Wine Dataset**?
- **Facile da usare** → Il dataset è **già pulito**, senza valori mancanti.
- **Dimensione gestibile** → Ha solo **13 feature** e **178 campioni**, quindi il training è veloce.
- **Perfetto per KNN** → Le feature numeriche ben distribuite aiutano KNN a funzionare bene.
- **Obiettivo chiaro** → Devi classificare il vino in **3 categorie** basandoti sulle caratteristiche chimiche.

## 📂 **Scaricare il dataset**
Il dataset è disponibile pubblicamente nel **UCI Machine Learning Repository**:
[Wine Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine)

Oppure puoi caricarlo direttamente in Python con **pandas**:
```python
import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
columns = ["Class", "Alcohol", "Malic_acid", "Ash", "Alcalinity", "Magnesium",
           "Total_phenols", "Flavanoids", "Nonflavanoid_phenols", "Proanthocyanins",
           "Color_intensity", "Hue", "OD280/OD315", "Proline"]
df = pd.read_csv(url, names=columns)
```

## 🛠️ **Passaggi per costruire il progetto KNN**
1️⃣ **Caricare e analizzare il dataset**
   - Stampare le prime righe con `df.head()`
   - Controllare valori mancanti con `df.isnull().sum()`
   - Visualizzare statistiche con `df.describe()`

2️⃣ **Suddividere i dati**
   - Separare features (`X`) e target (`y`).
   ```python
   X = df.iloc[:, 1:]  # Tutte le colonne tranne la prima
   y = df.iloc[:, 0]  # La prima colonna è la classe
   ```
   - Dividere in **train/test** con `train_test_split()`
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
   ```

3️⃣ **Normalizzare i dati**
   - KNN è sensibile alle scale delle feature, quindi **StandardScaler** è essenziale.
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   ```

4️⃣ **Applicare KNN e trovare il miglior valore di K**
   - **Usare GridSearchCV** per ottimizzare K.
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.model_selection import GridSearchCV
   param_grid = {'n_neighbors': range(1, 21)}
   grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
   grid_search.fit(X_train, y_train)
   best_k = grid_search.best_params_['n_neighbors']
   print(f'Miglior valore di K: {best_k}')
   ```

5️⃣ **Valutare il modello**
   ```python
   from sklearn.metrics import accuracy_score
   knn = KNeighborsClassifier(n_neighbors=best_k)
   knn.fit(X_train, y_train)
   y_pred = knn.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Accuratezza del modello: {accuracy:.2f}')
   ```

6️⃣ **Creare una funzione di previsione**
   ```python
   def predict_wine(sample):
       sample = scaler.transform([sample])  # Standardizza i dati
       prediction = knn.predict(sample)  # Effettua la previsione
       return prediction[0]  # Restituisce la classe predetta
   ```

## 🎯 **Obiettivo finale:**
- Creare uno script interattivo che permette di inserire le caratteristiche di un vino e predire la categoria.
- Salvare il modello addestrato con `joblib.dump()` e riutilizzarlo.
- Visualizzare le performance con un grafico dell'accuratezza rispetto a K.

🚀 **Ora tocca a te! Buon coding!**