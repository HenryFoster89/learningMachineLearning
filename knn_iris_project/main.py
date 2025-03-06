from src.train import train_model  # Importa la funzione per addestrare il modello
from src.evaluate import evaluate_model  # Importa la funzione per valutare il modello
from src.predict import predict  # Importa la funzione per effettuare previsioni
from src.utils import load_data, plot_accuracy, plot_data_distribution  # Importa funzioni di supporto per la gestione dei dati e la visualizzazione

# =============================
# Funzione principale del programma
# =============================
def main():
    while True:  # Avvia un loop infinito per mostrare il menu interattivo
        print("\nSeleziona un'opzione:")
        print("1 - Addestrare il modello")  # Opzione per addestrare il modello
        print("2 - Valutare il modello")  # Opzione per valutare il modello
        print("3 - Generare il grafico delle performance")  # Opzione per visualizzare le performance
        print("4 - Effettuare una previsione")  # Opzione per effettuare una previsione
        print("5 - Mostrare la distribuzione del dataset")  # Opzione per visualizzare la distribuzione del dataset
        print("6 - Eseguire tutti i passaggi")  # Opzione per eseguire l'intero workflow
        print("0 - Uscire")  # Opzione per uscire dal programma
        
        scelta = input("Inserisci il numero dell'opzione: ")  # Richiede all'utente di selezionare un'opzione
        
        # =============================
        # Opzione 1: Addestramento del modello
        # =============================
        if scelta == "1":
            print("Addestramento del modello...")
            train_model()  # Esegue la funzione per addestrare il modello
        
        # =============================
        # Opzione 2: Valutazione del modello
        # =============================
        elif scelta == "2":
            print("Valutazione del modello...")
            evaluate_model()  # Esegue la funzione per valutare il modello
        
        # =============================
        # Opzione 3: Visualizzazione della performance
        # =============================
        elif scelta == "3":
            print("Generazione del grafico delle performance...")
            X_train, X_test, y_train, y_test = load_data()  # Carica i dati
            plot_accuracy(X_train, X_test, y_train, y_test)  # Genera il grafico della performance
        
        # =============================
        # Opzione 4: Effettuare una previsione
        # =============================
        elif scelta == "4":
            print("Inserisci i valori per la previsione (4 valori separati da spazio):")
            print("Ordine delle feature: Sepal Length, Sepal Width, Petal Length, Petal Width")
            try:
                user_input = list(map(float, input("Valori (esempio: 5.1 3.5 1.4 0.2): ").split()))  # Legge e converte gli input dell'utente
                if len(user_input) != 4:
                    raise ValueError("Devono essere forniti esattamente 4 valori.")  # Controlla che ci siano esattamente 4 valori
                result = predict([user_input])  # Effettua la previsione con il modello
                print(f'Predizione per il nuovo campione: {result}')  # Stampa il risultato della previsione
            except ValueError as e:
                print(f"Errore di input: {e}. Riprova.")  # Messaggio di errore in caso di input errato
        
        # =============================
        # Opzione 5: Mostrare la distribuzione del dataset
        # =============================
        elif scelta == "5":
            print("Mostrando la distribuzione del dataset...")
            plot_data_distribution()  # Mostra la distribuzione del dataset
        
        # =============================
        # Opzione 6: Esecuzione completa del workflow
        # =============================
        elif scelta == "6":
            print("Esecuzione completa del workflow...")
            train_model()  # Addestra il modello
            evaluate_model()  # Valuta il modello
            X_train, X_test, y_train, y_test = load_data()  # Carica i dati
            plot_accuracy(X_train, X_test, y_train, y_test)  # Visualizza il grafico delle performance
            plot_data_distribution()  # Mostra la distribuzione del dataset
            print("Inserisci i valori per la previsione (4 valori separati da spazio):")
            print("Ordine delle feature: Sepal Length, Sepal Width, Petal Length, Petal Width")
            try:
                user_input = list(map(float, input("Valori (esempio: 5.1 3.5 1.4 0.2): ").split()))  # Input utente
                if len(user_input) != 4:
                    raise ValueError("Devono essere forniti esattamente 4 valori.")  # Controllo input
                result = predict([user_input])  # Effettua la previsione
                print(f'Predizione per il nuovo campione: {result}')  # Stampa il risultato
            except ValueError as e:
                print(f"Errore di input: {e}. Riprova.")  # Gestione errore
        
        # =============================
        # Opzione 0: Uscita dal programma
        # =============================
        elif scelta == "0":
            print("Uscita dal programma.")
            break  # Esce dal loop e termina il programma
        
        # =============================
        # Opzione non valida
        # =============================
        else:
            print("Opzione non valida. Riprova.")  # Messaggio di errore in caso di input non valido

# =============================
# Punto di ingresso del programma
# =============================
if __name__ == "__main__":
    main()  # Avvia la funzione principale