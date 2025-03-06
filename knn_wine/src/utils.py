import os, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                  # __file__ is the absolute path of the current file | poi usa i due dirname per tornare indietro di due cartelle. abspath ti da il path assoluto del file
DATA_DIR = os.path.join(BASE_DIR, 'data')                                              # join ti permette di unire i due path



def load_wine_data():
    feature_names = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"]
    
    file_path = os.path.join(DATA_DIR, 'wine.data')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non è stato trovato.")
        
    return pd.read_csv(file_path, header=None, names = feature_names)              # carica il file wine.data e lo mette in un dataframe
df = load_wine_data()


def eda():
    
    print(df.shape)
    print(df.describe())
    print(df.info())
