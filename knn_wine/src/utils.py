import os, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



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
    
    
    
def eda_chart(df, suffix = "prescaled"):
    # boxp PER FEATURES ESCLUDENDO MAGNESIO, PROLINE E ALCALINITY OF ASH
    df_long = df.copy().drop(["Magnesium", "Proline", "Alcalinity of ash"], axis = 1)
    df_long = df_long.melt(id_vars=["Class"])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = sns.boxplot(data = df_long, x = "value", y = "variable", orient="h", hue = "Class", palette="Set2")
    plt.tight_layout()

    plt.savefig(rf"chart\PP_boxp1_{suffix}.png")
    plt.close()

    # boxp PER FEATURES MAGNESIO, PROLINE E ALCALINITY OF ASH
    df_long = df.copy()[["Class", "Magnesium", "Proline", "Alcalinity of ash"]]
    df_long = df_long.melt(id_vars=["Class"])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = sns.boxplot(data = df_long, x = "value", y = "variable", orient="h", hue = "Class", palette="Set2")
    plt.tight_layout()

    plt.savefig(rf"chart\PP_boxp2_{suffix}.png")
    plt.close()

    # pairp
    sns.pairplot(df, hue="Class", diag_kind="kde", palette="Set2")
    plt.savefig(rf"chart\PP_pairp_{suffix}.png")
    plt.close()
    
    # heatm
        # Calcoliamo la matrice di correlazione
    corr_matrix = df.drop(columns=["Class"]).corr()

    # Calcoliamo la matrice di correlazione
    corr_matrix = df.drop(columns=["Class"]).corr()

    # Creiamo una maschera per nascondere la parte superiore della matrice
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Creiamo la heatm senza la parte superiore
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", 
                linewidths=0.5, vmin=-1, vmax=1, cbar_kws={"shrink": 0.8})

    plt.title("heatm delle Correlazioni tra le Feature (Senza Ridondanze)")

    plt.savefig(rf"chart\PP_heatm_{suffix}.png")
    plt.close()





def pca_plot(df, suffix = "prescaled"):
# Applichiamo PCA riducendo a 2 componenti principali
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df.drop(columns=["Class"]))

    # Creiamo un DataFrame con le nuove componenti principali
    df_pca = pd.DataFrame(df_pca, columns=["PC1", "PC2"])
    df_pca["Class"] = df["Class"]
    
    print(pca.explained_variance_ratio_.cumsum())
    
    # Scatterplot delle due componenti principali
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Class", palette="Set2")
    plt.title(f"PCA: Proiezione delle prime due componenti principali {pca.explained_variance_ratio_.cumsum()}")
    plt.savefig(rf"chart\pca_{suffix}.png")
    plt.close()
