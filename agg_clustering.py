import os
import shutil
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# ==========================================
# CONFIGURAZIONE
# ==========================================
# Inserisci qui il percorso della tua cartella con le foto originali.
CARTELLA_IMMAGINI_ORIGINALI = r"C:\Users\aless\OneDrive - Università di Napoli Federico II\OrchID\dataset_new_cropper" 
# ==========================================

def main():
    print("======================================================")
    print("   CLUSTERING GERARCHICO (AGGLOMERATIVE CLUSTERING)   ")
    print("======================================================\n")

    # 1. Ricerca automatica dei file .npz
    npz_files = [f for f in os.listdir('.') if f.endswith('.npz')]
    npz_files.sort()

    if not npz_files:
        print("ERRORE: Nessun file .npz trovato nella cartella corrente.")
        return

    print("Quale set di feature vuoi analizzare?")
    for i, file in enumerate(npz_files, start=1):
        print(f"{i}. {file}")

    while True:
        try:
            scelta_idx = int(input("\nScelta (numero): "))
            if 1 <= scelta_idx <= len(npz_files):
                file_dati = npz_files[scelta_idx - 1]
                break
            else:
                print("Scelta non valida.")
        except ValueError:
            print("Inserisci un numero intero.")

    # 2. Caricamento dati grezzi e Normalizzazione
    print(f"\nCaricamento dati da {file_dati}...")
    try:
        data = np.load(file_dati)
        names = data["names"]
        embeddings_raw = data["embeddings"]

        # NORMALIZZAZIONE L2
        embeddings_norm = normalize(embeddings_raw, norm='l2')
        
        print(f"[✓] Trovate {len(names)} immagini.")
    except Exception as e:
        print(f"Errore durante il caricamento del file: {e}")
        return

    # 3. Analisi Esplorativa: Silhouette Score
    print("\nAnalisi del dataset in corso per suggerire il numero di cluster (potrebbe richiedere qualche secondo)...")
    best_k = 2
    best_score = -1
    max_test_k = min(20, len(names) - 1) # Evita errori se ci sono pochissime immagini

    for test_k in range(2, max_test_k + 1):
        temp_clusterer = AgglomerativeClustering(n_clusters=test_k, metric='euclidean', linkage='ward')
        temp_labels = temp_clusterer.fit_predict(embeddings_norm)
        score = silhouette_score(embeddings_norm, temp_labels, metric='euclidean')
        
        if score > best_score:
            best_score = score
            best_k = test_k

    print(f"-> Il computer suggerisce: {best_k} cluster (Silhouette Score: {best_score:.3f})")

    # 4. Analisi Esplorativa: Dendrogramma
    mostra_dendro = input("\nVuoi visualizzare il dendrogramma per decidere visivamente? (s/n): ").strip().lower()
    if mostra_dendro == 's':
        print("Calcolo dell'albero gerarchico...")
        Z = linkage(embeddings_norm, method='ward', metric='euclidean')
        plt.figure(figsize=(12, 6))
        plt.title("Dendrogramma del Dataset Orchidee")
        plt.xlabel("Dimensione dei cluster (numero di immagini nel ramo)")
        plt.ylabel("Distanza di unione")
        dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12.)
        print("\n[!] ATTENZIONE: Chiudi la finestra del grafico per continuare l'esecuzione dello script.")
        plt.show()

    # 5. Setup e avvio del Clustering definitivo
    while True:
        try:
            k = int(input("\nQuanti gruppi/sub-cluster vuoi formare definitivamente? (es. 4, 5, 15): "))
            if k > 1:
                break
            print("Il numero di cluster deve essere maggiore di 1.")
        except ValueError:
            print("Inserisci un numero intero valido.")

    print("\nEsecuzione Agglomerative Clustering definitivo in corso...")
    clusterer = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    labels = clusterer.fit_predict(embeddings_norm)

    # 6. Salvataggio su Excel
    nome_base = os.path.splitext(file_dati)[0]
    cartella_output = f"Risultati_Agglomerative_{nome_base}"
    os.makedirs(cartella_output, exist_ok=True)
    
    file_excel = os.path.join(cartella_output, f"clustering_k{k}.xlsx")
    df = pd.DataFrame({'image name': names, 'ID_Cluster': labels})
    df.to_excel(file_excel, index=False)
    print(f"\n[✓] Clustering completato! Dati salvati in: {file_excel}")

    # 7. Organizzazione Automatica delle immagini
    if CARTELLA_IMMAGINI_ORIGINALI and CARTELLA_IMMAGINI_ORIGINALI != "inserisci_qui_il_percorso":
        if not os.path.exists(CARTELLA_IMMAGINI_ORIGINALI):
            print(f"\nERRORE: La cartella impostata '{CARTELLA_IMMAGINI_ORIGINALI}' non esiste. Verifica il percorso.")
        else:
            print(f"\nCopia delle immagini da '{CARTELLA_IMMAGINI_ORIGINALI}' in corso...")
            foto_copiate = 0
            foto_mancanti = 0
            
            for nome_img, cluster_id in zip(names, labels):
                percorso_sorgente = os.path.join(CARTELLA_IMMAGINI_ORIGINALI, nome_img)
                cartella_destinazione = os.path.join(cartella_output, f"Cluster_{cluster_id}")
                os.makedirs(cartella_destinazione, exist_ok=True)
                
                percorso_destinazione = os.path.join(cartella_destinazione, nome_img)
                
                if os.path.exists(percorso_sorgente):
                    shutil.copy2(percorso_sorgente, percorso_destinazione)
                    foto_copiate += 1
                else:
                    foto_mancanti += 1
            
            print(f"[✓] {foto_copiate} immagini organizzate nei cluster.")
            if foto_mancanti > 0:
                print(f"[!] {foto_mancanti} immagini non trovate nella cartella sorgente.")
    else:
        print("\n[!] Variabile CARTELLA_IMMAGINI_ORIGINALI non configurata. Salto la copia delle immagini.")

if __name__ == "__main__":
    main()