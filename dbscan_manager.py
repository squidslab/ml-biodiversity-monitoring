import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURAZIONE ---
IMAGE_FOLDER = "../dataset_new_cropper" 
EXCEL_FILE = "my_dataset.xlsx"  

# ------------------------
# METODO DEL GOMITO
# ------------------------
def esegui_metodo_gomito(embeddings_umap):
    print("\n[1] Calcolo del Metodo del Gomito...")

    nn = NearestNeighbors(n_neighbors=4)
    nn.fit(embeddings_umap)
    distanze, _ = nn.kneighbors(embeddings_umap)

    distanze = np.sort(distanze[:, 3], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(distanze)
    plt.title("Metodo del 'Gomito' per trovare l'eps perfetto")
    plt.xlabel("Punti ordinati")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.show()

# ------------------------
# ESECUZIONE DBSCAN
# ------------------------
def esegui_dbscan(embeddings_norm, names, file_output):
    print("\n[2] Esecuzione Clustering DBSCAN...")

    try:
        eps_val = float(input("Inserisci il valore EPS desiderato: "))
        dbscan = DBSCAN(eps=eps_val, min_samples=15)
        labels = dbscan.fit_predict(embeddings_norm)

        df_dbscan = pd.DataFrame({"image name": names, "ID_Cluster": labels})

        # MERGE E SALVATAGGIO
        print("Integrazione di tutte le informazioni dal dataset originale...")
        df_excel = pd.read_excel(EXCEL_FILE)
        
        df_dbscan.columns = df_dbscan.columns.str.strip()
        df_excel.columns = df_excel.columns.str.strip()

        df_final = pd.merge(
            df_dbscan, 
            df_excel, 
            on="image name", 
            how="left"
        )

        df_final.sort_values(by="ID_Cluster", inplace=True)

        df_final.to_excel(file_output, index=False)
        print(f"\nOperazione completata! Trovati {len(set(labels)) - (1 if -1 in labels else 0)} cluster.")
        print(f"File salvato in: {file_output}")

        return labels, df_final
    except Exception as e:
        print(f"Errore durante il clustering o il merge: {e}")
        return None, None

# ------------------------
# VISUALIZZAZIONE GRAFICO 
# ------------------------
def genera_grafico(embeddings_norm, labels, nome_grafico):
    if labels is None:
        print("(!) Errore: Esegui prima il clustering (opzione 2).")
        return

    print("\n[3] Generazione grafico 2D...")
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings_norm)

    plt.figure(figsize=(10, 8))
    
    noise_mask = (labels == -1)
    plt.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1], 
                c='lightgray', marker='x', label='Rumore (Outlier)', alpha=0.5, s=30)

    core_mask = (labels != -1)
    if any(core_mask):
        scatter = plt.scatter(embeddings_2d[core_mask, 0], embeddings_2d[core_mask, 1], 
                              c=labels[core_mask], cmap='tab10', alpha=0.9, s=60, edgecolors='k')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label("ID Cluster")
    else:
        print("Attenzione: Nessun cluster trovato con i parametri attuali (solo rumore).")

    plt.title("Clustering Orchidee (DBSCAN)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {nome_grafico}")

    plt.show()

# ------------------------
# ORGANIZZAZIONE FISICA DELLE IMMAGINI
# ------------------------
def organizza_cartelle(df_final, output_root):
    if df_final is None:
        print("(!) Errore: Dati non disponibili. Esegui prima il clustering (opzione 2).")
        return
    
    print(f"\nInizio organizzazione immagini in {output_root}...")
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    cluster_ids = sorted(df_final["ID_Cluster"].unique())

    for cid in cluster_ids:
        folder_name = f"Cluster_{cid}" if cid != -1 else "RUMORE"
        target_dir = os.path.join(output_root, folder_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        files_in_cluster = df_final[df_final["ID_Cluster"] == cid]["image name"].tolist()
        
        for f in files_in_cluster:
            src = os.path.join(IMAGE_FOLDER, f)
            dst = os.path.join(target_dir, f)
            
            if os.path.exists(src):
                shutil.copy2(src, dst) 
            else:
                print(f"   [!] File non trovato: {f}")

    print("\nOrganizzazione completata con successo!")

# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------
def main(tag_gruppo="ALL_DATA"):
    print(f"\n--- INIZIALIZZAZIONE DBSCAN (Sessione: {tag_gruppo}) ---")
    
    npz_files = [f for f in os.listdir('.') if f.endswith('.npz')]
    npz_files.sort()
    
    print("Quale set di feature vuoi analizzare?")
    if not npz_files:
        print("Nessun file .npz trovato nella cartella corrente.")
        
    for i, file in enumerate(npz_files, start=1):
        print(f"{i}. {file}")
    
    opzione_manuale = len(npz_files) + 1
    print(f"{opzione_manuale}. Inserisci path o nome file manualmente")
    
    while True:
        scelta_dati = input("\nScelta: ")
        try:
            scelta_idx = int(scelta_dati)
            if 1 <= scelta_idx <= opzione_manuale:
                break
            else:
                print(f"Scelta non valida. Inserisci un numero tra 1 e {opzione_manuale}.")
        except ValueError:
            print("Input non valido. Inserisci un numero intero.")

    if scelta_idx == opzione_manuale:
        features_dati = input("Scrivi il percorso esatto o il nome del file .npz: ")
        modello_tag = input("Nome del modello per i file di output: ").strip().replace(" ", "_")
    else:
        features_dati = npz_files[scelta_idx - 1]
        nome_base = os.path.splitext(features_dati)[0]
        modello_tag = nome_base.replace("features_", "")

    OUTPUT_ROOT = f"Clusters_Risultanti_{modello_tag}"
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    FILE_OUTPUT = os.path.join(OUTPUT_ROOT, f"clustering_dbscan_{modello_tag}_{tag_gruppo}.xlsx")
    NOME_GRAFICO = os.path.join(OUTPUT_ROOT, f"grafico_dbscan_{modello_tag}_{tag_gruppo}.png")

    try:
        print(f"Caricamento dati da {features_dati}...")
        data = np.load(features_dati)
        names = data["names"]
        embeddings_norm = normalize(data["embeddings"], norm='l2')
        #pca = PCA(n_components=50) 
        #embeddings_norm = pca.fit_transform(embeddings_norm)
        
    except FileNotFoundError:
        print(f"ERRORE: File '{features_dati}' non trovato. Hai effettuato l'estrazione per questo gruppo?")
        return

    labels = None
    df_final = None

    while True:
        print("\n--- MENU DBSCAN ---")
        print("1. Metodo del Gomito (Trova EPS)")
        print("2. Esegui DBSCAN & Merge Excel")
        print("3. Visualizza & Salva Grafico")
        print("4. Organizza Foto in Cartelle")
        print("0. Torna al menu principale")
        
        scelta = input("\nCosa vuoi fare? ")

        if scelta == "1":
            esegui_metodo_gomito(embeddings_norm)
        elif scelta == "2":
            labels, df_final = esegui_dbscan(embeddings_norm, names, FILE_OUTPUT)
        elif scelta == "3":
            genera_grafico(embeddings_norm, labels, NOME_GRAFICO)
        elif scelta == "4":
            organizza_cartelle(df_final, OUTPUT_ROOT)
        elif scelta == "0":
            print("Uscita dal modulo DBSCAN...")
            break
        else:
            print("Scelta non valida.")

if __name__ == "__main__":
    main()