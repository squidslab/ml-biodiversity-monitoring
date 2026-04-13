import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --- CONFIGURAZIONE ---
IMAGE_FOLDER = "../dataset" 
EXCEL_FILE = "my_dataset.xlsx"  

# ------------------------
# METODO DEL GOMITO
# ------------------------
def esegui_metodo_gomito(embeddings_pca):
    print("\n[1] Calcolo del Metodo del Gomito...")

    nn = NearestNeighbors(n_neighbors=4)
    nn.fit(embeddings_pca)
    distanze, _ = nn.kneighbors(embeddings_pca)

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
def esegui_dbscan(embeddings_pca, names, file_output):
    print("\n[2] Esecuzione Clustering DBSCAN...")

    try:
        eps_val = float(input("Inserisci il valore EPS desiderato: "))
        dbscan = DBSCAN(eps=eps_val, min_samples=4)
        labels = dbscan.fit_predict(embeddings_pca)

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
def genera_grafico(embeddings_scaled, labels, nome_grafico):
    if labels is None:
        print("(!) Errore: Esegui prima il clustering (opzione 2).")
        return

    print("\n[3] Generazione grafico PCA 2D...")
    pca_2d = PCA(n_components=2)
    embeddings_2d = pca_2d.fit_transform(embeddings_scaled)

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

    plt.title("Clustering Orchidee (DBSCAN + PCA)")
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
    
    file_resnet = f"features_resnet18_imagenet_{tag_gruppo}.npz"
    file_vit = f"features_vit_{tag_gruppo}.npz"
    
    print("Quale set di feature vuoi analizzare?")
    print(f"1. ResNet18 ({file_resnet})")
    print(f"2. ViT-B/16 ({file_vit})")
    print("3. Inserisci nome file manualmente")
    scelta_dati = input("Scelta: ")

    modello_tag = "custom"
    if scelta_dati == "1":
        features_dati = file_resnet
        modello_tag = "resnet18_imagenet"
    elif scelta_dati == "2":
        features_dati = file_vit
        modello_tag = "vit"
    else:
        features_dati = input("Scrivi il nome esatto del file .npz: ")
        modello_tag = input("Nome del modello per i file di output: ").strip().replace(" ", "_")

    OUTPUT_ROOT = f"Clusters_Risultanti_{modello_tag}_{tag_gruppo}"
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    FILE_OUTPUT = os.path.join(OUTPUT_ROOT, f"clustering_dbscan_{modello_tag}_{tag_gruppo}.xlsx")
    NOME_GRAFICO = os.path.join(OUTPUT_ROOT, f"grafico_dbscan_{modello_tag}_{tag_gruppo}.png")

    try:
        print(f"Caricamento dati da {features_dati}...")
        data = np.load(features_dati)
        names = data["names"]
        embeddings_scaled = StandardScaler().fit_transform(data["embeddings"])
        embeddings_pca = PCA(n_components=50, random_state=42).fit_transform(embeddings_scaled)
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
            esegui_metodo_gomito(embeddings_pca)
        elif scelta == "2":
            labels, df_final = esegui_dbscan(embeddings_pca, names, FILE_OUTPUT)
        elif scelta == "3":
            genera_grafico(embeddings_scaled, labels, NOME_GRAFICO)
        elif scelta == "4":
            organizza_cartelle(df_final, OUTPUT_ROOT)
        elif scelta == "0":
            print("Uscita dal modulo DBSCAN...")
            break
        else:
            print("Scelta non valida.")

if __name__ == "__main__":
    main()