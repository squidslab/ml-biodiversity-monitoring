import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# 1. CONFIGURAZIONI INIZIALI E FILE
# ==========================================
NUMERO_CLUSTER_TEST = 6          # Sappiamo che il test set ha 6 categorie
RANGE_VICINI = range(5, 16)      # Range per cercare i "ponti" ottimali
RANGE_CLUSTER_TED = range(2, 11) # Range per cercare il numero naturale di cluster sul TED

NPZ_FILE_TED = 'features_resnet_custom_cropped_ALL_DATA.npz'
EXCEL_FILE_TED = 'dataset.xlsx'

NPZ_FILE_TEST = 'features_resnet_custom_base_TEST.npz'
EXCEL_FILE_TEST = 'Registro_Dataset_ResNet18.xlsx'

print("=== AVVIO PIPELINE SPECTRAL CLUSTERING (SOLO 'CURATED') ===\n")

# ==========================================
# 2. CARICAMENTO DATI E NORMALIZZAZIONE
# ==========================================
try:
    data_ted = np.load(NPZ_FILE_TED)
    data_test = np.load(NPZ_FILE_TEST)
except FileNotFoundError as e:
    print(f"ERRORE: Impossibile trovare i file .npz ({e}).")
    exit()

emb_all = np.vstack((data_ted['embeddings'], data_test['embeddings']))
names_all = np.concatenate((data_ted['names'], data_test['names']))

# Normalizzazione L2 (Fondamentale a 512 dimensioni)
embeddings_norm = normalize(emb_all, norm='l2')

# ==========================================
# 3. MERGE METADATI (Aggiunta di personalAnnotation)
# ==========================================
df_global = pd.DataFrame({'image name': names_all})
nomi_test_set = set(data_test['names'])
df_global['is_test_set'] = df_global['image name'].apply(lambda x: x in nomi_test_set)

try:
    # Preparazione dataset TED
    df_excel_ted = pd.read_excel(EXCEL_FILE_TED)
    df_excel_ted['Specie Predetta'] = df_excel_ted['Specie Predetta'].fillna('Non definita')
    
    # Gestione colonna personalAnnotation
    if 'personalAnnotation' not in df_excel_ted.columns:
        df_excel_ted['personalAnnotation'] = 'Vuoto'
    df_ted_clean = df_excel_ted[['image name', 'Specie Predetta', 'personalAnnotation']]

    # Preparazione TEST SET
    df_excel_test = pd.read_excel(EXCEL_FILE_TEST)
    df_excel_test = df_excel_test.rename(columns={'Classe': 'Specie Predetta', 'Nome File': 'image name'})
    df_excel_test['Specie Predetta'] = 'test_' + df_excel_test['Specie Predetta'].astype(str)
    df_excel_test['personalAnnotation'] = 'Vuoto'
    df_test_clean = df_excel_test[['image name', 'Specie Predetta', 'personalAnnotation']]

    # Unione finale
    df_excel_all = pd.concat([df_ted_clean, df_test_clean], ignore_index=True)
    df_global = df_global.merge(df_excel_all, on='image name', how='left')
    
    # Pulizia NaNs
    df_global['Specie Predetta'] = df_global['Specie Predetta'].fillna('Sconosciuta')
    df_global['personalAnnotation'] = df_global['personalAnnotation'].fillna('Vuoto')

except FileNotFoundError as e:
    print(f"ATTENZIONE: File Excel non trovato ({e}).")
    df_global['Specie Predetta'] = 'Sconosciuta'
    df_global['personalAnnotation'] = 'Vuoto'


# ==========================================
# FASE 1: CALIBRAZIONE "N_NEIGHBORS" SUL TEST SET
# ==========================================
maschera_test = df_global['is_test_set']
X_test = embeddings_norm[maschera_test]
true_labels_test = df_global.loc[maschera_test, 'Specie Predetta'].values

print(f"--- FASE 1: Ricerca dei 'vicini' ottimali (Test Set: {len(X_test)} immagini) ---")

risultati_gs_vicini = []

for vicini in RANGE_VICINI:
    # Calcoliamo la mappa dei vicini usando la distanza Coseno
    grafo_test = kneighbors_graph(X_test, n_neighbors=vicini, metric='cosine', mode='connectivity', include_self=True)
    
    spectral_test = SpectralClustering(
        n_clusters=NUMERO_CLUSTER_TEST, 
        affinity='precomputed', 
        assign_labels='cluster_qr', 
        random_state=42
    )
    labels_pred = spectral_test.fit_predict(grafo_test)
    
    ami = adjusted_mutual_info_score(true_labels_test, labels_pred)
    ari = adjusted_rand_score(true_labels_test, labels_pred)
    risultati_gs_vicini.append({'n_neighbors': vicini, 'AMI': ami, 'ARI': ari})

df_risultati_vicini = pd.DataFrame(risultati_gs_vicini).sort_values(by=['AMI', 'ARI'], ascending=[False, False])
miglior_vicini = int(df_risultati_vicini.iloc[0]['n_neighbors'])

print(f"=> Parametro ottimale trovato: n_neighbors = {miglior_vicini} (AMI: {df_risultati_vicini.iloc[0]['AMI']:.4f})")

# ==========================================
# FASE 2: RICERCA "CIECA" DEI CLUSTER SUL TED SET (SOLO CURATED)
# ==========================================
# Filtro combinato: NON test_set E personalAnnotation == 'curated'
maschera_ted = (~df_global['is_test_set']) & (df_global['personalAnnotation'].astype(str).str.lower() == 'curated')

X_ted = embeddings_norm[maschera_ted]
df_ted = df_global[maschera_ted].copy()

print(f"\n--- FASE 2: Ricerca numero cluster naturali (TED Set Filtrato: {len(X_ted)} immagini) ---")

if len(X_ted) == 0:
    print("ERRORE: Nessuna immagine trovata con personalAnnotation=='curated'. Verifica il file Excel!")
    exit()

risultati_silhouette = []

# Calcoliamo il grafo Coseno per il TED set filtrato una volta sola
grafo_ted = kneighbors_graph(X_ted, n_neighbors=miglior_vicini, metric='cosine', mode='connectivity', include_self=True)

for k in RANGE_CLUSTER_TED:
    spectral_cieco = SpectralClustering(
        n_clusters=k, 
        affinity='precomputed', 
        assign_labels='cluster_qr', 
        random_state=42
    )
    labels_temp = spectral_cieco.fit_predict(grafo_ted)
    
    # Calcoliamo il punteggio usando esplicitamente la metrica 'cosine'
    score = silhouette_score(X_ted, labels_temp, metric='cosine')
    risultati_silhouette.append({'k_clusters': k, 'Silhouette_Score': score})

df_silhouette = pd.DataFrame(risultati_silhouette).sort_values(by='Silhouette_Score', ascending=False)
miglior_k = int(df_silhouette.iloc[0]['k_clusters'])

print(f"=> Numero ideale di cluster trovato: {miglior_k} (Silhouette Score: {df_silhouette.iloc[0]['Silhouette_Score']:.4f})")

# ==========================================
# FASE 3: ESECUZIONE FINALE SUL DATASET TED (SOLO CURATED)
# ==========================================
print(f"\n--- FASE 3: Esecuzione clustering finale ---")

spectral_finale = SpectralClustering(
    n_clusters=miglior_k, 
    affinity='precomputed', 
    assign_labels='cluster_qr', 
    random_state=42
)
df_ted['Cluster'] = spectral_finale.fit_predict(grafo_ted) 

# Salvataggio nel dataframe globale: chi non è curated resta a -1
if 'Cluster_Predetto' not in df_global.columns:
    df_global['Cluster_Predetto'] = -1 
df_global.loc[maschera_ted, 'Cluster_Predetto'] = df_ted['Cluster']


# ==========================================
# FASE 4: STAMPA DISTRIBUZIONE E MATRICE DI COMPOSIZIONE
# ==========================================
print("\n" + "="*65)
print(f"RISULTATI FINALI DATASET TED CURATED (Parametri: {miglior_k} Cluster, {miglior_vicini} Vicini)")
print("="*65)

print("\nTabella di Distribuzione (Numerosità per Cluster):")
print("-" * 45)
print(df_ted['Cluster'].value_counts().sort_index().to_frame(name='Numero Immagini').rename_axis('Cluster ID'))

print("\nMATRICE DI COMPOSIZIONE (Crosstab) - Specie Reali vs Cluster Predetti:")
print("-" * 65)
ct_finale = pd.crosstab(df_ted['Specie Predetta'], df_ted['Cluster'])
print(ct_finale)

print("\nCompletato con successo! 🎉")