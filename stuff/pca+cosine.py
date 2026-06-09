import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc
import itertools

NPZ_FILE_TED = 'features_resnet_custom_cropped_ALL_DATA.npz'
EXCEL_FILE_TED = 'dataset.xlsx'

NPZ_FILE_TEST = 'features_resnet_custom_base_TEST.npz'
EXCEL_FILE_TEST = 'Registro_Dataset_ResNet18.xlsx'

# --- PREPARAZIONE DATI --- 
data_ted = np.load(NPZ_FILE_TED)
data_test_set = np.load(NPZ_FILE_TEST)
emb_all = np.vstack((data_ted['embeddings'], data_test_set['embeddings']))
names_all = np.concatenate((data_ted['names'], data_test_set['names']))

embeddings_norm = normalize(emb_all, norm='l2')

# 1. PCA per la visualizzazione 3D
coords_3d = PCA(n_components=3).fit_transform(embeddings_norm)
df_global = pd.DataFrame(coords_3d, columns=['x', 'y', 'z'])
df_global['image name'] = names_all

# ---> LA RIGA CHE MANCAVA: Creazione della colonna 'is_test_set' <---
nomi_test = set(data_test_set['names'])
df_global['is_test_set'] = df_global['image name'].apply(lambda x: x in nomi_test)

# 2. PCA per il clustering (50 dimensioni)
pca_clustering = PCA(n_components=50)
embeddings_ridotti = pca_clustering.fit_transform(embeddings_norm)

ORDINE_CATEGORIE = ['Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others', 'Test Set']

# Merge con Excel
try:
    df_excel_ted = pd.read_excel(EXCEL_FILE_TED)
    df_excel_ted['datasetCategory'] = df_excel_ted['datasetCategory'].fillna('Vuoto')
    df_excel_ted['personalAnnotation'] = df_excel_ted['personalAnnotation'].fillna('Vuoto')
    df_ted_clean = df_excel_ted[['image name', 'datasetCategory', 'personalAnnotation', 'Specie Predetta']]

    df_excel_test = pd.read_excel(EXCEL_FILE_TEST)
    df_excel_test = df_excel_test.rename(columns={
        'Classe': 'Specie Predetta',
        'Nome File': 'image name' 
    })
    df_excel_test['Specie Predetta'] = 'test_' + df_excel_test['Specie Predetta'].astype(str)
    df_excel_test['datasetCategory'] = 'Vuoto'
    df_excel_test['personalAnnotation'] = 'Vuoto'
    df_test_clean = df_excel_test[['image name', 'datasetCategory', 'personalAnnotation', 'Specie Predetta']]

    df_excel_all = pd.concat([df_ted_clean, df_test_clean], ignore_index=True)

    df_global = df_global.merge(df_excel_all, on='image name', how='left')
    df_global['Specie Predetta'] = df_global['Specie Predetta'].fillna('Non definita')

except FileNotFoundError as e:
    print(f"ATTENZIONE: File non trovato ({e}). Verranno usate etichette fittizie.")
    df_global['Specie Predetta'] = 'Non definita'
    df_global['personalAnnotation'] = 'Vuoto'


# ==========================================
# 2. ISOLAMENTO DEL DATASET ETICHETTATO (TEST SET)
# ==========================================
maschera_etichettato = df_global['is_test_set']

# Usiamo gli embedding ridotti a 50 dimensioni
X_etichettato = embeddings_ridotti[maschera_etichettato] 
true_labels = df_global.loc[maschera_etichettato, 'Specie Predetta'].values

# ==========================================
# 3. ESECUZIONE DELLA GRID SEARCH
# ==========================================
eps_values = np.round(np.arange(0.05, 0.51, 0.01), 2)
min_samples_values = range(2, 31)

grid_search_results = []

print(f"Inizio Grid Search su {len(X_etichettato)} campioni etichettati (PCA 50D + Cosine)...")

for eps, min_samples in itertools.product(eps_values, min_samples_values):
    
    # DBSCAN con metrica Coseno
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(X_etichettato)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    if n_clusters < 1:
        continue
        
    noise_pct = (list(cluster_labels).count(-1) / len(cluster_labels)) * 100
    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    df_plot = pd.DataFrame({'Specie Predetta': true_labels, 'Cluster': cluster_labels})
    ct = pd.crosstab(df_plot['Specie Predetta'], df_plot['Cluster'])
    
    # Prevenzione errore: if ct è vuoto
    if ct.empty:
        dash_table = None
    else:
        dash_table = dbc.Table.from_dataframe(
            ct.reset_index(), striped=True, bordered=True, hover=True, responsive=True, className="mb-0"
        )
    
    grid_search_results.append({
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters,
        'noise_pct': noise_pct,
        'AMI': ami,
        'ARI': ari,
        'crosstab_df': ct,
        'dash_table': dash_table
    })

# Stampa i risultati
results_df = pd.DataFrame(grid_search_results)
if not results_df.empty:
    valid_results = results_df[results_df['noise_pct'] < 40]
    if not valid_results.empty:
        best_params_df = valid_results.sort_values(by=['AMI', 'ARI'], ascending=[False, False])
        print("\n--- MIGLIORI 10 CONFIGURAZIONI TROVATE ---")
        print(best_params_df[['eps', 'min_samples', 'n_clusters', 'noise_pct', 'AMI', 'ARI']].head(10))
    else:
        print("\nNessuna configurazione valida trovata (tutte superano il 40% di rumore).")
else:
    print("\nErrore nella Grid Search.")


# ==========================================
# 4. TEST FINALE SUL DATASET TED (SOLO CURATED)
# ==========================================
# Isoliamo il dataset TED usando la maschera combinata per "Curated"
maschera_ted_curated = (~df_global['is_test_set']) & (df_global['personalAnnotation'].astype(str).str.lower() == 'curated')

# IMPORTANTE: Usiamo gli stessi embedding ridotti a 50 dimensioni!
X_ted_curated = embeddings_ridotti[maschera_ted_curated]
df_ted_curated = df_global[maschera_ted_curated].copy()

if len(X_ted_curated) == 0:
    print("\nERRORE: Nessuna immagine trovata con personalAnnotation=='curated'. Verifica il file Excel!")
    exit()

print(f"\nApplicazione di DBSCAN su {len(X_ted_curated)} campioni filtrati (TED 'Curated')...")

# Scegliamo i parametri della riga 298 (che ha un ottimo bilanciamento di ARI e rumore)
# Ricorda di mantenere metric='cosine'
dbscan_finale = DBSCAN(eps=0.17, min_samples=20, metric='cosine')

cluster_labels_ted = dbscan_finale.fit_predict(X_ted_curated)
df_ted_curated['Cluster'] = cluster_labels_ted

n_cluster_ted = len(set(cluster_labels_ted)) - (1 if -1 in cluster_labels_ted else 0)
noise_ted = list(cluster_labels_ted).count(-1) / len(cluster_labels_ted) * 100

print(f"Sul dataset TED Curated sono stati trovati {n_cluster_ted} cluster.")
print(f"Percentuale di rumore sul dataset TED Curated: {noise_ted:.2f}%")

print("\nMATRICE DI COMPOSIZIONE (Crosstab) - DATASET TED CURATED:")
print("-" * 65)
print(pd.crosstab(df_ted_curated['Specie Predetta'], df_ted_curated['Cluster']))