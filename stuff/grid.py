import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import dash_bootstrap_components as dbc
import itertools

# ==========================================
# 1. IL TUO CODICE DI PREPARAZIONE DATI
# ==========================================
NPZ_FILE_TED = 'features_resnet_custom_cropped_ALL_DATA.npz'
EXCEL_FILE_TED = 'dataset.xlsx'

NPZ_FILE_TEST = 'features_resnet_custom_base_TEST.npz'
EXCEL_FILE_TEST = 'Registro_Dataset_ResNet18.xlsx'

data_ted = np.load(NPZ_FILE_TED)
data_test_set = np.load(NPZ_FILE_TEST)
emb_all = np.vstack((data_ted['embeddings'], data_test_set['embeddings']))
names_all = np.concatenate((data_ted['names'], data_test_set['names']))

embeddings_norm = normalize(emb_all, norm='l2')
coords_3d = PCA(n_components=3).fit_transform(embeddings_norm)

df_global = pd.DataFrame(coords_3d, columns=['x', 'y', 'z'])
df_global['image name'] = names_all

nomi_test = set(data_test_set['names'])
df_global['is_test_set'] = df_global['image name'].apply(lambda x: x in nomi_test)

try:
    df_excel_ted = pd.read_excel(EXCEL_FILE_TED)
    df_excel_ted['datasetCategory'] = df_excel_ted['datasetCategory'].fillna('Vuoto')
    df_excel_ted['personalAnnotation'] = df_excel_ted['personalAnnotation'].fillna('Vuoto')
    df_ted_clean = df_excel_ted[['image name', 'datasetCategory', 'personalAnnotation', 'Specie Predetta']]

    df_excel_test = pd.read_excel(EXCEL_FILE_TEST)
    df_excel_test = df_excel_test.rename(columns={'Classe': 'Specie Predetta', 'Nome File': 'image name'})
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
    # Per evitare errori se l'excel non c'è
    df_global['personalAnnotation'] = 'Vuoto' 

# ==========================================
# 2. ISOLAMENTO DEL DATASET ETICHETTATO
# ==========================================
maschera_etichettato = df_global['is_test_set']

X_etichettato = embeddings_norm[maschera_etichettato] 
true_labels = df_global.loc[maschera_etichettato, 'Specie Predetta'].values

# ==========================================
# 3. ESECUZIONE DELLA GRID SEARCH
# ==========================================
eps_values = np.round(np.arange(0.05, 0.51, 0.01), 2)
min_samples_values = range(2, 31)

grid_search_results = []

print(f"Inizio Grid Search su {len(X_etichettato)} campioni...")

for eps, min_samples in itertools.product(eps_values, min_samples_values):
    
    # Inizializzazione ed esecuzione di DBSCAN solo sul set etichettato
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_etichettato)
    
    # Calcolo Metriche
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    # Se DBSCAN mette tutto nel rumore o crea un solo cluster, AMI/ARI non hanno senso calcolarli
    if n_clusters < 1:
        continue
        
    noise_pct = (list(cluster_labels).count(-1) / len(cluster_labels)) * 100
    ami = adjusted_mutual_info_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    # Creazione DataFrame temporaneo per la crosstab
    df_plot = pd.DataFrame({
        'Specie Predetta': true_labels, # La tua Ground Truth
        'Cluster': cluster_labels
    })
    
    # Generazione Crosstab
    ct = pd.crosstab(df_plot['Specie Predetta'], df_plot['Cluster'])
    
    # Creazione tabella Dash Bootstrap
    if not ct.empty:
        dash_table = dbc.Table.from_dataframe(
            ct.reset_index(), 
            striped=True, 
            bordered=True, 
            hover=True, 
            responsive=True,
            className="mb-0"
        )
    else:
        dash_table = None
    
    # Salvataggio nel dizionario
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

# ==========================================
# 4. ANALISI DEI RISULTATI
# ==========================================
results_df = pd.DataFrame(grid_search_results)

if not results_df.empty:
    # Filtriamo configurazioni con troppo rumore (es. > 40%) prima di ordinare
    valid_results = results_df[results_df['noise_pct'] < 40]

    # Ordiniamo in base all'Adjusted Mutual Information
    best_params_df = valid_results.sort_values(by=['AMI', 'ARI'], ascending=[False, False])

    print("\n--- MIGLIORI 10 CONFIGURAZIONI TROVATE ---")
    print(best_params_df[['eps', 'min_samples', 'n_clusters', 'noise_pct', 'AMI', 'ARI']].head(10))

    # Esempio di come estrarre i parametri migliori e la rispettiva tabella per Dash
    if not best_params_df.empty:
        miglior_eps = best_params_df.iloc[0]['eps']
        miglior_min_samples = best_params_df.iloc[0]['min_samples']
        migliore_tabella_dash = best_params_df.iloc[0]['dash_table']
        
        print(f"\nI parametri ideali scelti dal Test Set sono eps={miglior_eps} e min_samples={miglior_min_samples}")
else:
    print("Errore nella Grid Search: nessuna configurazione salvata.")


# ==========================================
# 5. APPLICAZIONE DEI PARAMETRI AL DATASET TED (SOLO CURATED)
# ==========================================
# 1. Isoliamo il dataset NON etichettato (TED) aggiungendo il filtro 'curated'
maschera_ted_curated = (~df_global['is_test_set']) & (df_global['personalAnnotation'].astype(str).str.lower() == 'curated')

# Assicurati di usare lo stesso tipo di dato usato nella grid search (es. embeddings_norm)
X_ted_curated = embeddings_norm[maschera_ted_curated]
df_ted_curated = df_global[maschera_ted_curated].copy()

if len(X_ted_curated) == 0:
    print("\nERRORE: Nessuna immagine trovata con personalAnnotation=='curated'. Verifica il file Excel!")
    exit()

print(f"\nApplicazione di DBSCAN su {len(X_ted_curated)} campioni filtrati (TED 'Curated')...")

# 2. Inizializziamo DBSCAN (qui stai usando i valori hard-coded 0.36 e 12 che avevi impostato,
# per renderlo automatico potresti sostituirli con `eps=miglior_eps` e `min_samples=miglior_min_samples`)
dbscan_finale = DBSCAN(eps=0.36, min_samples=12)

# 3. Predizione sui dati senza etichetta
cluster_labels_ted = dbscan_finale.fit_predict(X_ted_curated)
df_ted_curated['Cluster'] = cluster_labels_ted

# 4. Salviamo i risultati nel dataframe globale per usarli nella Dashboard
# Le immagini scartate o non 'curated' manterranno il valore -1
if 'Cluster_Predetto' not in df_global.columns:
    df_global['Cluster_Predetto'] = -1 
    
df_global.loc[maschera_ted_curated, 'Cluster_Predetto'] = cluster_labels_ted

# Vediamo quanti cluster ha trovato sul nuovo dataset e quanto rumore c'è
n_cluster_ted = len(set(cluster_labels_ted)) - (1 if -1 in cluster_labels_ted else 0)
noise_ted = list(cluster_labels_ted).count(-1) / len(cluster_labels_ted) * 100

print(f"Sul dataset TED Curated sono stati trovati {n_cluster_ted} cluster.")
print(f"Percentuale di rumore sul dataset TED Curated: {noise_ted:.2f}%")

print("\nMATRICE DI COMPOSIZIONE (Crosstab) - DATASET TED CURATED:")
print("-" * 65)
print(pd.crosstab(df_ted_curated['Specie Predetta'], df_ted_curated['Cluster']))