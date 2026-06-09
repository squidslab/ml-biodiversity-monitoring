import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# ==========================================
# 1. CONFIGURAZIONI E PARAMETRI
# ==========================================
NUMERO_CLUSTER = 4
NUMERO_VICINI = 18

NPZ_FILE_TED = 'features_resnet_custom_cropped_ALL_DATA.npz'
EXCEL_FILE_TED = 'dataset.xlsx'

NPZ_FILE_TEST = 'features_resnet_custom_base_TEST.npz'
EXCEL_FILE_TEST = 'Registro_Dataset_ResNet18.xlsx'

print(f"--- AVVIO SPECTRAL CLUSTERING ({NUMERO_CLUSTER} Cluster, {NUMERO_VICINI} Vicini) ---\n")

# ==========================================
# 2. CARICAMENTO DATI (.npz) E NOMI
# ==========================================
try:
    data_ted = np.load(NPZ_FILE_TED)
    data_test = np.load(NPZ_FILE_TEST)
except FileNotFoundError as e:
    print(f"ERRORE: Impossibile trovare i file .npz ({e}).")
    exit()

emb_ted = data_ted['embeddings']
names_ted = data_ted['names']

emb_test = data_test['embeddings']
names_test = data_test['names']

# Unione array per normalizzazione e PCA congiunta
emb_all = np.vstack((emb_ted, emb_test))
names_all = np.concatenate((names_ted, names_test))

embeddings_norm = normalize(emb_all, norm='l2')

pca_clustering = PCA(n_components=50, random_state=42)
emb_pca_all = pca_clustering.fit_transform(embeddings_norm)

# ==========================================
# 3. CREAZIONE DATAFRAME GLOBALE E MERGE EXCEL
# ==========================================
df_global = pd.DataFrame({'image name': names_all})

# Identifichiamo quali righe appartengono al Test Set
nomi_test_set = set(names_test)
df_global['is_test_set'] = df_global['image name'].apply(lambda x: x in nomi_test_set)

try:
    df_excel_ted = pd.read_excel(EXCEL_FILE_TED)
    df_excel_ted['Specie Predetta'] = df_excel_ted['Specie Predetta'].fillna('Non definita')
    df_ted_clean = df_excel_ted[['image name', 'Specie Predetta']]

    df_excel_test = pd.read_excel(EXCEL_FILE_TEST)
    df_excel_test = df_excel_test.rename(columns={'Classe': 'Specie Predetta', 'Nome File': 'image name'})
    df_excel_test['Specie Predetta'] = 'test_' + df_excel_test['Specie Predetta'].astype(str)
    df_test_clean = df_excel_test[['image name', 'Specie Predetta']]

    df_excel_all = pd.concat([df_ted_clean, df_test_clean], ignore_index=True)
    df_global = df_global.merge(df_excel_all, on='image name', how='left')
    df_global['Specie Predetta'] = df_global['Specie Predetta'].fillna('Sconosciuta')

except FileNotFoundError as e:
    print(f"ATTENZIONE: File Excel non trovato ({e}).")
    df_global['Specie Predetta'] = 'Sconosciuta'

# ==========================================
# 4. SPECTRAL CLUSTERING SUL TEST SET
# ==========================================
# Isoliamo dati ed embedding del Test
maschera_test = df_global['is_test_set']
df_test = df_global[maschera_test].copy()
X_test = emb_pca_all[maschera_test]

print("\n" + "="*60)
print(f"ANALISI TEST SET ETICHETTATO ({len(X_test)} immagini)")
print("="*60)

spectral_test = SpectralClustering(
    n_clusters=NUMERO_CLUSTER, affinity='nearest_neighbors',
    n_neighbors=NUMERO_VICINI, assign_labels='cluster_qr', random_state=42
)
df_test['Cluster'] = spectral_test.fit_predict(X_test)

print("\nMATRICE DI COMPOSIZIONE (Crosstab) - TEST SET:")
print("-" * 60)
ct_test = pd.crosstab(df_test['Specie Predetta'], df_test['Cluster'])
print(ct_test)


# ==========================================
# 5. SPECTRAL CLUSTERING SUL DATASET TED
# ==========================================
# Isoliamo dati ed embedding del TED
maschera_ted = ~df_global['is_test_set']
df_ted = df_global[maschera_ted].copy()
X_ted = emb_pca_all[maschera_ted]

print("\n\n" + "="*60)
print(f"ANALISI DATASET TED NON ETICHETTATO ({len(X_ted)} immagini)")
print("="*60)

spectral_ted = SpectralClustering(
    n_clusters=NUMERO_CLUSTER, affinity='nearest_neighbors',
    n_neighbors=NUMERO_VICINI, assign_labels='cluster_qr', random_state=42
)
df_ted['Cluster'] = spectral_ted.fit_predict(X_ted)

print("\nMATRICE DI COMPOSIZIONE (Crosstab) - DATASET TED:")
print("-" * 60)
ct_ted = pd.crosstab(df_ted['Specie Predetta'], df_ted['Cluster'])
print(ct_ted)
print("\nCompletato con successo!")