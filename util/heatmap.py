import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import adjusted_rand_score

# --- DEFINIZIONE VARIABILI ---
FILE_NAME_A = r"C:\Users\aless\OneDrive - Università di Napoli Federico II\OrchID\After_CROPPING\Risultati_Agglomerative_features_k4_resnet_custom_cropped_curated\agg_k4.xlsx"
FILE_NAME_B = r"C:\Users\aless\OneDrive - Università di Napoli Federico II\OrchID\After_CROPPING\Risultati_Agglomerative_k4_features_resnet18_imagenet_cropped_curated\agg_k4.xlsx"

ID_COLUMN = 'image name' 
CLUSTER_COLUMN = 'ID_Cluster'

# ----------------------------

# Caricamento dati
df1 = pd.read_excel(FILE_NAME_A)
df2 = pd.read_excel(FILE_NAME_B)

df1[ID_COLUMN] = df1[ID_COLUMN].astype(str).str.strip()
df2[ID_COLUMN] = df2[ID_COLUMN].astype(str).str.strip()

# Merge dei dati
df_final = pd.merge(df1, df2, on=ID_COLUMN, suffixes=('_Custom', '_Imagenet'))
df_final = df_final.dropna(subset=[f'{CLUSTER_COLUMN}_Custom', f'{CLUSTER_COLUMN}_Imagenet'])

# Creazione della matrice di contingenza 
cross_tab = pd.crosstab(df_final[f'{CLUSTER_COLUMN}_Imagenet'], 
                        df_final[f'{CLUSTER_COLUMN}_Custom'])

# Normalizzazione 
cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0)

# Visualizzazione Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cross_tab_norm, annot=True, cmap="YlGnBu", fmt=".2f")

plt.title('Sovrapposizione Cluster: Imagenet vs Custom A')
plt.xlabel('Cluster Custom')
plt.ylabel('Cluster Imagenet')

plt.show()

# Calcolo del punteggio ARI 
ari_score = adjusted_rand_score(df_final[f'{CLUSTER_COLUMN}_Custom'], 
                                df_final[f'{CLUSTER_COLUMN}_Imagenet'])

print("-" * 30)
print(f"Numero di immagini analizzate (matchate): {len(df_final)}")
print(f"L'Adjusted Rand Index (ARI) tra le due reti è: {ari_score:.4f}")
print("-" * 30)