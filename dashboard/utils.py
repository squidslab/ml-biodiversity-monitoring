import numpy as np
import pandas as pd
import dash
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# ==========================================
# COSTANTI E PERCORSI FILE
# ==========================================
NPZ_FILE_TED = 'features_resnet_custom_cropped_ALL_DATA.npz'
EXCEL_FILE_TED = 'dataset.xlsx'
NPZ_FILE_TEST = 'features_resnet_custom_base_TEST.npz'
EXCEL_FILE_TEST = 'Registro_Dataset_ResNet18.xlsx'

# ==========================================
# FUNZIONE PRINCIPALE DI CARICAMENTO DATI
# ==========================================
def carica_tutti_i_dati():
    """
    Carica i file .npz, calcola la PCA 3D per i grafici, legge i file Excel
    e restituisce la matrice degli embedding e il DataFrame globale.
    """
    try:
        data_ted = np.load(NPZ_FILE_TED)
        data_test = np.load(NPZ_FILE_TEST)
        emb_all = np.vstack((data_ted['embeddings'], data_test['embeddings']))
        names_all = np.concatenate((data_ted['names'], data_test['names']))
        
        embeddings_norm = normalize(emb_all, norm='l2')
        
        # PCA per i grafici 
        coords_3d = PCA(n_components=3).fit_transform(embeddings_norm)
        df_global = pd.DataFrame(coords_3d, columns=['x', 'y', 'z'])
        df_global['image name'] = names_all
        
        # Maschera Test Set
        nomi_test = set(data_test['names'])
        df_global['is_test_set'] = df_global['image name'].apply(lambda x: x in nomi_test)
        
        # Merge Excel TED
        df_excel_ted = pd.read_excel(EXCEL_FILE_TED)
        df_excel_ted['datasetCategory'] = df_excel_ted['datasetCategory'].fillna('Vuoto')
        df_excel_ted['personalAnnotation'] = df_excel_ted['personalAnnotation'].fillna('Vuoto')
        df_ted_clean = df_excel_ted[['image name', 'datasetCategory', 'personalAnnotation', 'Specie Predetta']]
        
        # Merge Excel TEST
        df_excel_test = pd.read_excel(EXCEL_FILE_TEST)
        df_excel_test = df_excel_test.rename(columns={'Classe': 'Specie Predetta', 'Nome File': 'image name'})
        df_excel_test['Specie Predetta'] = 'labeled_' + df_excel_test['Specie Predetta'].astype(str)
        df_excel_test['personalAnnotation'] = 'Vuoto'
        df_excel_test['datasetCategory'] = 'Labeled Set'
        df_test_clean = df_excel_test[['image name', 'datasetCategory', 'personalAnnotation', 'Specie Predetta']]
        
        df_excel_all = pd.concat([df_ted_clean, df_test_clean], ignore_index=True)
        df_global = df_global.merge(df_excel_all, on='image name', how='left')
        
        df_global['Specie Predetta'] = df_global['Specie Predetta'].fillna('Sconosciuta')
        def assegna_categoria_unificata(row):
            if row['is_test_set']:
                return 'Labeled Set'
            
            cat = str(row['datasetCategory']).strip().upper()
            ann = str(row['personalAnnotation']).strip().lower()
            
            if ann == 'curated': return 'Curated'
            if cat == 'USABLE' and ann == 'vuoto': return 'Usable'
            if cat == 'HARDCORE' and ann == 'vuoto': return 'Hardcore'
            if ann == 'ruined_surface': return 'Ruined Surface'
            if ann == 'hands': return 'Hands'
            if ann == 'others': return 'Others'
            
            return 'Sconosciuta' 
        
        df_global['UnifiedCategory'] = df_global.apply(assegna_categoria_unificata, axis=1)
        
        return embeddings_norm, df_global
        
    except Exception as e:
        print(f"ERRORE CRITICO in utils.py: {e}")
        return np.array([]), pd.DataFrame()

# ==========================================
# COMPONENTI UI CONDIVISI (Grafici e Tabelle)
# ==========================================
def genera_grafico_3d(df_plot, titolo="Spazio Latente 3D"):
    """
    Riceve un DataFrame (con x, y, z, Cluster, image name, Specie Predetta)
    e restituisce una figura Plotly standardizzata.
    """
    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z', 
        color='Cluster',
        hover_name='image name',
        custom_data=['Specie Predetta', 'image name'],
        title=titolo,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.8),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Specie: %{customdata[0]}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40), 
        scene=dict(xaxis_title='', yaxis_title='', zaxis_title='')
    )
    return fig

def genera_tabella_crosstab(df_plot):
    """
    Riceve un DataFrame, calcola la crosstab e restituisce una tabella Dash Bootstrap.
    """
    ct = pd.crosstab(df_plot['Specie Predetta'], df_plot['Cluster'])
    tabella = dbc.Table.from_dataframe(
        ct.reset_index(), 
        striped=True, bordered=True, hover=True, responsive=True, size="sm"
    )
    return tabella

def calcola_percorso_hover(hoverData):
    """
    Estrae il nome dell'immagine dall'hoverData di Plotly
    e genera il percorso corretto per Dash.
    """
    if hoverData is None:
        return "", "Passa il mouse su un punto"

    nome_immagine = hoverData['points'][0]['customdata'][1]
    
    percorso_immagine = dash.get_asset_url(nome_immagine)
    
    return percorso_immagine, f"File: {nome_immagine}"

# ==========================================
# 4. ESECUZIONE AL CARICAMENTO (Cache)
# ==========================================
# Chiamiamo la funzione una volta sola quando il modulo viene importato.
# In questo modo i dati restano in memoria (RAM) e l'app è velocissima.
EMBEDDINGS_GLOBALI, DF_GLOBALE = carica_tutti_i_dati()