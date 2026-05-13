import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

NPZ_FILE = 'features_resnet_custom_cropped_ALL_DATA.npz'
EXCEL_FILE = 'dataset.xlsx'

# --- PREPARAZIONE DATI ---
data = np.load(NPZ_FILE)
embeddings_norm = normalize(data['embeddings'], norm='l2')
coords_3d = PCA(n_components=3).fit_transform(data['embeddings'])

df_global = pd.DataFrame(coords_3d, columns=['x', 'y', 'z'])
df_global['image name'] = data['names']

ORDINE_CATEGORIE = ['Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others']

try:
    df_excel = pd.read_excel(EXCEL_FILE)
    df_excel['datasetCategory'] = df_excel['datasetCategory'].fillna('Vuoto')
    df_excel['personalAnnotation'] = df_excel['personalAnnotation'].fillna('Vuoto')
    
    df_global = df_global.merge(df_excel[['image name', 'datasetCategory', 'personalAnnotation', 'Specie Predetta']], on='image name', how='left')
    df_global['Specie Predetta'] = df_global['Specie Predetta'].fillna('Non definita')

    def assegna_categoria_unificata(row):
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

except FileNotFoundError:
    print(f"ATTENZIONE: File '{EXCEL_FILE}' non trovato. Verranno usate etichette fittizie.")
    df_global['UnifiedCategory'] = 'Tutte'

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# --- LAYOUT ---
app.layout = dbc.Container([
    # Titolo e logo
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Img(src=app.get_asset_url('logo.png'), height="60px", className="me-3"),
                html.H2("Orchid Visual Explorer", className="text-primary mb-0 fw-bold")
            ], className="d-flex justify-content-start align-items-center mt-3 mb-4"), 
            width=12
        )
    ]),
    
    dbc.Row([
        # Selezione Filtri
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("🔍 Filtra Dataset", className="text-secondary mb-4 fw-bold text-center"),
                html.Div([
                    dbc.Checklist(
                        id='filter-main',
                        options=[{'label': 'Seleziona Tutti', 'value': 'ALL'}] + 
                                [{'label': c, 'value': c} for c in ORDINE_CATEGORIE],
                        value=['Curated'], 
                        inline=False, 
                        className="mb-2"
                    )
                ], className="d-flex flex-column justify-content-center", style={'padding': '10px', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'backgroundColor': '#f8f9fa', 'height': '100%', 'overflowY': 'auto'})
            ], className="d-flex flex-column h-100")
        ], className="shadow-sm h-100"), width=2),

        # Selezione Parametri DBSCAN
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("⚙️ Parametri DBSCAN", className="card-title text-secondary mb-4 text-center fw-bold"),
                html.Div([
                    html.Label("EPS (Distanza L2):", className="fw-bold small"),
                    dcc.Slider(id='eps', min=0.05, max=0.5, step=0.01, value=0.3, tooltip={"always_visible": True}),
                    html.Br(),
                    html.Label("Min Samples (Densità):", className="fw-bold small"),
                    dcc.Slider(id='min-samples', min=2, max=30, step=1, value=15, tooltip={"always_visible": True})
                ], className="d-flex flex-column justify-content-center flex-grow-1") 
            ], className="d-flex flex-column h-100") 
        ], className="shadow-sm h-100"), width=5),

        #Tabella Conteggi
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("📊 Distribuzione Specie nei Cluster", className="text-secondary mb-3 fw-bold text-center"),
                html.Div(id='tabella-cluster-container', style={'overflowX': 'auto', 'fontSize': '0.8rem'})
            ], className="d-flex flex-column h-100")
        ], className="shadow-sm h-100"), width=5)
        
    ], className="mb-5 align-items-stretch", style={'minHeight': '38vh'}),
    
    # RIGA INFERIORE: Mappa 3D + Immagine Hover
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                dcc.Loading(
                    custom_spinner=html.Div("🌸", className="spin-emoji"),
                    children=dcc.Graph(id='plot', style={'height': '55vh'})
                )
            ])
        ], className="shadow-sm"), width=8),
        
        dbc.Col(dbc.Card([
            dbc.CardBody(id='hover-info', className="d-flex flex-column align-items-center justify-content-center", style={'minHeight': '55vh'})
        ], className="shadow-sm h-100"), width=4)
    ], className="mb-4 align-items-stretch")
    
], fluid=True, style={'backgroundColor': "#EFEFEF", 'minHeight': '100vh', 'padding': '20px'}) 

# --- FUNZIONI E CALLBACKS ---
def get_filtered_data(selected_list):
    if not selected_list: 
        selected_list = []
        
    if 'ALL' in selected_list:
        mask = df_global['UnifiedCategory'].isin(ORDINE_CATEGORIE + ['Sconosciuta'])
    else:
        mask = df_global['UnifiedCategory'].isin(selected_list)
        
    idx_mask = mask.values
    return embeddings_norm[idx_mask], df_global[idx_mask].copy()

@app.callback(
    Output('plot', 'figure'),
    [Input('eps', 'value'), Input('min-samples', 'value'),
     Input('filter-main', 'value')]
)
def update_plot(eps, min_s, selected_list):
    emb_filt, df_plot = get_filtered_data(selected_list)
    
    if len(emb_filt) <= min_s:
        fig = px.scatter_3d(title=f"⚠️ Punti insufficienti per creare cluster")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        return fig

    labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(emb_filt)
    df_plot['Cluster'] = [f'C_{c}' if c != -1 else 'Rumore' for c in labels]
    df_plot.sort_values('Cluster', inplace=True)
    
    color_map = {c: px.colors.qualitative.Plotly[i % 10] for i, c in enumerate(df_plot['Cluster'].unique()) if c != 'Rumore'}
    color_map['Rumore'] = '#888888'
    
    fig = px.scatter_3d(df_plot, x='x', y='y', z='z', color='Cluster', custom_data=['image name'], color_discrete_map=color_map)
    fig.update_traces(marker=dict(size=4, line=dict(width=0)))
    
    for trace in fig.data:
        if trace.name == 'Rumore':
            trace.marker.color = 'rgba(100, 100, 100, 0.1)'
        else:
            trace.marker.opacity = 0.9
            
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', scene=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

@app.callback(
    Output('hover-info', 'children'),
    Input('plot', 'hoverData')
)
def update_hover(hoverData):
    if not hoverData or 'customdata' not in hoverData['points'][0]: 
        return html.Div([
            html.I(className="bi bi-image text-muted", style={'fontSize': '4rem'}),
            html.P("Passa il cursore su un punto👀", className="text-muted mt-3 text-center")
        ])
    
    img_name = hoverData['points'][0]['customdata'][0]
    return html.Div([
        html.Img(src=app.get_asset_url(img_name), style={'maxWidth': '100%', 'maxHeight': '45vh', 'objectFit': 'contain', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'}),
        html.H5(img_name, className="mt-3 text-center text-primary fw-bold")
    ], style={'width': '100%', 'textAlign': 'center'})

@app.callback(
    Output('tabella-cluster-container', 'children'),
    [Input('eps', 'value'), Input('min-samples', 'value'),
     Input('filter-main', 'value')]
)
def update_table(eps, min_s, selected_list):
    emb_filt, df_plot = get_filtered_data(selected_list)
    
    if len(emb_filt) <= min_s:
        return html.P("Dati insufficienti per la tabella", className="text-muted text-center")

    labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(emb_filt)
    df_plot['Cluster'] = [f'C_{c}' if c != -1 else 'Rumore' for c in labels]

    ct = pd.crosstab(df_plot['Specie Predetta'], df_plot['Cluster'])
    
    # Trasforma la tabella Pandas in una tabella HTML 
    return dbc.Table.from_dataframe(
        ct.reset_index(), 
        striped=True, 
        bordered=True, 
        hover=True, 
        responsive=True,
        className="mb-0"
    )

@app.callback(
    Output('filter-main', 'value'),
    [Input('filter-main', 'value')],
    [State('filter-main', 'options')]
)
def sync_checklists(selected_values, options):
    ctx = dash.callback_context
    if not ctx.triggered:
        return selected_values

    all_values = [opt['value'] for opt in options if opt['value'] != 'ALL']
    if 'ALL' in selected_values and len(selected_values) < len(options):
        return ['ALL'] + all_values
    
    if 'ALL' in selected_values and len(selected_values) == len(options):
        if len(selected_values) <= len(all_values): 
             return [v for v in selected_values if v != 'ALL']

    if 'ALL' not in selected_values and len(selected_values) == len(all_values):
        return []

    return selected_values

# --- SERVER RUN ---
if __name__ == '__main__':
    app.run(debug=True)
