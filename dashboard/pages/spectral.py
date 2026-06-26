import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from dash.exceptions import PreventUpdate

from utils import (
    DATASET_CONFIG,
    GLOBAL_EMBEDDINGS, 
    GLOBAL_DF, 
    DYNAMIC_CATEGORIES,
    generate_3d_scatter_plot,
    generate_crosstab_table,
    get_hover_image_path
)

dash.register_page(__name__, path='/spectral', name='Spectral Clustering')


# ==========================================
# PREPARAZIONE DATI INIZIALI
# ==========================================
maschera_test = GLOBAL_DF['is_labeled_set'] == True
X_labeled = GLOBAL_EMBEDDINGS[maschera_test]
df_labeled = GLOBAL_DF[maschera_test].copy()

# ==========================================
# LAYOUT
# ==========================================
layout = html.Div([
    html.Div([
        dbc.Row([
        
            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("💡 How it works: ", className="fw-bold text-primary", style={"color": "#2E4C66"}),
                        html.Span("This algorithm treats the dataset as a connected network (graph), " \
                        "where specimens are nodes and the links between them represent their degree of " \
                        "similarity. It then identifies the optimal cuts to divide the network into a " \
                        "predefined number of distinct groups.", className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),
                
            ], width=12, lg=5, className="mb-3 mb-lg-0"),

            # SECONDA COLONNA
            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("🎯 When to use it: ", className="fw-bold text-primary", style={"color": "#2E4C66"}),
                        html.Span("It is highly effective for identifying complex, non-spherical botanical structures or when specimens are linked by continuous evolutionary transitions. " \
                        "However it is best suited when there is a strong taxonomic hypothesis regarding the number of species present in the dataset.", className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),

            ], width=12, lg=5, className="mb-3 mb-lg-0"),

        ], justify="center", className="text-secondary p-4", style={"fontSize": "0.95rem", "lineHeight": "1.6"})     
    ], className="mb-2"),

    html.Hr(className="mb-5 text-muted"),
    
    # ---------------------------------------------------------
    # STEP 1: LABELED SET
    # ---------------------------------------------------------
    html.Div([
        html.H3([
            html.I(className="bi bi-gear-fill me-2 text-primary"), 
            "Step 1: Tool Calibration on Labeled Data"
        ], className="mb-4 fw-bold step-title"),
    ], className="d-flex align-items-center"),

    dbc.Row([
        # COLUMN 1: PARAMETERS (Left)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Algorithm Parameters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([

                    # Clusters Slider
                    html.Div([
                        html.Label("Number of Clusters", className="fw-bold text-primary mb-2"),
                        html.Div(
                            "Must match the number of species (already calibrated for the pre-loaded dataset).", 
                            className="text-muted small mb-3"
                        ),
                        dcc.Slider(id='spectral-slider-clusters',min=2, max=10, step=1, value=6, marks={i: str(i) for i in range(2, 11)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False, "placement": "bottom"},),
                    ], className="custom-slider-box mb-4"),
                    
                    # Neighbors Slider 
                    html.Div([
                        html.Div([
                            html.Label("Number of Neighbors", className="fw-bold text-primary mb-0"),
                            dbc.Button("Optimize ✨", id="btn-opt-neighbors-lab", size="sm", color="warning", outline=True, className="auto-tune-btn rounded-pill px-3 py-1 fw-bold"),
                        ], className="d-flex justify-content-between align-items-center mb-2"), 
                        
                        html.Div(
                            "Defines the local neighborhood size for specimen connectivity.", 
                            className="text-muted small mb-3"
                        ),

                        dcc.Slider(id='spectral-slider-neighbors', min=4, max=30, step=1, value=12, marks={i: str(i) for i in range(2, 31, 4)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False, "placement": "bottom"}),

                    ], className="custom-slider-box mb-4"),

                    # Metrics Container
                    html.Div([
                        html.Div(id='spectral-metriche-box')
                    ]),
                    
                    dcc.Store(id='store-top5-labeled', data=[])
                ], className="p-4") 
            ], className="shadow-sm border-0 h-100 rounded-3")
        ], width=12, lg=3, className="mb-4 mb-lg-0"), 
        
        # COLUMN 2: VISUALIZATION & PREVIEW (Right)
        dbc.Col([
            dbc.Row([
                # 3D Graph
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Latent Space Visualization", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}),
                            dcc.Graph(id='spectral-grafico-3d', style={'height': '380px'}, clear_on_unhover=True),
                        ], className="p-2"), 
                        style={'height': '460px'},
                        className="shadow-sm border-0 rounded-3 mb-4 mb-xl-0"
                    ),
                    width=12, xl=8
                ),
                
                # Image Preview Hover
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Specimen Preview", className="fw-bold text-muted mb-3 text-uppercase text-center", style={"letterSpacing": "1px"}),
                            html.Div([
                                html.H6(id='spectral-hover-text', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(
                                    id='spectral-hover-image', 
                                    style={
                                        'maxWidth': '100%', 
                                        'maxHeight': '320px',    
                                        'objectFit': 'contain', 
                                        'borderRadius': '8px' # Softer corners for the image
                                    }
                                )
                            ], className="d-flex flex-column align-items-center justify-content-center h-100") 
                        ], className="p-4") 
                    ], style={'height': '460px'}, className="shadow-sm border-0 rounded-3"), 
                    width=12, xl=4
                )
            ], className="mb-4"),
            
            # Crosstab Table
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H6("Distribution Matrix (Labeled Set)", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}), 
                            className="bg-transparent border-bottom-0 pt-4 pb-0"
                        ),
                        dbc.CardBody(id='spectral-tabella-crosstab', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ]),


    html.Hr(className="my-5"),

    # ---------------------------------------------------------
    # FASE 2: UNLEABLED SET
    # ---------------------------------------------------------
    html.Div([
        html.H3([
            html.I(className="bi bi-robot me-2 text-primary"), 
            "Step 2: Analyzing Unlabeled Data"
        ], className="mb-4 fw-bold step-title"),
    ], className="d-flex align-items-center mt-5"),

    dbc.Row([
        # COLUMN 1: PARAMETERS (Left)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Algorithm Parameters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    dbc.Button("Import Parameters", id="btn-sync-params", color="success", outline=True, className="sync-btn w-100 mb-2 rounded-pill shadow-sm fw-bold"),
                    html.Div(
                        "Syncs your values with the ones calibrated in Step 1", 
                        className="text-muted small text-center mb-4"
                    ),
                    
                    # Clusters Slider
                    html.Div([
                        html.Div([
                            html.Label("Number of Clusters", className="fw-bold text-primary mb-0"),
                            dbc.Button("Optimize ✨", id="btn-opt-clusters-ted", size="sm", color="warning", outline=True, className="auto-tune-btn rounded-pill px-3 py-1 fw-bold"),
                        ], className="d-flex justify-content-between align-items-center mb-2"),
                        html.Div("Adjust to match expected species diversity.", className="text-muted small mb-3"),
                        dcc.Slider(id='ted-slider-clusters', min=2, max=10, step=1, value=4, marks={i: str(i) for i in range(2, 11)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False, "placement": "bottom"}),
                    ], className="custom-slider-box mb-4"),
                    
                    # Neighbors Slider
                    html.Div([
                        html.Label("Number of Neighbors", className="fw-bold text-primary mb-2"),
                        html.Div("Defines the local neighborhood size for specimen connectivity.", className="text-muted small mb-3"),
                        dcc.Slider(id='ted-slider-neighbors', min=4, max=30, step=1, value=5, marks={i: str(i) for i in range(2, 31, 4)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False, "placement": "bottom"}),
                    ], className="custom-slider-box mb-4"),
                    
                    html.Hr(),
                    html.Div(id='ted-metriche-box')
                ], className="p-4")
            ], className="shadow-sm border-0 rounded-3 mb-4"),

            # Filter Card
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Dataset Filters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    dbc.Checklist(
                        id='filter-main',
                        options=[{'label': 'Select All', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in DYNAMIC_CATEGORIES],
                        
                        value=[DYNAMIC_CATEGORIES[1]] if len(DYNAMIC_CATEGORIES) > 1 else (DYNAMIC_CATEGORIES[:1] if DYNAMIC_CATEGORIES else []), 
                        
                        className="mb-2"
                    )
                ], className="p-4")
            ], className="shadow-sm border-0 rounded-3")
        ], width=12, lg=3, className="mb-4 mb-lg-0"),

        # COLUMN 2: VISUALIZATION & PREVIEW (Right)
        dbc.Col([
            dbc.Row([
                # 3D Graph
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Latent Space Visualization", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}),
                            dcc.Graph(id='ted-grafico-3d', style={'height': '380px'}, clear_on_unhover=True),
                        ], className="p-2"), 
                        style={'height': '460px'},
                        className="shadow-sm border-0 rounded-3 mb-4"
                    ),
                    width=12, xl=8
                ),
                # Image Preview
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Specimen Preview", className="fw-bold text-muted mb-3 text-uppercase text-center", style={"letterSpacing": "1px"}),
                            html.Div([
                                html.H6(id='ted-hover-text', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(id='ted-hover-image', style={'maxWidth': '100%', 'maxHeight': '320px', 'objectFit': 'contain', 'borderRadius': '8px'})
                            ], className="d-flex flex-column align-items-center justify-content-center h-100")
                        ], className="p-4")
                    ], style={'height': '460px'}, className="shadow-sm border-0 rounded-3"),
                    width=12, xl=4
                )
            ]),
            
            # Distribution Matrix
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.Div([
                                html.H6("Distribution Matrix (Unlabeled Set)", className="fw-bold text-muted mb-0 text-uppercase", style={"letterSpacing": "1px"}),
                                html.Div([
                                    dcc.Download(id="btn-download-excel-target"),
                                    dbc.Button(
                                        "📥 Download Clusters Distribution",
                                        id="btn-download-excel",
                                        color="success",
                                        outline=True,
                                        className="fw-bold sync-btn rounded-pill shadow-sm"
                                    )
                                ])
                            ], className="d-flex justify-content-between align-items-center px-3"),
                            className="bg-transparent border-bottom-0 pt-4 pb-0"
                        ),
                        dbc.CardBody(id='ted-tabella-crosstab', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ])
], className="mb-5")

# ==========================================
# CALLBACKS
# ==========================================

# Labeled Set: Update 3d visualization, distribution matrix and validation metrics
# ==========================================
@callback(
    [Output('spectral-grafico-3d', 'figure'), 
     Output('spectral-tabella-crosstab', 'children'), 
     Output('spectral-metriche-box', 'children')],
    [Input('spectral-slider-clusters', 'value'), 
     Input('spectral-slider-neighbors', 'value'),
     Input('store-top5-labeled', 'data')]
)
def aggiorna_spectral_labeled(n_clusters, n_neighbors, top5_data):
    grafo = kneighbors_graph(X_labeled, n_neighbors=n_neighbors, metric='cosine', mode='connectivity', include_self=True)
    grafo = 0.5 * (grafo + grafo.T)
    
    labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
    
    df_plot = df_labeled.copy()
    df_plot['Cluster'] = labels.astype(str)
    
    fig = generate_3d_scatter_plot(df_plot, title="")
    tabella = generate_crosstab_table(df_plot) 
    
    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    ami = adjusted_mutual_info_score(df_plot[colonna_target], labels)
    ari = adjusted_rand_score(df_plot[colonna_target], labels)
    
    elementi_box = [
        html.Hr(className="mb-4"),
        html.H6("Validation Metrics", className="fw-bold text-muted mb-3 text-uppercase", style={"letterSpacing": "1px"}),
        html.P(f"AMI Score: {ami:.4f}", className="mb-1"),
        html.P(f"ARI Score: {ari:.4f}", className="mb-1")
    ]
    
    if top5_data and n_neighbors == top5_data[0]['vicini'] and n_clusters == top5_data[0].get('k_clusters'):
        elementi_box.append(html.Hr(className="mb-4"))
        elementi_box.append(html.H6("Best Values:", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Neighbors: {res['vicini']} | AMI: {res['ami']:.3f} | ARI: {res['ari']:.3f}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
    
    box = html.Div(elementi_box)
    
    return fig, tabella, box

# Labeled Set: AUTO number of neighbors
# ==========================================
@callback(
    [Output('spectral-slider-neighbors', 'value', allow_duplicate=True),
     Output('store-top5-labeled', 'data')],
    Input('btn-opt-neighbors-lab', 'n_clicks'),
    State('spectral-slider-clusters', 'value'),
    prevent_initial_call=True
)
def auto_ottimizza_vicini_labeled(n_clicks, k_clusters):
    if not n_clicks:
        raise PreventUpdate

    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    risultati = []
    best_score = -1
    best_vicini = 5
    
    for vicini in range(4, 30):
        grafo = kneighbors_graph(X_labeled, n_neighbors=vicini, metric='cosine', mode='connectivity', include_self=True)
        grafo = 0.5 * (grafo + grafo.T) 
        
        labels = SpectralClustering(n_clusters=k_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
        
        ami = adjusted_mutual_info_score(df_labeled[colonna_target], labels)
        ari = adjusted_rand_score(df_labeled[colonna_target], labels)
        score = (ami + ari) / 2.0
        
        risultati.append({'vicini': vicini, 'ami': ami, 'ari': ari, 'score': score, 'k_clusters': k_clusters}) 
        
        if score > best_score:
            best_score = score
            best_vicini = vicini
            
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    return best_vicini, top5

# Labeled Set: Hover dell'Immagine
# ==========================================
@callback(
    [Output('spectral-hover-image', 'src'),
     Output('spectral-hover-text', 'children')],
    Input('spectral-grafico-3d', 'hoverData')
)
def aggiorna_hover_spectral_labeled(hoverData):
    return get_hover_image_path(hoverData)

# UNlabeled Set: Aggiornamento Grafico, Tabella e Metriche
# ==========================================
@callback(
    [Output('ted-grafico-3d', 'figure'),
     Output('ted-tabella-crosstab', 'children'),
     Output('ted-metriche-box', 'children')],
    [Input('filter-main', 'value'),
     Input('ted-slider-clusters', 'value'),
     Input('ted-slider-neighbors', 'value')]
)
def aggiorna_spectral_ted(categorie, n_clusters, n_neighbors):
    if not categorie:
        return px.scatter_3d(title="Seleziona almeno un filtro"), "Nessun dato", html.Div("Seleziona almeno un filtro")
    
    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie

    maschera_ted = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    df_ted = GLOBAL_DF[maschera_ted].copy()
    
    n_samples = len(X_ted)
    
    # Il numero di immagini deve essere matematicamente maggiore del numero di cluster
    if n_samples <= n_clusters:
        return px.scatter_3d(title=f"Dati insufficienti ({n_samples} img per {n_clusters} cluster)"), "Nessun dato", html.Div("Riduci i cluster o seleziona più dati")
        
    n_neighbors_safe = min(n_neighbors, n_samples)
        
    grafo = kneighbors_graph(X_ted, n_neighbors=n_neighbors_safe, metric='cosine', mode='connectivity', include_self=True)
    grafo = 0.5 * (grafo + grafo.T)
    
    # Fallback di sicurezza: per piccolissimi dataset convertiamo in matrice densa
    if n_samples < 20:
        grafo = grafo.toarray()
        
    labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
    df_ted['Cluster'] = labels.astype(str)
    
    fig = generate_3d_scatter_plot(df_ted, title=" ")
    tabella = generate_crosstab_table(df_ted)
    
    try:
        score = silhouette_score(X_ted, labels, metric='cosine')
    except:
        score = 0

    metriche = html.Div([
        html.H6(f"Number of samples: {n_samples}", className="text-secondary fw-bold"),
        html.P(f"Silhouette Score: {score:.4f}", className="text-success fw-bold")
    ])
    
    return fig, tabella, metriche

# UNlabeled Set: Hover dell'Immagine 
# ==========================================
@callback(
    [Output('ted-hover-image', 'src'),
     Output('ted-hover-text', 'children')],
    Input('ted-grafico-3d', 'hoverData')
)
def mostra_immagine_hover(hoverData):
    return get_hover_image_path(hoverData)

# UNlabeled Set: Gestione Filtri "Seleziona Tutti" e Sincronizzazione Parametri
# ==========================================
@callback(
    [Output('filter-main', 'value'),
     Output('ted-slider-clusters', 'value'),
     Output('ted-slider-neighbors', 'value')],
    [Input('filter-main', 'value'),
     Input('btn-sync-params', 'n_clicks')],
    [State('spectral-slider-clusters', 'value'),
     State('spectral-slider-neighbors', 'value'),
     State('ted-slider-clusters', 'value'),
     State('ted-slider-neighbors', 'value')]
)
def gestisci_input_ted(filtri_selezionati, n_clicks, lab_k, lab_neigh, ted_k, ted_neigh):
    triggered_id = ctx.triggered_id

    nuovi_filtri = filtri_selezionati
    if triggered_id == 'filter-main':
        if 'ALL' in filtri_selezionati:
            nuovi_filtri = ['ALL'] + DYNAMIC_CATEGORIES
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    if triggered_id == 'btn-sync-params':
        return nuovi_filtri, lab_k, lab_neigh

    return nuovi_filtri, ted_k, ted_neigh


# Unlabeled Set: Massimizza Silhouette Score
# ==========================================
@callback(
    Output('ted-slider-clusters', 'value', allow_duplicate=True),
    Input('btn-opt-clusters-ted', 'n_clicks'),
    [State('ted-slider-neighbors', 'value'),
     State('filter-main', 'value')],
    prevent_initial_call=True
)
def auto_ottimizza_cluster_ted(n_clicks, n_neighbors, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = (~GLOBAL_DF['is_labeled_set']) & (GLOBAL_DF['UnifiedCategory'].isin(categorie_attive))
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    if len(X_ted) < 10: 
        raise PreventUpdate

    grafo = kneighbors_graph(X_ted, n_neighbors=n_neighbors, metric='cosine', mode='connectivity', include_self=True)
    grafo = 0.5 * (grafo + grafo.T)

    best_silhouette = -1
    best_k = 4

    for k in range(2, 11):
        labels = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
        
        try:
            score = silhouette_score(X_ted, labels, metric='cosine')
            if score > best_silhouette:
                best_silhouette = score
                best_k = k
        except:
            continue
           
    return best_k

# ==========================================
# DOWNOLOAD
# ==========================================
@callback(
    Output("btn-download-excel-target", "data"),
    Input("btn-download-excel", "n_clicks"),
    [State('filter-main', 'value'),
     State('ted-slider-clusters', 'value'),
     State('ted-slider-neighbors', 'value')],
    prevent_initial_call=True
)
def download_excel_spectral(n_clicks, categorie, n_clusters, n_neighbors):
    if not n_clicks or not categorie:
        raise PreventUpdate

    # 1. Filtriamo i dati esattamente come nella visualizzazione della Fase 2
    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    df_ted = GLOBAL_DF[maschera_ted].copy()
    
    if len(X_ted) < n_clusters:
        raise PreventUpdate
        
    # 2. Calcoliamo i cluster attuali
    grafo = kneighbors_graph(X_ted, n_neighbors=n_neighbors, metric='cosine', mode='connectivity', include_self=True)
    grafo = 0.5 * (grafo + grafo.T)
    labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
    
    df_ted['Cluster'] = labels.astype(str)
    
    # 3. Isoliamo solo le colonne richieste usando DATASET_CONFIG
    colonna_id = DATASET_CONFIG['IMAGE_ID_COL']
    df_download = df_ted[[colonna_id, 'Cluster']].copy()
    
    # Ordeniamo per rendere il file ordinato e leggibile
    df_download = df_download.sort_values(by='Cluster')

    # 4. Inviamo il file Excel all'utente tramite Dash
    return dcc.send_data_frame(df_download.to_excel, "selected_clusters_spectral.xlsx", index=False)