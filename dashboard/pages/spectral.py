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

from utils import EMBEDDINGS_GLOBALI, DF_GLOBALE, genera_grafico_3d, genera_tabella_crosstab, calcola_percorso_hover

dash.register_page(__name__, path='/spectral', name='Spectral Clustering')

ORDINE_CATEGORIE = ['Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others', 'Labeled Set']

# ==========================================
# PREPARAZIONE DATI INIZIALI
# ==========================================
# Labeled Set 
maschera_test = DF_GLOBALE['is_test_set'] == True
X_labeled = EMBEDDINGS_GLOBALI[maschera_test]
df_labeled = DF_GLOBALE[maschera_test].copy()

# ==========================================
# LAYOUT
# ==========================================
layout = html.Div([
    html.H2("Spectral Clustering", className="mb-3", style={"color": "#430783"}),
    html.Hr(),
    
    # ---------------------------------------------------------
    # FASE 1: LABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 1: Calibrazione su un Labeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                # Selezione parametri Labelded 
                dbc.CardHeader("⚙️ Parametri Spectral Clustering", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Label("Numero di Cluster:", className="fw-bold"),
                    html.Br(),
                    dcc.Slider(id='spectral-slider-clusters', min=2, max=10, step=1, value=6, marks={i: str(i) for i in range(2, 11)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Div([
                        html.Label("Numero di Vicini:", className="fw-bold mt-3"),
                        dbc.Button("✨ Auto", id="btn-opt-neighbors-lab", size="sm", color="warning", outline=True, className="float-end mt-2")
                    ], className="d-flex justify-content-between align-items-center mt-3"),
                    dcc.Slider(id='spectral-slider-neighbors', min=2, max=20, step=1, value=5, marks={i: str(i) for i in range(2, 21, 2)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Hr(),
                    html.Div(id='spectral-metriche-box', className="mt-3"),
                    dcc.Store(id='store-top5-labeled', data=[])
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                # Grafico 3D Labeled 
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='spectral-grafico-3d', style={'height': '420px'}, clear_on_unhover=True),
                            className="p-0" 
                        ), 
                        style={'height': '420px'},
                        className="shadow-sm border-0"
                    ),
                    width=9
                ),
                
                # Box Immagine Hover Labeled 
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Preview", className="text-center fw-bold", style={"backgroundColor": "#C499F9", "fontSize": "12px", "height": "40px", "padding": "8px"}),
                        dbc.CardBody([
                            html.H6(id='spectral-hover-text', className="text-center text-secondary mb-2", style={'fontSize': '11px', 'height': '15px'}),
                        
                            html.Img(
                                id='spectral-hover-image', 
                                style={
                                    'maxWidth': '100%', 
                                    'maxHeight': '330px',    
                                    'objectFit': 'contain', 
                                    'borderRadius': '5px'
                                }
                            )
                        ], className="d-flex flex-column align-items-center justify-content-center", style={'height': '380px'}) 
                    ], style={'height': '420px'}, className="shadow-sm border-0"),
                    width=3
                )
            ], className="border-0 mb-3 align-items-stretch"),
            
            # Tabella Distribuzione Labeled
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Matrice di Distribuzione (Labeled Set)", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                        dbc.CardBody(id='spectral-tabella-crosstab')
                    ], className="shadow-sm border-0")
                )
            ])
        ], width=9)
    ]),

    html.Hr(className="my-5"),

    # ---------------------------------------------------------
    # FASE 2: UNLEABLED SET
    # ---------------------------------------------------------
    html.H4("Fase 2: Applicazione dei parametri su un Unlabeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                # Selezione parametri Unlabeled 
                dbc.CardHeader("⚙️ Parametri Spectral Clustering", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    dbc.Button("🔄 Usa parametri Labeled Set", id="btn-sync-params", color="success", outline=True, className="w-100 mb-4 shadow-sm"),
                    
                    html.Div([
                        html.Label("Numero di Cluster:", className="fw-bold"),
                        dbc.Button("✨ Auto", id="btn-opt-clusters-ted", size="sm", color="warning", outline=True, className="float-end")
                    ], className="d-flex justify-content-between align-items-center mt-3"),
                    dcc.Slider(id='ted-slider-clusters', min=2, max=10, step=1, value=4, marks={i: str(i) for i in range(2, 11)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Numero di Vicini:", className="fw-bold"),
                    dcc.Slider(id='ted-slider-neighbors', min=2, max=30, step=1, value=5, marks={i: str(i) for i in range(2, 31, 4)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Hr(),
                    html.Div(id='ted-metriche-box', className="mt-3")
                ])
            ], className="shadow-sm border-0 h-auto mb-3"),
            dbc.Card([
                dbc.CardHeader("🔍 Filtri Dataset", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    
                    html.Div([
                        dbc.Checklist(
                            id='filter-main',
                            options=[{'label': 'Seleziona Tutti', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in ORDINE_CATEGORIE],
                            value=['Curated'], 
                            inline=False, 
                            className="mb-2"
                        )
                    ], className="d-flex flex-column justify-content-flex-start", style={'padding': '10px', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'height': '100%', 'overflowY': 'auto'})
                ], className="d-flex flex-column h-100")
            ], className="shadow-sm h-auto border-0")
        ], className="border-0 mb-3 align-items-stretch",width=3),

        dbc.Col([
            dbc.Row([
                # Grafico 3D Unlabeled
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='ted-grafico-3d', style={'height': '420px'}, clear_on_unhover=True),
                            className="p-0"
                        ), 
                        style={'height': '420px'},
                        className="shadow-sm border-0"
                    ), 
                    width=9
                ),
                # Box Immagine Hover Unlabeled
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Preview", className="text-center fw-bold", style={"backgroundColor": "#C499F9", "fontSize": "12px", "height": "40px", "padding": "8px"}),
                        dbc.CardBody([
                            html.H6(id='ted-hover-text', className="text-center text-secondary mb-2", style={'fontSize': '11px'}),
                            
                            html.Img(
                                id='ted-hover-image', 
                                style={
                                    'maxWidth': '100%', 
                                    'maxHeight': '330px',    
                                    'objectFit': 'contain', 
                                    'borderRadius': '5px'
                                }
                            )
                        ], className="d-flex flex-column align-items-center justify-content-center", style={'height': '380px'}) 
                    ], style={'height': '420px'}, className="shadow-sm border-0"),
                    width=3
                )
            ], className="border-0 mb-3 align-items-stretch"),

            # Tabella Distribuzione Unlabeled
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Matrice di Distribuzione (Unlabeled Set)", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                    dbc.CardBody(id='ted-tabella-crosstab')
                ], className="shadow-sm border-0"))
            ])
        ], width=9)
    ])
], className="mb-5")


# ==========================================
# CALLBACKS
# ==========================================

# Labeled Set: Aggiornamento Grafico, Tabella e Metriche
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
    
    fig = genera_grafico_3d(df_plot, "Labeled Set")
    tabella = genera_tabella_crosstab(df_plot) 
    
    ami = adjusted_mutual_info_score(df_plot['Specie Predetta'], labels)
    ari = adjusted_rand_score(df_plot['Specie Predetta'], labels)
    
    elementi_box = [
        html.H6("Metriche Labeled:", className="text-success fw-bold"), 
        html.P(f"AMI Score: {ami:.4f}", className="mb-1"),
        html.P(f"ARI Score: {ari:.4f}", className="mb-1")
    ]
    
    if top5_data and n_neighbors == top5_data[0]['vicini'] and n_clusters == top5_data[0].get('k_clusters'):
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("🏆 Valori Migliori:", className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Vicini: {res['vicini']} | AMI: {res['ami']:.3f} | ARI: {res['ari']:.3f}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
    
    box = html.Div(elementi_box)
    
    return fig, tabella, box

# Labeled Set: Calcolo miglior valore num di vicini
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

    risultati = []
    best_score = -1
    best_vicini = 5
    
    for vicini in range(2, 21):
        grafo = kneighbors_graph(X_labeled, n_neighbors=vicini, metric='cosine', mode='connectivity', include_self=True)
        grafo = 0.5 * (grafo + grafo.T) 
        
        labels = SpectralClustering(n_clusters=k_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
        
        ami = adjusted_mutual_info_score(df_labeled['Specie Predetta'], labels)
        ari = adjusted_rand_score(df_labeled['Specie Predetta'], labels)
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
    return calcola_percorso_hover(hoverData)

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

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    df_ted = DF_GLOBALE[maschera_ted].copy()
    
    if len(X_ted) < n_clusters:
        return px.scatter_3d(title=f"Dati insufficienti ({len(X_ted)} immagini)"), "Nessun dato", html.Div("Pochi dati")
        
    grafo = kneighbors_graph(X_ted, n_neighbors=n_neighbors, metric='cosine', mode='connectivity', include_self=True)
    grafo = 0.5 * (grafo + grafo.T)
    labels = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='cluster_qr', random_state=42).fit_predict(grafo)
    df_ted['Cluster'] = labels.astype(str)
    
    fig = genera_grafico_3d(df_ted, titolo="Unlabeled Set")
    tabella = genera_tabella_crosstab(df_ted)
    
    try:
        score = silhouette_score(X_ted, labels, metric='cosine')
    except:
        score = 0

    metriche = html.Div([
        html.H6(f"Immagini analizzate: {len(X_ted)}", className="text-secondary fw-bold"),
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
    return calcola_percorso_hover(hoverData)

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
            nuovi_filtri = ['ALL'] + ORDINE_CATEGORIE
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    # Logica per il pulsante "Usa Parametri Labeled Set"
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
    maschera_ted = (~DF_GLOBALE['is_test_set']) & (DF_GLOBALE['UnifiedCategory'].isin(categorie_attive))
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
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

