import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from dash.exceptions import PreventUpdate

from utils import EMBEDDINGS_GLOBALI, DF_GLOBALE, genera_grafico_3d, genera_tabella_crosstab, calcola_percorso_hover

dash.register_page(__name__, path='/agglomerative', name='Agglomerative Clustering')

ORDINE_CATEGORIE = ['Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others', 'Labeled Set']

# ==========================================
# PREPARAZIONE DATI INIZIALI
# ==========================================
# Labeled Set 
maschera_test = DF_GLOBALE['is_test_set'] == True
X_labeled = EMBEDDINGS_GLOBALI[maschera_test]
df_labeled = DF_GLOBALE[maschera_test].copy()

# Opzioni per il dropdown del Linkage
LINKAGE_OPTIONS = [
    {'label': 'Ward (Varianza Minima)', 'value': 'ward'},
    {'label': 'Average (Media Distanze)', 'value': 'average'},
    {'label': 'Complete (Distanza Massima)', 'value': 'complete'},
    {'label': 'Single (Distanza Minima)', 'value': 'single'}
]

# ==========================================
# LAYOUT
# ==========================================
layout = html.Div([
    html.H2("Agglomerative Clustering", className="mb-3", style={"color": "#430783"}),
    html.Hr(),
    
    # ---------------------------------------------------------
    # FASE 1: LABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 1: Calibrazione su un Labeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                # Selezione parametri Labeled 
                dbc.CardHeader("⚙️ Parametri Agglomerative", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Label("Numero di Cluster:", className="fw-bold"),
                    html.Br(),
                    dcc.Slider(id='agg-slider-clusters', min=2, max=10, step=1, value=6, marks={i: str(i) for i in range(2, 11)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Div([
                        html.Label("Metodo di Linkage:", className="fw-bold mt-3"),
                        dbc.Button("✨ Auto", id="btn-opt-linkage-lab", size="sm", color="warning", outline=True, className="float-end mt-2")
                    ], className="d-flex justify-content-between align-items-center mt-3"),
                    
                    # Sostituito lo slider con il Dropdown
                    dcc.Dropdown(
                        id='agg-dropdown-linkage',
                        options=LINKAGE_OPTIONS,
                        value='ward',
                        clearable=False,
                        className="mb-4 mt-2"
                    ),
                    
                    html.Hr(),
                    html.Div(id='agg-metriche-box', className="mt-3"),
                    dcc.Store(id='store-top5-labeled-agg', data=[])
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                # Grafico 3D Labeled 
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='agg-grafico-3d', style={'height': '420px'}, clear_on_unhover=True),
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
                            html.H6(id='agg-hover-text', className="text-center text-secondary mb-2", style={'fontSize': '11px', 'height': '15px'}),
                        
                            html.Img(
                                id='agg-hover-image', 
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
                        dbc.CardBody(id='agg-tabella-crosstab')
                    ], className="shadow-sm border-0")
                )
            ])
        ], width=9)
    ]),

    html.Hr(className="my-5"),

    # ---------------------------------------------------------
    # FASE 2: UNLABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 2: Applicazione dei parametri su un Unlabeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                # Selezione parametri Unlabeled 
                dbc.CardHeader("⚙️ Parametri Agglomerative", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    dbc.Button("🔄 Usa parametri Labeled Set", id="btn-sync-params-agg", color="success", outline=True, className="w-100 mb-4 shadow-sm"),
                    
                    html.Div([
                        html.Label("Numero di Cluster:", className="fw-bold"),
                        dbc.Button("✨ Auto", id="btn-opt-clusters-ted-agg", size="sm", color="warning", outline=True, className="float-end")
                    ], className="d-flex justify-content-between align-items-center mt-3"),
                    dcc.Slider(id='ted-slider-clusters-agg', min=2, max=10, step=1, value=4, marks={i: str(i) for i in range(2, 11)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Metodo di Linkage:", className="fw-bold mt-2"),
                    
                    # Sostituito lo slider con il Dropdown
                    dcc.Dropdown(
                        id='ted-dropdown-linkage',
                        options=LINKAGE_OPTIONS,
                        value='ward',
                        clearable=False,
                        className="mb-4 mt-2"
                    ),
                    
                    html.Hr(),
                    html.Div(id='ted-metriche-box-agg', className="mt-3")
                ])
            ], className="shadow-sm border-0 h-auto mb-3"),
            dbc.Card([
                dbc.CardHeader("🔍 Filtri Dataset", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    
                    html.Div([
                        dbc.Checklist(
                            id='filter-main-agg',
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
                            dcc.Graph(id='ted-grafico-3d-agg', style={'height': '420px'}, clear_on_unhover=True),
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
                            html.H6(id='ted-hover-text-agg', className="text-center text-secondary mb-2", style={'fontSize': '11px'}),
                            
                            html.Img(
                                id='ted-hover-image-agg', 
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
                    dbc.CardBody(id='ted-tabella-crosstab-agg')
                ], className="shadow-sm border-0"))
            ])
        ], width=9)
    ])
], className="mb-5")

# Labeled Set: Aggiornamento Grafico, Tabella e Metriche
# ==========================================
@callback(
    [Output('agg-grafico-3d', 'figure'), 
     Output('agg-tabella-crosstab', 'children'), 
     Output('agg-metriche-box', 'children')],
    [Input('agg-slider-clusters', 'value'), 
     Input('agg-dropdown-linkage', 'value'), # Sostituisce neighbors
     Input('store-top5-labeled-agg', 'data')]
)
def aggiorna_agg_labeled(n_clusters, linkage, top5_data):
    # Il metodo 'ward' richiede tassativamente la distanza euclidea
    metrica = 'euclidean' if linkage == 'ward' else 'cosine'
    
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric=metrica, linkage=linkage)
    labels = clusterer.fit_predict(X_labeled)
    
    df_plot = df_labeled.copy()
    df_plot['Cluster'] = labels.astype(str)
    
    fig = genera_grafico_3d(df_plot, "Labeled Set (Agglomerative)")
    tabella = genera_tabella_crosstab(df_plot) 
    
    ami = adjusted_mutual_info_score(df_plot['Specie Predetta'], labels)
    ari = adjusted_rand_score(df_plot['Specie Predetta'], labels)
    
    elementi_box = [
        html.H6("Metriche Labeled:", className="text-success fw-bold"), 
        html.P(f"AMI Score: {ami:.4f}", className="mb-1"),
        html.P(f"ARI Score: {ari:.4f}", className="mb-1")
    ]
    
    if top5_data and linkage == top5_data[0]['linkage'] and n_clusters == top5_data[0].get('k_clusters'):
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("🏆 Valori Migliori:", className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Linkage: {res['linkage']} | AMI: {res['ami']:.3f} | ARI: {res['ari']:.3f}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
    
    box = html.Div(elementi_box)
    
    return fig, tabella, box

# Labeled Set: Calcolo miglior valore Linkage (Auto-Tuning)
# ==========================================
@callback(
    [Output('agg-dropdown-linkage', 'value', allow_duplicate=True),
     Output('store-top5-labeled-agg', 'data')],
    Input('btn-opt-linkage-lab', 'n_clicks'),
    State('agg-slider-clusters', 'value'),
    prevent_initial_call=True
)
def auto_ottimizza_linkage_labeled(n_clicks, k_clusters):
    if not n_clicks:
        raise PreventUpdate

    risultati = []
    best_score = -1
    best_linkage = 'ward'
    
    # Esploriamo le combinazioni di Linkage e Metrica possibili
    combinazioni = [
        ('ward', 'euclidean'),
        ('average', 'cosine'), ('average', 'euclidean'),
        ('complete', 'cosine'), ('complete', 'euclidean'),
        ('single', 'cosine')
    ]
    
    for link, metrica in combinazioni:
        labels = AgglomerativeClustering(n_clusters=k_clusters, metric=metrica, linkage=link).fit_predict(X_labeled)
        
        ami = adjusted_mutual_info_score(df_labeled['Specie Predetta'], labels)
        ari = adjusted_rand_score(df_labeled['Specie Predetta'], labels)
        score = (ami + ari) / 2.0
        
        risultati.append({'linkage': link, 'metrica': metrica, 'ami': ami, 'ari': ari, 'score': score, 'k_clusters': k_clusters}) 
        
        if score > best_score:
            best_score = score
            best_linkage = link
            
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    return best_linkage, top5

# Labeled Set: Hover dell'Immagine
# ==========================================
@callback(
    [Output('agg-hover-image', 'src'),
     Output('agg-hover-text', 'children')],
    Input('agg-grafico-3d', 'hoverData')
)
def aggiorna_hover_agg_labeled(hoverData):
    return calcola_percorso_hover(hoverData)


# UNlabeled Set: Aggiornamento Grafico, Tabella e Metriche
# ==========================================
@callback(
    [Output('ted-grafico-3d-agg', 'figure'),
     Output('ted-tabella-crosstab-agg', 'children'),
     Output('ted-metriche-box-agg', 'children')],
    [Input('filter-main-agg', 'value'),
     Input('ted-slider-clusters-agg', 'value'),
     Input('ted-dropdown-linkage', 'value')]
)
def aggiorna_agg_ted(categorie, n_clusters, linkage):
    if not categorie:
        return px.scatter_3d(title="Seleziona almeno un filtro"), "Nessun dato", html.Div("Seleziona almeno un filtro")

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    df_ted = DF_GLOBALE[maschera_ted].copy()
    
    if len(X_ted) < n_clusters:
        return px.scatter_3d(title=f"Dati insufficienti ({len(X_ted)} immagini)"), "Nessun dato", html.Div("Pochi dati")
        
    metrica = 'euclidean' if linkage == 'ward' else 'cosine'
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric=metrica, linkage=linkage).fit_predict(X_ted)
    df_ted['Cluster'] = labels.astype(str)
    
    fig = genera_grafico_3d(df_ted, titolo="Unlabeled Set (Agglomerative)")
    tabella = genera_tabella_crosstab(df_ted)
    
    try:
        score = silhouette_score(X_ted, labels, metric=metrica)
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
    [Output('ted-hover-image-agg', 'src'),
     Output('ted-hover-text-agg', 'children')],
    Input('ted-grafico-3d-agg', 'hoverData')
)
def mostra_immagine_hover_agg(hoverData):
    return calcola_percorso_hover(hoverData)

# UNlabeled Set: Gestione Filtri "Seleziona Tutti" e Sincronizzazione Parametri
# ==========================================
@callback(
    [Output('filter-main-agg', 'value'),
     Output('ted-slider-clusters-agg', 'value'),
     Output('ted-dropdown-linkage', 'value')],
    [Input('filter-main-agg', 'value'),
     Input('btn-sync-params-agg', 'n_clicks')],
    [State('agg-slider-clusters', 'value'),
     State('agg-dropdown-linkage', 'value'),
     State('ted-slider-clusters-agg', 'value'),
     State('ted-dropdown-linkage', 'value')]
)
def gestisci_input_agg_ted(filtri_selezionati, n_clicks, lab_k, lab_link, ted_k, ted_link):
    triggered_id = ctx.triggered_id
    nuovi_filtri = filtri_selezionati
    
    if triggered_id == 'filter-main-agg':
        if 'ALL' in filtri_selezionati:
            nuovi_filtri = ['ALL'] + ORDINE_CATEGORIE
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    if triggered_id == 'btn-sync-params-agg':
        return nuovi_filtri, lab_k, lab_link

    return nuovi_filtri, ted_k, ted_link


# Unlabeled Set: Massimizza Silhouette Score (Auto-Tuning Clusters)
# ==========================================
@callback(
    Output('ted-slider-clusters-agg', 'value', allow_duplicate=True),
    Input('btn-opt-clusters-ted-agg', 'n_clicks'),
    [State('ted-dropdown-linkage', 'value'),
     State('filter-main-agg', 'value')],
    prevent_initial_call=True
)
def auto_ottimizza_cluster_agg_ted(n_clicks, linkage, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = (~DF_GLOBALE['is_test_set']) & (DF_GLOBALE['UnifiedCategory'].isin(categorie_attive))
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    
    if len(X_ted) < 10: 
        raise PreventUpdate

    best_silhouette = -1
    best_k = 4
    metrica = 'euclidean' if linkage == 'ward' else 'cosine'

    for k in range(2, 11):
        labels = AgglomerativeClustering(n_clusters=k, metric=metrica, linkage=linkage).fit_predict(X_ted)
        
        try:
            score = silhouette_score(X_ted, labels, metric=metrica)
            if score > best_silhouette:
                best_silhouette = score
                best_k = k
        except:
            continue
            
    return best_k