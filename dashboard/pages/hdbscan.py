import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np
import hdbscan
import umap.umap_ as umap
from sklearn.metrics import fowlkes_mallows_score, silhouette_score

from utils import EMBEDDINGS_GLOBALI, DF_GLOBALE, genera_grafico_3d, genera_tabella_crosstab, calcola_percorso_hover

dash.register_page(__name__, path='/hdbscan', name='HDBSCAN')

ORDINE_CATEGORIE = ['Labeled Set', 'Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others']

# ==========================================
# PREPARAZIONE DATI INIZIALI
# ==========================================
maschera_labeled = DF_GLOBALE['UnifiedCategory'] == 'Labeled Set'
X_labeled = EMBEDDINGS_GLOBALI[maschera_labeled]
df_labeled = DF_GLOBALE[maschera_labeled].copy()

# =========================================================
# LAYOUT
# =========================================================

layout = html.Div([
    html.Div([
        dbc.Row([
        
            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("💡 How it works: ", className="fw-bold", style={"color": "#2E4C66"}),
                        html.Span("Using UMAP, it reduces dataset complexity bringing specimens with similar features closer together while separating dissimilar ones. " \
                                "Unlike standard DBSCAN, it does not require a fixed global distance threshold and extracts the most stable and persistent clusters " \
                                "among different values of point density.", className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),
                
            ], width=12, lg=5, className="mb-3 mb-lg-0"),

            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("🎯 When to use it: ", className="fw-bold", style={"color": "#2E4C66"}),
                        html.Span("Since it eliminates the need for a fixed global Epsilon, it is the ideal choice " \
                                "when analyzing heterogeneous botanical collections, as it accurately identifies both extremely compact clusters (nearly identical specimens) and more dispersed ones (specimens exhibiting high natural variability)." \
                                , className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),

            ], width=12, lg=5, className="mb-3 mb-lg-0"),

        ], justify="center", className="text-secondary p-4", style={"fontSize": "0.95rem", "lineHeight": "1.6"})     
    ], className="mb-2"),

    html.Hr(className="mb-4"),
    
    # ---------------------------------------------------------
    # FASE 1: LABELED SET
    # ---------------------------------------------------------
    html.Div([
        html.H3([
            html.I(className="bi bi-gear-fill me-2 text-primary"), 
            "Step 1: Tool Calibration on Labeled Data"
        ], className="mb-4 fw-bold step-title"),
    ], className="d-flex align-items-center"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("UMAP Parameter", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}),                    
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    # UMAP Neighbors Slider
                    html.Div([
                        html.Label("Number of Neighbors", className="fw-bold text-primary mb-0"),
                        html.Div("Choose low values to separate closely related species or identify subspecies." \
                        " Choose high values to cluster into broader taxonomic groups.", className="text-muted small mb-3"),
                        dcc.Slider(id='uh-slider-umap-neighbors-lab', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),
                    
                    html.Hr(className="my-4"),
                    
                    html.H6("Algorithm Parameters", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}),  

                    # Min Cluster Size Slider
                    html.Div([
                        html.Label("Min Cluster Size", className="fw-bold text-primary mb-2"),
                        html.Div("Minimum size of clusters. Determines the smallest grouping to be considered a cluster.", className="text-muted small mb-3"),
                        dcc.Slider(id='uh-slider-mcs-lab', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),

                    # Min Samples Slider
                    html.Div([
                        html.Label("Min Samples", className="fw-bold text-primary mb-2"),
                        html.Div("Controls how strict the algorithm is about background noise and outliers.", className="text-muted small mb-3"),
                        dcc.Slider(id='uh-slider-ms-lab', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(5, 21, 5)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),

                    html.Div(
                        dbc.Button("Optimize ✨", id="uh-btn-opt-lab", size="sm", color="warning", outline=True, className="auto-tune-btn rounded-pill px-3 py-1 fw-bold shadow-sm"),
                        className="d-flex justify-content-center mb-2"
                    ),

                    html.Div([
                        html.Div(id='uh-metriche-box-lab'),
                    ]),
                    
                    dcc.Store(id='store-top5-uh-lab', data=[])
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
                            dcc.Graph(id='uh-grafico-3d-lab', style={'height': '380px'}, clear_on_unhover=True),
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
                                html.H6(id='uh-hover-text-lab', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(
                                    id='uh-hover-image-lab', 
                                    style={'maxWidth': '100%', 'maxHeight': '320px', 'objectFit': 'contain', 'borderRadius': '8px'}
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
                        dbc.CardBody(id='uh-tabella-crosstab-lab', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ]),

    html.Hr(className="my-5"),

    # ---------------------------------------------------------
    # FASE 2: UNLABELED SET
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
                    html.H6("UMAP Parameter", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    dbc.Button("Import Parameters", id="btn-sync-params-hdbscan", color="success", outline=True, className="sync-btn w-100 mb-2 rounded-pill shadow-sm fw-bold"),
                    html.Div("Syncs your values with the ones calibrated in Step 1", className="text-muted small text-center mb-4"),

                    # UMAP Neighbors Slider
                    html.Div([
                        html.Label("Number of Neighbors", className="fw-bold text-primary mb-2"),
                        html.Div("Choose low values to separate closely related species or identify subspecies." \
                        " Choose high values to cluster into broader taxonomic groups.", className="text-muted small mb-3"),
                        dcc.Slider(id='uh-slider-umap-neighbors-ted', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),
                    
                    html.Hr(className="my-4"),
                    
                    html.H6("Algorithm Parameters", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}),  

                    # Min Cluster Size Slider
                    html.Div([
                        html.Label("Min Cluster Size", className="fw-bold text-primary mb-2"),
                        html.Div("Minimum size of clusters for unlabeled data.", className="text-muted small mb-3"),
                        dcc.Slider(id='uh-slider-mcs-ted', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),
                    
                    # Min Samples Slider
                    html.Div([
                        html.Label("Min Samples", className="fw-bold text-primary mb-2"),
                        html.Div("Minimum points required to form a dense region (cluster).", className="text-muted small mb-3"),
                        dcc.Slider(id='uh-slider-ms-ted', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(5, 21, 5)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),

                    html.Div(
                        dbc.Button("Optimize ✨", id="uh-btn-opt-ted", size="sm", color="warning", outline=True, className="mb-2 rounded-pill px-3 py-1 shadow-sm fw-bold auto-tune-btn"),
                        className="d-flex justify-content-center"
                    ),
                    html.Div("Works also for mixed tests (Labeled and Unlabeld datasets combined)", className="text-muted small text-center mb-2"),

                    # Metrics Container
                    html.Div([
                        html.Div(id='uh-metriche-box-ted'),
                    ]),
                    
                    dcc.Store(id='store-top5-uh-ted', data=[])  
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
                        id='filter-main-uh',
                        options=[{'label': 'Select All', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in ORDINE_CATEGORIE],
                        value=['Curated'], 
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
                            dcc.Graph(id='uh-grafico-3d-ted', style={'height': '380px'}, clear_on_unhover=True),
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
                                html.H6(id='uh-hover-text-ted', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(id='uh-hover-image-ted', style={'maxWidth': '100%', 'maxHeight': '320px', 'objectFit': 'contain', 'borderRadius': '8px'})
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
                            html.H6("Distribution Matrix (Unlabeled Set)", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}),
                            className="bg-transparent border-bottom-0 pt-4 pb-0"
                        ),
                        dbc.CardBody(id='uh-tabella-crosstab-ted', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ])
], className="mb-5")

# =========================================================
# CALLBACKS: FASE 1 (Auto-Tuning)
# =========================================================

@callback(
    [Output('uh-slider-mcs-lab', 'value', allow_duplicate=True),
     Output('uh-slider-ms-lab', 'value', allow_duplicate=True),
     Output('store-top5-uh-lab', 'data')],
    Input('uh-btn-opt-lab', 'n_clicks'),
    State('uh-slider-umap-neighbors-lab', 'value'),
    prevent_initial_call=True
)
def auto_ottimizza_uh_labeled(n_clicks, umap_neighbors):
    if not n_clicks:
        raise PreventUpdate

    risultati = []
    best_score = -1
    best_mcs = 15
    best_ms = 5
    
    # Riduzione della dimensionalità
    reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_compresso = reducer.fit_transform(X_labeled)
    
    # Grid Search 
    for mcs in range(10, 45, 5):
        for ms in range(1, 15, 2):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
            labels = clusterer.fit_predict(X_compresso)
            
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            # Esclude combinazioni che generano un livello di Noise superiore al 40%
            if noise_ratio > 0.4:
                continue 
                
            fmi_base = fowlkes_mallows_score(df_labeled['Specie Predetta'], labels)
            
            score_finale = fmi_base * (1.0 - noise_ratio)
            
            risultati.append({
                'mcs': mcs, 'ms': ms, 'fmi': fmi_base, 'noise': noise_ratio, 'score': score_finale
            })
            
            if score_finale > best_score:
                best_score = score_finale
                best_mcs = mcs
                best_ms = ms
                
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    return best_mcs, best_ms, top5

@callback(
    [Output('uh-grafico-3d-lab', 'figure'), 
     Output('uh-tabella-crosstab-lab', 'children'), 
     Output('uh-metriche-box-lab', 'children')],
    [Input('uh-slider-umap-neighbors-lab', 'value'),
     Input('uh-slider-mcs-lab', 'value'), 
     Input('uh-slider-ms-lab', 'value'),
     Input('store-top5-uh-lab', 'data')]
)
def aggiorna_uh_labeled(umap_neighbors, mcs, ms, top5_data):
    reducer_clustering = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_clustering = reducer_clustering.fit_transform(X_labeled)
    
    # Estrazione dei Cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_clustering)
    
    reducer_plot = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3, n_components=3, metric='cosine', random_state=42)
    X_plot = reducer_plot.fit_transform(X_labeled)
    
    df_plot = df_labeled.copy()
    df_plot['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels]
    df_plot['x'] = X_plot[:, 0]
    df_plot['y'] = X_plot[:, 1]
    df_plot['z'] = X_plot[:, 2]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    fmi = fowlkes_mallows_score(df_plot['Specie Predetta'], labels)

    fig = genera_grafico_3d(df_plot, " ")
    
    if 'Noise' in df_plot['Cluster'].values:
        fig.update_traces(selector=dict(name="Noise"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = genera_tabella_crosstab(df_plot)
    
    elementi_box = [
        html.Hr(className="mb-4"),
        html.H6("Validation Metrics", className="fw-bold text-muted mb-3 text-uppercase", style={"letterSpacing": "1px"}),
        html.P(f"FMI: {fmi:.4f}", className="mb-1"),
        html.P(f"Noise: {noise_ratio:.1%}", className="mb-1 text-danger")
    ]
    
    if top5_data and mcs == top5_data[0]['mcs'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="mb-4"))
        elementi_box.append(html.H6("Best Values:", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"MCS: {res['mcs']} | MS: {res['ms']} | FMI: {res['fmi']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
            
    return fig, tabella, html.Div(elementi_box)

# =========================================================
# GESTIONE FILTRI E SYNC
# =========================================================

@callback(
    [Output('filter-main-uh', 'value'),
     Output('uh-slider-umap-neighbors-ted', 'value'),
     Output('uh-slider-mcs-ted', 'value'),
     Output('uh-slider-ms-ted', 'value')],
    [Input('filter-main-uh', 'value'),
     Input('btn-sync-params-hdbscan', 'n_clicks')],
    [State('uh-slider-umap-neighbors-lab', 'value'),
     State('uh-slider-mcs-lab', 'value'),
     State('uh-slider-ms-lab', 'value'),
     State('uh-slider-umap-neighbors-ted', 'value'),
     State('uh-slider-mcs-ted', 'value'),
     State('uh-slider-ms-ted', 'value')]
)
def gestisci_input_uh_ted(filtri_selezionati, n_clicks, lab_neigh, lab_mcs, lab_ms, ted_neigh, ted_mcs, ted_ms):
    triggered_id = ctx.triggered_id
    nuovi_filtri = filtri_selezionati
    
    if triggered_id == 'filter-main-uh':
        if 'ALL' in filtri_selezionati and len(filtri_selezionati) < len(ORDINE_CATEGORIE) + 1:
            nuovi_filtri = ['ALL'] + ORDINE_CATEGORIE
        elif filtri_selezionati == ['ALL']:
            nuovi_filtri = []

    if triggered_id == 'btn-sync-params-hdbscan':
        return nuovi_filtri, lab_neigh, lab_mcs, lab_ms

    if nuovi_filtri == filtri_selezionati and triggered_id == 'filter-main-uh':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    return nuovi_filtri, ted_neigh, ted_mcs, ted_ms

# =========================================================
# FASE 2: UNLABELED SET
# =========================================================
@callback(
    [Output('uh-grafico-3d-ted', 'figure'), 
     Output('uh-tabella-crosstab-ted', 'children'), 
     Output('uh-metriche-box-ted', 'children')],
    [Input('uh-slider-umap-neighbors-ted', 'value'),
     Input('uh-slider-mcs-ted', 'value'), 
     Input('uh-slider-ms-ted', 'value'),
     Input('store-top5-uh-ted', 'data'),
     Input('filter-main-uh', 'value')]
)
def aggiorna_uh_ted(umap_neighbors, mcs, ms, top5_data, categorie):
    if not categorie:
        return dash.no_update, "Nessun dato", html.Div("Seleziona almeno un filtro")

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    df_ted = DF_GLOBALE[maschera_ted].copy()
    
    if len(X_ted) < mcs:
        return dash.no_update, "Pochi dati", html.Div("Dati insufficienti.")

    reducer_clustering = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_clustering = reducer_clustering.fit_transform(X_ted)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_clustering)
    
    reducer_plot = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3, n_components=3, metric='cosine', random_state=42)
    X_plot = reducer_plot.fit_transform(X_ted)

    df_ted['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels]
    df_ted['x'] = X_plot[:, 0]
    df_ted['y'] = X_plot[:, 1]
    df_ted['z'] = X_plot[:, 2]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    fig = genera_grafico_3d(df_ted, " ")
    
    if 'Noise' in df_ted['Cluster'].values:
        fig.update_traces(selector=dict(name="Noise"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = genera_tabella_crosstab(df_ted)

    elementi_box = [
        html.Hr(className="mb-4"),
        html.H6("Validation Metrics", className="fw-bold text-muted mb-3 text-uppercase", style={"letterSpacing": "1px"}),
        html.H6(f"Number of samples: {len(X_ted)}", className="text-secondary fw-bold"),
        html.P(f"Noise: {noise_ratio:.1%}", className="text-danger fw-bold mb-1")
    ]

    if top5_data and mcs == top5_data[0]['mcs'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="mb-4"))
        elementi_box.append(html.H6("Best Values:", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"N: {res['neigh']} | MCS: {res['mcs']} | MS: {res['ms']} | K: {res['k']} | {res['metrica']}: {res['val']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )

    return fig, tabella, html.Div(elementi_box)

@callback(
    [Output('uh-slider-mcs-ted', 'value', allow_duplicate=True),
     Output('uh-slider-ms-ted', 'value', allow_duplicate=True),
     Output('store-top5-uh-ted', 'data')],
    Input('uh-btn-opt-ted', 'n_clicks'),
    [State('filter-main-uh', 'value'),
     State('uh-slider-umap-neighbors-ted', 'value')],
    prevent_initial_call=True
)
def auto_ottimizza_uh_ted_unsupervised_total(n_clicks, categorie, umap_neighbors):
    if not n_clicks or not categorie:
        raise PreventUpdate

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    df_ted = DF_GLOBALE[maschera_ted].copy()
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    
    if len(X_ted) < 10:
        raise PreventUpdate

    maschera_labeled_interna = df_ted['UnifiedCategory'] == 'Labeled Set'
    usa_ancore = maschera_labeled_interna.sum() > 0

    risultati = []
    best_score = -2.0 
    best_mcs, best_ms = 15, 5
    
    reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_compresso = reducer.fit_transform(X_ted)
    
    for mcs in range(10, 35, 5):
        for ms in range(3, 10, 2): 
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
            labels = clusterer.fit_predict(X_compresso)
            
            noise_ratio = np.sum(labels == -1) / len(labels)
            mask_validi = labels != -1
            n_clusters = len(set(labels[mask_validi])) if np.sum(mask_validi) > 0 else 0
            
            if n_clusters < 3 or n_clusters > 15 or noise_ratio > 0.4:
                continue
                
            if usa_ancore:
                labels_pred_ancore = labels[maschera_labeled_interna]
                labels_reali_ancore = df_ted[maschera_labeled_interna]['Specie Predetta']
                
                valore_metrica = fowlkes_mallows_score(labels_reali_ancore, labels_pred_ancore)
                nome_metrica = "FMI"
                
                specie_conosciute = len(labels_reali_ancore.unique())
                k_penalty = 1.0
                if n_clusters < specie_conosciute:
                    k_penalty = 0.5
                
                penalty_Noise = max(0.01, 1.0 - (noise_ratio * 1.5))
                score_finale = valore_metrica * penalty_Noise * k_penalty
                
            else:
                try:
                    valore_metrica = silhouette_score(X_compresso[mask_validi], labels[mask_validi], metric='euclidean')
                    nome_metrica = "Sil"
                    penalty_Noise = max(0.01, 1.0 - (noise_ratio * 3.0)) 
                    score_finale = valore_metrica * penalty_Noise
                except:
                    continue
                    
            risultati.append({
                'neigh': umap_neighbors, 'mcs': mcs, 'ms': ms, 'val': valore_metrica, 
                'noise': noise_ratio, 'score': score_finale, 'k': n_clusters, 'metrica': nome_metrica
            })
            
            if score_finale > best_score:
                best_score = score_finale
                best_mcs = mcs
                best_ms = ms
                
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    if best_score == -2.0 or not top5:
        return 15, 5, []
        
    return best_mcs, best_ms, top5

# =========================================================
# CALLBACK HOVER
# =========================================================

@callback(
    [Output('uh-hover-image-lab', 'src'), Output('uh-hover-text-lab', 'children')],
    Input('uh-grafico-3d-lab', 'hoverData')
)
def hover_lab(hoverData): return calcola_percorso_hover(hoverData)

@callback(
    [Output('uh-hover-image-ted', 'src'), Output('uh-hover-text-ted', 'children')],
    Input('uh-grafico-3d-ted', 'hoverData')
)
def hover_ted(hoverData): return calcola_percorso_hover(hoverData)