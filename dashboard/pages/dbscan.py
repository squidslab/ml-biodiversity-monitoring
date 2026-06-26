import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, no_update
from dash.exceptions import PreventUpdate
import dash
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import fowlkes_mallows_score, silhouette_score
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, adjusted_mutual_info_score

from utils import DATASET_CONFIG, GLOBAL_EMBEDDINGS, GLOBAL_DF, DYNAMIC_CATEGORIES, generate_3d_scatter_plot, generate_crosstab_table, get_hover_image_path

dash.register_page(__name__, path='/dbscan', name='DBSCAN')

ORDINE_CATEGORIE = ['Labeled Set', 'Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others']

# ==========================================
# PREPARAZIONE DATI INIZIALI
# ==========================================
maschera_labeled = GLOBAL_DF['UnifiedCategory'] == 'Labeled Set'
X_labeled = GLOBAL_EMBEDDINGS[maschera_labeled]
df_labeled = GLOBAL_DF[maschera_labeled].copy()

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
                        html.Span("The algorithm aggregates specimens by finding areas in space with a high concentration of points, " \
                                "separated by regions of low density (classified as noise).", className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),
                
            ], width=12, lg=5, className="mb-3 mb-lg-0"),

            # SECONDA COLONNA
            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("🎯 When to use it: ", className="fw-bold text-primary", style={"color": "#2E4C66"}),
                        html.Span("Because it relies on a fixed global distance threshold (Epsilon), it successfully identifies clusters of the same density, " \
                                "but struggles if the dataset simultaneously contains both highly homogeneous species (tight clusters) and species " \
                                "with high natural variance (spread-out clusters).", className="text-muted")
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
                    # Epsilon Slider
                    html.Div([
                        html.Label("Epsilon", className="fw-bold text-primary mb-0"),
                        html.Div("Defines the maximum distance between points to be considered neighbors.", className="text-muted small mb-3"),
                        dcc.Slider(id='dbscan-slider-eps-lab', min=0.01, max=0.5, step=0.01, value=0.3, marks={round(i*0.1, 1): str(round(i*0.1, 1)) for i in range(1, 6)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),
                    
                    # Min Samples Slider
                    html.Div([
                        html.Label("Min Samples", className="fw-bold text-primary mb-2"),
                        html.Div("Minimum points required to form a dense region (cluster).", className="text-muted small mb-3"),
                        dcc.Slider(id='dbscan-slider-min-samples-lab', min=2, max=30, step=1, value=15, marks={i: str(i) for i in range(5, 31, 5)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),

                    html.Div(
                        dbc.Button("Optimize ✨", id="btn-opt-dbscan-lab", size="sm", color="warning", outline=True, className="auto-tune-btn rounded-pill px-3 py-1 fw-bold shadow-sm"),
                        className="d-flex justify-content-center mb-2"
                    ),
                    # Metrics Container
                    html.Div([
                        html.Div(id='dbscan-metriche-box-lab'),
                    ]),
                    
                    dcc.Store(id='store-top5-dbscan-lab', data=[])  
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
                            dcc.Graph(id='dbscan-grafico-3d-lab', style={'height': '380px'}, clear_on_unhover=True),
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
                                html.H6(id='dbscan-hover-text-lab', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(
                                    id='dbscan-hover-image-lab', 
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
                        dbc.CardBody(id='dbscan-tabella-crosstab-lab', className="p-4")
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
                    html.H6("Algorithm Parameters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    dbc.Button("Import Parameters", id="btn-sync-params-dbscan", color="success", outline=True, className="sync-btn w-100 mb-2 rounded-pill shadow-sm fw-bold"),
                    html.Div("Syncs your values with the ones calibrated in Step 1", className="text-muted small text-center mb-4"),
                    
                    # Epsilon Slider
                    html.Div([
                        html.Label("Epsilon", className="fw-bold text-primary mb-2"),
                        dcc.Slider(id='dbscan-slider-eps-ted', min=0.01, max=0.5, step=0.01, value=0.3, marks={round(i*0.1, 1): str(round(i*0.1, 1)) for i in range(1, 6)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box"),
                    
                    # Min Samples Slider
                    html.Div([
                        html.Label("Min Samples", className="fw-bold text-primary mb-2"),
                        dcc.Slider(id='dbscan-slider-min-samples-ted', min=2, max=30, step=1, value=15, marks={i: str(i) for i in range(5, 31, 5)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box"),

                    html.Div(
                        dbc.Button("Optimize Mixed Test ✨", id="btn-opt-dbscan-ted", size="sm", color="warning", outline=True, className="mb-2 rounded-pill shadow-sm px-3 py-1 fw-bold auto-tune-btn"),
                        className="d-flex justify-content-center"
                    ),
                    html.Div("Only works for mixed tests (Labeled and Unlabeld datasets combined)", className="text-muted small text-center mb-2"),
                    
                    # Metrics Container
                    html.Div([
                        html.Div(id='dbscan-metriche-box-ted'),
                    ]),
                    
                    dcc.Store(id='store-top5-dbscan-ted', data=[])  
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
                        id='filter-main-dbscan',
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
                            dcc.Graph(id='dbscan-grafico-3d-ted', style={'height': '380px'}, clear_on_unhover=True),
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
                                html.H6(id='dbscan-hover-text-ted', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(id='dbscan-hover-image-ted', style={'maxWidth': '100%', 'maxHeight': '320px', 'objectFit': 'contain', 'borderRadius': '8px'})
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
                                    dcc.Download(id="btn-download-excel-target-dbscan"),
                                    dbc.Button(
                                        "📥 Download Clusters Distribution",
                                        id="btn-download-excel-dbscan",
                                        color="success",
                                        outline=True,
                                        className="fw-bold sync-btn rounded-pill shadow-sm"
                                    )
                                ])
                            ], className="d-flex justify-content-between align-items-center px-3"),
                            className="bg-transparent border-bottom-0 pt-4 pb-0"
                        ),
                        dbc.CardBody(id='dbscan-tabella-crosstab-ted', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ])
], className="mb-5")
@callback(
    [Output('dbscan-slider-eps-lab', 'value', allow_duplicate=True),
     Output('dbscan-slider-min-samples-lab', 'value', allow_duplicate=True),
     Output('store-top5-dbscan-lab', 'data')],
    Input('btn-opt-dbscan-lab', 'n_clicks'),
    prevent_initial_call=True
)
def auto_ottimizza_dbscan_labeled(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    risultati = []
    best_score = -1
    best_eps = 0.3
    best_ms = 5
    
    classi_uniche = df_labeled[colonna_target].unique()
    
    for eps in np.arange(0.1, 1.05, 0.05):
        for ms in range(5, 31, 2):
            clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
            labels = clusterer.fit_predict(X_labeled)
            
            noise_ratio = np.sum(labels == -1) / len(labels)
            if noise_ratio > 0.4:
                continue 
                
            scores_simulati = []
            
            for classe_da_ignorare in classi_uniche:
                maschera_loco = df_labeled[colonna_target] != classe_da_ignorare
                labels_reali = df_labeled[maschera_loco][colonna_target]
                labels_pred = labels[maschera_loco]
                
                fmi = fowlkes_mallows_score(labels_reali, labels_pred)
                scores_simulati.append(fmi)
                
            mean_fmi = np.mean(scores_simulati)
            score_finale = mean_fmi * (1.0 - noise_ratio)
            fmi_base = fowlkes_mallows_score(df_labeled[colonna_target], labels)
            
            risultati.append({
                'eps': round(eps, 2), 'ms': ms, 'fmi': fmi_base, 
                'noise': noise_ratio, 'score': score_finale
            })
            
            if score_finale > best_score:
                best_score = score_finale
                best_eps = round(eps, 2)
                best_ms = ms
                
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    return best_eps, best_ms, top5


@callback(
    [Output('dbscan-grafico-3d-lab', 'figure'), 
     Output('dbscan-tabella-crosstab-lab', 'children'), 
     Output('dbscan-metriche-box-lab', 'children')],
    [Input('dbscan-slider-eps-lab', 'value'), 
     Input('dbscan-slider-min-samples-lab', 'value'),
     Input('store-top5-dbscan-lab', 'data')]
)
def aggiorna_dbscan_labeled(eps, ms, top5_data):
    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
    labels = clusterer.fit_predict(X_labeled)
    
    df_plot = df_labeled.copy()
    
    df_plot['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    fmi = fowlkes_mallows_score(df_plot[colonna_target], labels)
    ami = adjusted_mutual_info_score(df_plot[colonna_target], labels)
    ari = adjusted_rand_score(df_plot[colonna_target], labels)      
    # ------------------------
 
    if df_plot.empty:
        return dash.no_update, "No data (All of it is classified as Noise)", html.Div("Tutte le foto sono state etichettate come Noise.")

    fig = generate_3d_scatter_plot(df_plot, title=" ")
    
    if 'Noise' in df_plot['Cluster'].values:
        fig.update_traces(selector=dict(name="Noise"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = generate_crosstab_table(df_plot)
    
    elementi_box = [
        html.Hr(className="mb-4"),
        html.H6("Validation Metrics", className="fw-bold text-muted mb-3 text-uppercase", style={"letterSpacing": "1px"}),
        html.P(f"FMI: {fmi:.4f}", className="mb-1"),
        html.P(f"AMI: {ami:.4f}", className="mb-1"), 
        html.P(f"ARI: {ari:.4f}", className="mb-1"), 
        html.P(f"Noise: {noise_ratio:.1%}", className="mb-1 text-danger")
    ]
    
    if top5_data and eps == top5_data[0]['eps'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="mb-4"))
        elementi_box.append(html.H6("Best Values:", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Eps: {res['eps']} | MS: {res['ms']} | FMI: {res['fmi']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
            
    return fig, tabella, html.Div(elementi_box)

@callback(
    [Output('filter-main-dbscan', 'value'),
     Output('dbscan-slider-eps-ted', 'value'),
     Output('dbscan-slider-min-samples-ted', 'value')],
    [Input('filter-main-dbscan', 'value'),
     Input('btn-sync-params-dbscan', 'n_clicks')],
    [State('dbscan-slider-eps-lab', 'value'),
     State('dbscan-slider-min-samples-lab', 'value'),
     State('dbscan-slider-eps-ted', 'value'),
     State('dbscan-slider-min-samples-ted', 'value')]
)
def gestisci_input_dbscan_ted(filtri_selezionati, n_clicks, lab_eps, lab_ms, ted_eps, ted_ms):
    triggered_id = ctx.triggered_id
    nuovi_filtri = filtri_selezionati
    
    if triggered_id == 'filter-main-dbscan':
        if 'ALL' in filtri_selezionati:
            nuovi_filtri = ['ALL'] + DYNAMIC_CATEGORIES
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    if triggered_id == 'btn-sync-params-dbscan':
        return nuovi_filtri, lab_eps, lab_ms

    return nuovi_filtri, ted_eps, ted_ms


@callback(
    [Output('dbscan-grafico-3d-ted', 'figure'), 
     Output('dbscan-tabella-crosstab-ted', 'children'), 
     Output('dbscan-metriche-box-ted', 'children')],
    [Input('dbscan-slider-eps-ted', 'value'), 
     Input('dbscan-slider-min-samples-ted', 'value'),
     Input('filter-main-dbscan', 'value'),
     Input('store-top5-dbscan-ted', 'data')]
)
def aggiorna_dbscan_ted(eps, ms, categorie, top5_data):
    if not categorie:
        return dash.no_update, "No data", html.Div("Select at least one filter")

    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie

    maschera_ted = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    df_ted = GLOBAL_DF[maschera_ted].copy()
    
    if len(X_ted) < ms:
        return dash.no_update, "Not enough data", html.Div("Dati insufficienti.")

    clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
    labels = clusterer.fit_predict(X_ted)
    
    df_ted['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    mask_no_noise = labels != -1
    try:
        if num_clusters > 1:
            sil_score = silhouette_score(X_ted[mask_no_noise], labels[mask_no_noise], metric='euclidean')
            sil_text = f"{sil_score:.3f}"
        else:
            sil_text = "N/A with just 1 cluster"
    except Exception:
        sil_text = "Errore Calcolo"

    fig = generate_3d_scatter_plot(df_ted, title=" ")
    
    if 'Noise' in df_ted['Cluster'].values:
        fig.update_traces(selector=dict(name="Noise"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = generate_crosstab_table(df_ted)

    elementi_box = [
        html.Hr(className="mb-4"),
        html.H6("Validation Metrics", className="fw-bold text-muted mb-3 text-uppercase", style={"letterSpacing": "1px"}),
        html.H6(f"Number of samples: {len(X_ted)}", className="text-secondary fw-bold"),
        html.P(f"Noise: {noise_ratio:.1%}", className="text-danger fw-bold mb-1"),
        html.P(f"Silhouette Score: {sil_text}", className="text-success fw-bold")
    ]
    
    if top5_data and eps == top5_data[0]['eps'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("Best Values Based on Labeled:", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Eps: {res['eps']} | MS: {res['ms']} | Score: {res['score']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
    
    return fig, tabella, html.Div(elementi_box)


@callback(
    [Output('dbscan-hover-image-lab', 'src'), Output('dbscan-hover-text-lab', 'children')],
    Input('dbscan-grafico-3d-lab', 'hoverData')
)
def hover_lab(hoverData): return get_hover_image_path(hoverData)


@callback(
    [Output('dbscan-hover-image-ted', 'src'), Output('dbscan-hover-text-ted', 'children')],
    Input('dbscan-grafico-3d-ted', 'hoverData')
)
def hover_ted(hoverData): return get_hover_image_path(hoverData)


@callback(
    [Output('dbscan-slider-eps-ted', 'value', allow_duplicate=True),
     Output('dbscan-slider-min-samples-ted', 'value', allow_duplicate=True),
     Output('store-top5-dbscan-ted', 'data')], 
    Input('btn-opt-dbscan-ted', 'n_clicks'),  
    State('filter-main-dbscan', 'value'),     
    prevent_initial_call=True
)
def auto_ottimizza_dbscan_misto(n_clicks, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    
    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie
        
    maschera_totale = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    
    df_subset = GLOBAL_DF[maschera_totale].copy()
    X_subset = GLOBAL_EMBEDDINGS[maschera_totale]

    maschera_labeled_interna = df_subset['UnifiedCategory'] == 'Labeled Set'
    if not maschera_labeled_interna.any():
        return dash.no_update, dash.no_update, dash.no_update

    labels_reali_labeled = df_subset[maschera_labeled_interna][colonna_target]

    risultati = []
    best_score = -1
    best_eps = 0.3
    best_ms = 5
    
    for eps in np.arange(0.1, 1.05, 0.05):
        for ms in range(5, 31, 2):
            clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
            labels_totali = clusterer.fit_predict(X_subset)
            
            noise_ratio_totale = np.sum(labels_totali == -1) / len(labels_totali)
            
            labels_pred_labeled = labels_totali[maschera_labeled_interna]
            
            ari = adjusted_rand_score(labels_reali_labeled, labels_pred_labeled)
            ami = adjusted_mutual_info_score(labels_reali_labeled, labels_pred_labeled)
            fmi = fowlkes_mallows_score(labels_reali_labeled, labels_pred_labeled)
            
            media_metriche = (ari + ami + fmi) / 3.0
            score_finale = media_metriche * (1.0)
            
            risultati.append({
                'eps': round(eps, 2), 
                'ms': ms, 
                'ari': ari, 
                'ami': ami, 
                'fmi': fmi, 
                'noise': noise_ratio_totale, 
                'score': score_finale
            })
            
            if score_finale > best_score:
                best_score = score_finale
                best_eps = round(eps, 2)
                best_ms = ms
                
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    return best_eps, best_ms, top5

# ==========================================
# DOWNLOAD
# ==========================================

@callback(
    Output("btn-download-excel-target-dbscan", "data"),
    Input("btn-download-excel-dbscan", "n_clicks"),
    [State('dbscan-slider-eps-ted', 'value'), 
     State('dbscan-slider-min-samples-ted', 'value'),
     State('filter-main-dbscan', 'value')],
    prevent_initial_call=True
)
def download_excel_dbscan(n_clicks, eps, ms, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie
        
    maschera_ted = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    df_ted = GLOBAL_DF[maschera_ted].copy()
    
    if len(X_ted) < ms:
        raise PreventUpdate

    clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
    labels = clusterer.fit_predict(X_ted)
    
    df_ted['Cluster'] = [str(l) if l != -1 else 'Noise' for l in labels]
    
    colonna_id = DATASET_CONFIG['IMAGE_ID_COL']
    df_download = df_ted[[colonna_id, 'Cluster']].copy()
    df_download = df_download.sort_values(by='Cluster')

    return dcc.send_data_frame(df_download.to_excel, "selected_clusters_dbscan.xlsx", index=False)