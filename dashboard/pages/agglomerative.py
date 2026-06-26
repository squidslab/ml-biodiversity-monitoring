import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from dash.exceptions import PreventUpdate

from utils import DATASET_CONFIG, GLOBAL_EMBEDDINGS, GLOBAL_DF, DYNAMIC_CATEGORIES, generate_3d_scatter_plot, generate_crosstab_table, get_hover_image_path

dash.register_page(__name__, path='/agglomerative', name='Agglomerative Clustering')


maschera_test = GLOBAL_DF['is_labeled_set'] == True
X_labeled = GLOBAL_EMBEDDINGS[maschera_test]
df_labeled = GLOBAL_DF[maschera_test].copy()

LINKAGE_OPTIONS = [
    {'label': 'Ward (Minimum Variance)', 'value': 'ward'},
    {'label': 'Average (Mean Distance)', 'value': 'average'},
    {'label': 'Complete (Maximum Distance)', 'value': 'complete'},
    {'label': 'Single (Minimum Distance)', 'value': 'single'}
]

layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("💡 How it works: ", className="fw-bold text-primary", style={"color": "#2E4C66"}),
                        html.Span("The algorithm aggregates specimens by recursively merging the closest pairs of clusters based on a specific linkage criterion, building a hierarchy from the bottom up until the desired number of clusters is reached.", className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),
            ], width=12, lg=5, className="mb-3 mb-lg-0"),

            dbc.Col([
                dbc.Card([
                    html.P([
                        html.Span("🎯 When to use it: ", className="fw-bold text-primary", style={"color": "#2E4C66"}),
                        html.Span("Because it merges data bottom-up to build a hierarchy, this algorithm naturally mirrors biological and categorical classifications. It is perfect for analyzing specimens that share an internal taxonomic structure, especially when you know the exact number of clusters to extract.", className="text-muted")
                    ], className="mb-0 text-center") 
                ], className="shadow-sm border-0 rounded-3 h-100 card-custom-sx p-4"),
            ], width=12, lg=5, className="mb-3 mb-lg-0"),

        ], justify="center", className="text-secondary p-4", style={"fontSize": "0.95rem", "lineHeight": "1.6"})     
    ], className="mb-2"),

    html.Hr(className="mb-5 text-muted"),
    
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
                    html.H6("Algorithm Parameters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}),                    
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    html.Div([
                        html.Label("Number of Clusters", className="fw-bold text-primary mb-0"),
                        html.Div("Must match the number of species (already calibrated for the pre-loaded dataset).", className="text-muted small mb-3"),
                        dcc.Slider(id='agg-slider-clusters', min=2, max=10, step=1, value=6, marks={i: str(i) for i in range(2, 11)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),
                    
                    html.Div([
                        html.Div([
                            html.Label("Linkage Method", className="fw-bold text-primary mb-0"),
                            dbc.Button("Optimize ✨", id="btn-opt-linkage-lab", size="sm", color="warning", outline=True, className="auto-tune-btn rounded-pill px-3 py-1 fw-bold shadow-sm")
                        ], className="d-flex justify-content-between align-items-center mb-2"),
                        html.Div("Determines the distance metric used for merging clusters.", className="text-muted small mb-3"),
                        dcc.Dropdown(
                            id='agg-dropdown-linkage',
                            options=LINKAGE_OPTIONS,
                            value='ward',
                            clearable=False,
                            className="mb-4 mt-2"
                        ),
                    ], className="custom-slider-box mb-4"),

                    html.Div([
                        html.Div(id='agg-metriche-box', className="mt-3"),
                    ]),
                    
                    dcc.Store(id='store-top5-labeled-agg', data=[])  
                ], className="p-4") 
            ], className="shadow-sm border-0 h-100 rounded-3")
        ], width=12, lg=3, className="mb-4 mb-lg-0"), 
        
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Latent Space Visualization", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}),
                            dcc.Graph(id='agg-grafico-3d', style={'height': '380px'}, clear_on_unhover=True),
                        ], className="p-2"), 
                        style={'height': '460px'},
                        className="shadow-sm border-0 rounded-3 mb-4 mb-xl-0"
                    ),
                    width=12, xl=8
                ),
                
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Specimen Preview", className="fw-bold text-muted mb-3 text-uppercase text-center", style={"letterSpacing": "1px"}),
                            html.Div([
                                html.H6(id='agg-hover-text', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(
                                    id='agg-hover-image', 
                                    style={'maxWidth': '100%', 'maxHeight': '320px', 'objectFit': 'contain', 'borderRadius': '8px'}
                                )
                            ], className="d-flex flex-column align-items-center justify-content-center h-100") 
                        ], className="p-4") 
                    ], style={'height': '460px'}, className="shadow-sm border-0 rounded-3"), 
                    width=12, xl=4
                )
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.H6("Distribution Matrix (Labeled Set)", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}), 
                            className="bg-transparent border-bottom-0 pt-4 pb-0"
                        ),
                        dbc.CardBody(id='agg-tabella-crosstab', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ]),

    html.Hr(className="my-5"),

    html.Div([
        html.H3([
            html.I(className="bi bi-robot me-2 text-primary"), 
            "Step 2: Analyzing Unlabeled Data"
        ], className="mb-4 fw-bold step-title"),
    ], className="d-flex align-items-center mt-5"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H6("Algorithm Parameters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    dbc.Button("Import Parameters", id="btn-sync-params-agg", color="success", outline=True, className="sync-btn w-100 mb-2 rounded-pill shadow-sm fw-bold"),
                    html.Div("Syncs your values with the ones calibrated in Step 1", className="text-muted small text-center mb-4"),
                    
                    html.Div([
                        html.Div([
                            html.Label("Number of Clusters", className="fw-bold text-primary mb-0"),
                            dbc.Button("Optimize ✨", id="btn-opt-clusters-ted-agg", size="sm", color="warning", outline=True, className="auto-tune-btn rounded-pill px-3 py-1 fw-bold shadow-sm")
                        ], className="d-flex justify-content-between align-items-center mb-2"),
                        html.Div("Adjust to match expected species diversity.", className="text-muted small mb-3"),
                        dcc.Slider(id='ted-slider-clusters-agg', min=2, max=10, step=1, value=4, marks={i: str(i) for i in range(2, 11)}, className="mb-4 force-blue-slider", tooltip={"always_visible": False}),
                    ], className="custom-slider-box mb-4"),
                    
                    html.Div([
                        html.Label("Linkage Method", className="fw-bold text-primary mb-2"),
                        dcc.Dropdown(
                            id='ted-dropdown-linkage',
                            options=LINKAGE_OPTIONS,
                            value='ward',
                            clearable=False,
                            className="mb-4 mt-2"
                        ),
                    ], className="custom-slider-box"),

                    html.Div([
                        html.Div(id='ted-metriche-box-agg', className="mt-3"),
                    ]),
                ], className="p-4")
            ], className="shadow-sm border-0 rounded-3 mb-4"),

            dbc.Card([
                dbc.CardHeader(
                    html.H6("Dataset Filters", className="mb-0 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}), 
                    className="bg-transparent border-bottom-0 pt-4 pb-0"
                ),
                dbc.CardBody([
                    dbc.Checklist(
                        id='filter-main-agg',
                        options=[{'label': 'Select All', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in DYNAMIC_CATEGORIES],
                        value=[DYNAMIC_CATEGORIES[1]] if len(DYNAMIC_CATEGORIES) > 1 else (DYNAMIC_CATEGORIES[:1] if DYNAMIC_CATEGORIES else []),
                        className="mb-2"
                    )
                ], className="p-4")
            ], className="shadow-sm border-0 rounded-3")
        ], width=12, lg=3, className="mb-4 mb-lg-0"),

        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6("Latent Space Visualization", className="fw-bold text-muted mb-3 px-3 pt-2 text-uppercase", style={"letterSpacing": "1px"}),
                            dcc.Graph(id='ted-grafico-3d-agg', style={'height': '380px'}, clear_on_unhover=True),
                        ], className="p-2"), 
                        style={'height': '460px'},
                        className="shadow-sm border-0 rounded-3 mb-4"
                    ),
                    width=12, xl=8
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Specimen Preview", className="fw-bold text-muted mb-3 text-uppercase text-center", style={"letterSpacing": "1px"}),
                            html.Div([
                                html.H6(id='ted-hover-text-agg', className="text-center text-secondary mb-2", style={'fontSize': '12px', 'minHeight': '18px'}),
                                html.Img(id='ted-hover-image-agg', style={'maxWidth': '100%', 'maxHeight': '320px', 'objectFit': 'contain', 'borderRadius': '8px'})
                            ], className="d-flex flex-column align-items-center justify-content-center h-100")
                        ], className="p-4")
                    ], style={'height': '460px'}, className="shadow-sm border-0 rounded-3"),
                    width=12, xl=4
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(
                            html.Div([
                                html.H6("Distribution Matrix (Unlabeled Set)", className="fw-bold text-muted mb-0 text-uppercase", style={"letterSpacing": "1px"}),
                                html.Div([
                                    dcc.Download(id="btn-download-excel-target-agg"),
                                    dbc.Button(
                                        "📥 Download Selected Clusters (Excel)",
                                        id="btn-download-excel-agg",
                                        color="success",
                                        outline=True,
                                        className="fw-bold sync-btn rounded-pill shadow-sm"
                                    )
                                ])
                            ], className="d-flex justify-content-between align-items-center px-3"),
                            className="bg-transparent border-bottom-0 pt-4 pb-0"
                        ),
                        dbc.CardBody(id='ted-tabella-crosstab-agg', className="p-4")
                    ], className="shadow-sm border-0 rounded-3")
                )
            ])
        ], width=12, lg=9)
    ])
], className="mb-5")

@callback(
    [Output('agg-grafico-3d', 'figure'), 
     Output('agg-tabella-crosstab', 'children'), 
     Output('agg-metriche-box', 'children')],
    [Input('agg-slider-clusters', 'value'), 
     Input('agg-dropdown-linkage', 'value'), 
     Input('store-top5-labeled-agg', 'data')]
)
def aggiorna_agg_labeled(n_clusters, linkage, top5_data):
    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    metrica = 'euclidean' if linkage == 'ward' else 'cosine'
    
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric=metrica, linkage=linkage)
    labels = clusterer.fit_predict(X_labeled)
    
    df_plot = df_labeled.copy()
    df_plot['Cluster'] = labels.astype(str)
    
    fig = generate_3d_scatter_plot(df_plot, title=" ")
    tabella = generate_crosstab_table(df_plot) 
    
    ami = adjusted_mutual_info_score(df_plot[colonna_target], labels)
    ari = adjusted_rand_score(df_plot[colonna_target], labels)
    
    elementi_box = [
        html.Hr(className="mb-4"),
        html.H6("Validation Metrics", className="fw-bold text-muted mb-3 text-uppercase", style={"letterSpacing": "1px"}),
        html.P(f"AMI Score: {ami:.4f}", className="mb-1"),
        html.P(f"ARI Score: {ari:.4f}", className="mb-1")
    ]
    
    if top5_data and linkage == top5_data[0]['linkage'] and n_clusters == top5_data[0].get('k_clusters'):
        elementi_box.append(html.Hr(className="mb-4"))
        elementi_box.append(html.H6("Best Values:", className="mb-2 fw-bold text-uppercase text-muted", style={"letterSpacing": "1px"}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Linkage: {res['linkage']} | AMI: {res['ami']:.3f} | ARI: {res['ari']:.3f}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
    
    box = html.Div(elementi_box)
    
    return fig, tabella, box

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

    colonna_target = DATASET_CONFIG['PREDICTION_COL']
    risultati = []
    best_score = -1
    best_linkage = 'ward'
    
    combinazioni = [
        ('ward', 'euclidean'),
        ('average', 'cosine'), ('average', 'euclidean'),
        ('complete', 'cosine'), ('complete', 'euclidean'),
        ('single', 'cosine')
    ]
    
    for link, metrica in combinazioni:
        labels = AgglomerativeClustering(n_clusters=k_clusters, metric=metrica, linkage=link).fit_predict(X_labeled)
        
        ami = adjusted_mutual_info_score(df_labeled[colonna_target], labels)
        ari = adjusted_rand_score(df_labeled[colonna_target], labels)
        score = (ami + ari) / 2.0
        
        risultati.append({'linkage': link, 'metrica': metrica, 'ami': ami, 'ari': ari, 'score': score, 'k_clusters': k_clusters}) 
        
        if score > best_score:
            best_score = score
            best_linkage = link
            
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    return best_linkage, top5

@callback(
    [Output('agg-hover-image', 'src'),
     Output('agg-hover-text', 'children')],
    Input('agg-grafico-3d', 'hoverData')
)
def aggiorna_hover_agg_labeled(hoverData):
    return get_hover_image_path(hoverData)
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

    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie

    maschera_ted = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    df_ted = GLOBAL_DF[maschera_ted].copy()
    
    if len(X_ted) < n_clusters:
        return px.scatter_3d(title=f"Dati insufficienti ({len(X_ted)} immagini)"), "Nessun dato", html.Div("Pochi dati")
        
    metrica = 'euclidean' if linkage == 'ward' else 'cosine'
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric=metrica, linkage=linkage).fit_predict(X_ted)
    df_ted['Cluster'] = labels.astype(str)
    
    fig = generate_3d_scatter_plot(df_ted, title=" ")
    tabella = generate_crosstab_table(df_ted)
    
    try:
        score = silhouette_score(X_ted, labels, metric=metrica)
    except:
        score = 0

    metriche = html.Div([
        html.H6(f"Number of samples: {len(X_ted)}", className="text-secondary fw-bold"),
        html.P(f"Silhouette Score: {score:.4f}", className="text-success fw-bold")
    ])
    
    return fig, tabella, metriche

@callback(
    [Output('ted-hover-image-agg', 'src'),
     Output('ted-hover-text-agg', 'children')],
    Input('ted-grafico-3d-agg', 'hoverData')
)
def mostra_immagine_hover_agg(hoverData):
    return get_hover_image_path(hoverData)

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
            nuovi_filtri = ['ALL'] + DYNAMIC_CATEGORIES
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    if triggered_id == 'btn-sync-params-agg':
        return nuovi_filtri, lab_k, lab_link

    return nuovi_filtri, ted_k, ted_link

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

    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie

    # Nota: la tua maschera originale manteneva anche il controllo su 'is_labeled_set'.
    maschera_ted = (~GLOBAL_DF['is_labeled_set']) & (GLOBAL_DF['UnifiedCategory'].isin(categorie_attive))
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    
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

@callback(
    Output("btn-download-excel-target-agg", "data"),
    Input("btn-download-excel-agg", "n_clicks"),
    [State('ted-slider-clusters-agg', 'value'),
     State('ted-dropdown-linkage', 'value'),
     State('filter-main-agg', 'value')],
    prevent_initial_call=True
)
def download_excel_agg(n_clicks, n_clusters, linkage, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    if 'ALL' in categorie:
        categorie_attive = DYNAMIC_CATEGORIES
    else:
        categorie_attive = categorie

    maschera_ted = GLOBAL_DF['UnifiedCategory'].isin(categorie_attive)
    X_ted = GLOBAL_EMBEDDINGS[maschera_ted]
    df_ted = GLOBAL_DF[maschera_ted].copy()
    
    if len(X_ted) < n_clusters:
        raise PreventUpdate

    metrica = 'euclidean' if linkage == 'ward' else 'cosine'
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric=metrica, linkage=linkage).fit_predict(X_ted)
    df_ted['Cluster'] = labels.astype(str)
    
    colonna_id = DATASET_CONFIG['IMAGE_ID_COL']
    df_download = df_ted[[colonna_id, 'Cluster']].copy()
    df_download = df_download.sort_values(by='Cluster')

    return dcc.send_data_frame(df_download.to_excel, "selected_clusters_agglomerative.xlsx", index=False)