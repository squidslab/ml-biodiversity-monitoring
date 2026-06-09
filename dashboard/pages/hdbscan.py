import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np
import hdbscan
import umap.umap_ as umap
from sklearn.metrics import fowlkes_mallows_score, silhouette_score

# Assicurati di importare le variabili e le funzioni dal tuo file utils
from utils import (
    DF_GLOBALE, EMBEDDINGS_GLOBALI, 
    genera_grafico_3d, genera_tabella_crosstab, calcola_percorso_hover
)

# Creiamo i dataset Labeled filtrando i dati globali all'avvio della pagina
maschera_labeled = DF_GLOBALE['UnifiedCategory'] == 'Labeled Set'
X_labeled = EMBEDDINGS_GLOBALI[maschera_labeled]
df_labeled = DF_GLOBALE[maschera_labeled].copy()

dash.register_page(__name__, path='/hdbscan', name='UMAP + HDBSCAN')

ORDINE_CATEGORIE = ['Labeled Set', 'Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others']

# =========================================================
# LAYOUT
# =========================================================

layout = html.Div([
    html.H2("UMAP + HDBSCAN Clustering", className="mb-3", style={"color": "#430783"}),
    html.Hr(),
    
    # ---------------------------------------------------------
    # FASE 1: LABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 1: Topologia e Densità su Labeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚙️ Parametri UMAP & HDBSCAN", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.H6("1. Compressione UMAP", className="fw-bold text-primary mb-2"),
                    html.Label("N° Vicini (Struttura Globale):", className="fw-bold"),
                    html.Div(
                        "Valori bassi = focus sui micro-dettagli. Valori alti = focus sulle grandi galassie.", 
                        className="text-muted mb-2", style={"fontSize": "12px"}
                    ),
                    dcc.Slider(id='uh-slider-umap-neighbors-lab', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Hr(),
                    
                    html.Div([
                        html.H6("2. Clustering HDBSCAN", className="fw-bold text-success mb-0"),
                        dbc.Button("✨ Auto Tune", id="uh-btn-opt-lab", size="sm", color="warning", outline=True)
                    ], className="d-flex justify-content-between align-items-center mb-2"),
                    
                    html.Label("Min Cluster Size:", className="fw-bold mt-2"),
                    dcc.Slider(id='uh-slider-mcs-lab', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Min Samples:", className="fw-bold mt-2"),
                    dcc.Slider(id='uh-slider-ms-lab', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(5, 21, 5)}, className="mb-4", tooltip={"always_visible": False}),

                    dbc.Checklist(
                        id="uh-toggle-noise-lab",
                        options=[{"label": " Mostra Rumore", "value": True}],
                        value=[True],
                        switch=True,
                        className="fw-bold text-secondary mt-2 mb-3"
                    ),

                    html.Hr(),
                    html.Div(id='uh-metriche-box-lab', className="mt-3"),
                    dcc.Store(id='store-top5-uh-lab', data=[])
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='uh-grafico-3d-lab', style={'height': '420px'}, clear_on_unhover=True),
                            className="p-0" 
                        ), style={'height': '420px'}, className="shadow-sm border-0"
                    ), width=9
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Preview", className="text-center fw-bold", style={"backgroundColor": "#C499F9", "fontSize": "12px", "height": "40px", "padding": "8px"}),
                        dbc.CardBody([
                            html.H6(id='uh-hover-text-lab', className="text-center text-secondary mb-2", style={'fontSize': '11px', 'height': '15px'}),
                            html.Img(
                                id='uh-hover-image-lab', 
                                style={'maxWidth': '100%', 'maxHeight': '330px', 'objectFit': 'contain', 'borderRadius': '5px'}
                            )
                        ], className="d-flex flex-column align-items-center justify-content-center", style={'height': '380px'}) 
                    ], style={'height': '420px'}, className="shadow-sm border-0"), width=3
                )
            ], className="border-0 mb-3 align-items-stretch"),
            
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Matrice di Distribuzione (Labeled Set)", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                        dbc.CardBody(id='uh-tabella-crosstab-lab')
                    ], className="shadow-sm border-0")
                )
            ])
        ], width=9)
    ]),

    html.Hr(className="my-5"),

    # ---------------------------------------------------------
    # FASE 2: UNLABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 2: Applicazione parametri su Unlabeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚙️ Parametri UMAP & HDBSCAN", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Div([
                        html.H6("2. Clustering HDBSCAN (Auto)", className="fw-bold text-success mb-0"),
                        dbc.Button("✨ Auto Tune Unsupervised", id="uh-btn-opt-ted", size="sm", color="danger", outline=True)
                    ], className="d-flex justify-content-between align-items-center mb-2"),

                    html.H6("1. Compressione UMAP", className="fw-bold text-primary mb-2"),
                    html.Label("N° Vicini:", className="fw-bold"),
                    dcc.Slider(id='uh-slider-umap-neighbors-ted', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Hr(),
                    
                    html.H6("2. Clustering HDBSCAN", className="fw-bold text-success mb-2"),
                    html.Label("Min Cluster Size:", className="fw-bold mt-2"),
                    dcc.Slider(id='uh-slider-mcs-ted', min=5, max=50, step=5, value=15, marks={i: str(i) for i in range(10, 51, 10)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Min Samples:", className="fw-bold mt-2"),
                    dcc.Slider(id='uh-slider-ms-ted', min=1, max=20, step=1, value=5, marks={i: str(i) for i in range(5, 21, 5)}, className="mb-4", tooltip={"always_visible": False}),

                    dbc.Checklist(
                        id="uh-toggle-noise-ted",
                        options=[{"label": " Mostra Rumore", "value": True}],
                        value=[True],
                        switch=True,
                        className="fw-bold text-secondary mt-2 mb-3"
                    ),

                    html.Hr(),
                    html.Div(id='uh-metriche-box-ted', className="mt-3"),
                    dcc.Store(id='store-top5-uh-ted', data=[])
                ])
            ], className="shadow-sm border-0 h-auto mb-3"),
            dbc.Card([
                dbc.CardHeader("🔍 Filtri Dataset", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Div([
                        dbc.Checklist(
                            id='filter-main-uh',
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
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='uh-grafico-3d-ted', style={'height': '420px'}, clear_on_unhover=True),
                            className="p-0"
                        ), style={'height': '420px'}, className="shadow-sm border-0"
                    ), width=9
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Preview", className="text-center fw-bold", style={"backgroundColor": "#C499F9", "fontSize": "12px", "height": "40px", "padding": "8px"}),
                        dbc.CardBody([
                            html.H6(id='uh-hover-text-ted', className="text-center text-secondary mb-2", style={'fontSize': '11px'}),
                            html.Img(
                                id='uh-hover-image-ted', 
                                style={'maxWidth': '100%', 'maxHeight': '330px', 'objectFit': 'contain', 'borderRadius': '5px'}
                            )
                        ], className="d-flex flex-column align-items-center justify-content-center", style={'height': '380px'}) 
                    ], style={'height': '420px'}, className="shadow-sm border-0"), width=3
                )
            ], className="border-0 mb-3 align-items-stretch"),

            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Matrice di Distribuzione (Unlabeled Set)", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                    dbc.CardBody(id='uh-tabella-crosstab-ted')
                ], className="shadow-sm border-0"))
            ])
        ], width=9)
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
    
    classi_uniche = df_labeled['Specie Predetta'].unique()
    
    # 🌟 CORREZIONE 1: Riduciamo a 15 Dimensioni per il clustering (mantiene la topologia)
    # 🌟 CORREZIONE 2: min_dist alzato a 0.25 (o 0.3) per evitare che i punti si impacchino
    # troppo, creando falsi bordi che HDBSCAN scambierebbe per divisioni tra cluster.
    reducer = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_compresso = reducer.fit_transform(X_labeled)
    
    # Grid Search HDBSCAN sul nuovo spazio a 15D
    for mcs in range(10, 45, 5):
        for ms in range(1, 15, 2):
            # 🌟 CORREZIONE 3: cluster_selection_method='eom' (Excess of Mass)
            # Sostituisce 'leaf'. L'algoritmo 'eom' è essenziale per estrarre macro-cluster
            # ed evitare l'over-segmentazione (i 16 cluster che avevi prima).
            clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
            labels = clusterer.fit_predict(X_compresso)
            
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            # Penalizziamo soluzioni con troppo rumore
            if noise_ratio > 0.4:
                continue 
                
            scores_simulati = []
            
            for classe_da_ignorare in classi_uniche:
                maschera_loco = df_labeled['Specie Predetta'] != classe_da_ignorare
                labels_reali = df_labeled[maschera_loco]['Specie Predetta']
                labels_pred = labels[maschera_loco]
                
                fmi = fowlkes_mallows_score(labels_reali, labels_pred)
                scores_simulati.append(fmi)
                
            mean_fmi = np.mean(scores_simulati)
            score_finale = mean_fmi * (1.0 - noise_ratio)
            fmi_base = fowlkes_mallows_score(df_labeled['Specie Predetta'], labels)
            
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
     Input('uh-toggle-noise-lab', 'value'),
     Input('store-top5-uh-lab', 'data')]
)
def aggiorna_uh_labeled(umap_neighbors, mcs, ms, mostra_rumore, top5_data):
    # 🌟 1. Spazio per il CLUSTERING (15D)
    # Aumentiamo min_dist a 0.25 per non schiacciare troppo i dati
    reducer_clustering = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_clustering = reducer_clustering.fit_transform(X_labeled)
    
    # 🌟 2. Estrazione dei Cluster
    # Usiamo 'eom' (Excess of Mass) invece di 'leaf' per evitare l'over-segmentazione
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_clustering)
    
    # 🌟 3. Spazio per il GRAFICO PLOTLY (3D)
    # Usiamo una proiezione separata solo per la visualizzazione estetica
    reducer_plot = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3, n_components=3, metric='cosine', random_state=42)
    X_plot = reducer_plot.fit_transform(X_labeled)
    
    df_plot = df_labeled.copy()
    df_plot['Cluster'] = [str(l) if l != -1 else 'Rumore' for l in labels]
    
    # 🌟 4. Sovrascriviamo le coordinate x, y, z usando SOLO l'output 3D (X_plot)
    df_plot['x'] = X_plot[:, 0]
    df_plot['y'] = X_plot[:, 1]
    df_plot['z'] = X_plot[:, 2]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    fmi = fowlkes_mallows_score(df_plot['Specie Predetta'], labels)
    
    if not mostra_rumore:
        df_plot = df_plot[df_plot['Cluster'] != 'Rumore']

    fig = genera_grafico_3d(df_plot, "Galassie UMAP 3D (Labeled Set)")
    
    if 'Rumore' in df_plot['Cluster'].values:
        fig.update_traces(selector=dict(name="Rumore"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = genera_tabella_crosstab(df_plot)
    
    elementi_box = [
        html.H6("Metriche UMAP+HDBSCAN:", className="text-success fw-bold"), 
        html.P(f"FMI (Purezza): {fmi:.4f}", className="mb-1"),
        html.P(f"Rumore (Scartate): {noise_ratio:.1%}", className="mb-1 text-danger")
    ]
    
    if top5_data and mcs == top5_data[0]['mcs'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("🏆 Top 5 Parametri HDBSCAN:", className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"MCS: {res['mcs']} | MS: {res['ms']} | FMI: {res['fmi']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
            
    return fig, tabella, html.Div(elementi_box)

# =========================================================
# GESTIONE FILTRI
# =========================================================

@callback(
    Output('filter-main-uh', 'value'),
    Input('filter-main-uh', 'value')
)
def gestisci_filtri_uh(filtri_selezionati):
    triggered_id = ctx.triggered_id
    nuovi_filtri = filtri_selezionati
    
    if triggered_id == 'filter-main-uh':
        # Logica per il tasto "Seleziona Tutti"
        if 'ALL' in filtri_selezionati and len(filtri_selezionati) < len(ORDINE_CATEGORIE) + 1:
            nuovi_filtri = ['ALL'] + ORDINE_CATEGORIE
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    # Evita loop di aggiornamento se i filtri non sono cambiati
    if nuovi_filtri == filtri_selezionati and triggered_id:
        return dash.no_update
        
    return nuovi_filtri

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
     Input('uh-toggle-noise-ted', 'value'),
     Input('store-top5-uh-ted', 'data'),
     Input('filter-main-uh', 'value')]
)
def aggiorna_uh_ted(umap_neighbors, mcs, ms, mostra_rumore, top5_data, categorie):
    if not categorie:
        return dash.no_update, "Nessun dato", html.Div("Seleziona almeno un filtro")

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    df_ted = DF_GLOBALE[maschera_ted].copy()
    
    if len(X_ted) < mcs:
        return dash.no_update, "Pochi dati", html.Div("Dati insufficienti.")

    # 🌟 1. Spazio per il CLUSTERING (15D)
    # Aumentiamo min_dist a 0.25 per mantenere la struttura intatta
    reducer_clustering = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
    X_clustering = reducer_clustering.fit_transform(X_ted)
    
    # 🌟 2. Estrazione dei Cluster
    # Assicurati di usare X_clustering. L'algoritmo 'eom' era già impostato correttamente!
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_clustering)
    
    # 🌟 3. Spazio per il GRAFICO PLOTLY (3D)
    # Proiezione estetica separata
    reducer_plot = umap.UMAP(n_neighbors=umap_neighbors, min_dist=0.3, n_components=3, metric='cosine', random_state=42)
    X_plot = reducer_plot.fit_transform(X_ted)

    df_ted['Cluster'] = [str(l) if l != -1 else 'Rumore' for l in labels]
    
    # 🌟 4. Sovrascriviamo le coordinate x, y, z usando SOLO l'output 3D (X_plot)
    df_ted['x'] = X_plot[:, 0]
    df_ted['y'] = X_plot[:, 1]
    df_ted['z'] = X_plot[:, 2]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if not mostra_rumore:
        df_ted = df_ted[df_ted['Cluster'] != 'Rumore']

    fig = genera_grafico_3d(df_ted, "Galassie UMAP 3D (Unlabeled Set)")
    
    if 'Rumore' in df_ted['Cluster'].values:
        fig.update_traces(selector=dict(name="Rumore"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = genera_tabella_crosstab(df_ted)

    # 1. Creiamo la LISTA base degli elementi
    elementi_box = [
        html.H6(f"Immagini analizzate: {len(X_ted)}", className="text-secondary fw-bold"),
        html.P(f"Galassie Trovate: {num_clusters}", className="text-primary fw-bold mb-1"),
        html.P(f"Rumore (Scartate): {noise_ratio:.1%}", className="text-danger fw-bold")
    ]
    
    # 2. Aggiungiamo i dati della Top 5 alla LISTA (se esistono)
    if top5_data and mcs == top5_data[0]['mcs'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="my-2"))
        titolo_top5 = "🏆 Top 5 Parametri (Ibrido FMI):" if top5_data[0]['metrica'] == 'FMI' else "🏆 Top 5 Parametri (Ciechi):"
        elementi_box.append(html.H6(titolo_top5, className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"N: {res['neigh']} | MCS: {res['mcs']} | MS: {res['ms']} | K: {res['k']} | {res['metrica']}: {res['val']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )

    # 3. Impacchettiamo tutta la lista nel Div finale
    metriche = html.Div(elementi_box)

    return fig, tabella, metriche

# 🌟 Aggiunto l'Output per sovrascrivere in automatico anche lo slider di UMAP!
@callback(
    [Output('uh-slider-umap-neighbors-ted', 'value', allow_duplicate=True),
     Output('uh-slider-mcs-ted', 'value', allow_duplicate=True),
     Output('uh-slider-ms-ted', 'value', allow_duplicate=True),
     Output('store-top5-uh-ted', 'data')],
    Input('uh-btn-opt-ted', 'n_clicks'),
    State('filter-main-uh', 'value'),
    prevent_initial_call=True
)
def auto_ottimizza_uh_ted_unsupervised_total(n_clicks, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    df_ted = DF_GLOBALE[maschera_ted].copy()
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    
    if len(X_ted) < 10:
        raise PreventUpdate

    # 🌟 IL TRUCCO DELLE ANCORE: Controlliamo se l'utente ha incluso il Labeled Set
    maschera_labeled_interna = df_ted['UnifiedCategory'] == 'Labeled Set'
    usa_ancore = maschera_labeled_interna.sum() > 0

    risultati = []
    best_score = -2.0 
    best_neigh, best_mcs, best_ms = 15, 15, 5
    
    # ... (codice precedente invariato) ...
    
    for n_neigh in [5, 10, 15, 20]:
        reducer = umap.UMAP(n_neighbors=n_neigh, min_dist=0.25, n_components=15, metric='cosine', random_state=42)
        X_compresso = reducer.fit_transform(X_ted)
        
        for mcs in range(10, 35, 5):
            # 🌟 FIX 1: Mai più MS=1. Partiamo da 3 per distruggere i "ponti" tra le galassie
            for ms in range(3, 10, 2): 
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean', cluster_selection_method='eom')
                labels = clusterer.fit_predict(X_compresso)
                
                noise_ratio = np.sum(labels == -1) / len(labels)
                mask_validi = labels != -1
                n_clusters = len(set(labels[mask_validi])) if np.sum(mask_validi) > 0 else 0
                
                # Accettiamo fino a 15 cluster (lasciamo margine per frammentazioni naturali)
                if n_clusters < 3 or n_clusters > 15 or noise_ratio > 0.4:
                    continue
                    
                if usa_ancore:
                    # 🌟 TUNING SEMI-SUPERVISIONATO
                    labels_pred_ancore = labels[maschera_labeled_interna]
                    labels_reali_ancore = df_ted[maschera_labeled_interna]['Specie Predetta']
                    
                    valore_metrica = fowlkes_mallows_score(labels_reali_ancore, labels_pred_ancore)
                    nome_metrica = "FMI"
                    
                    # 🌟 FIX 2: Il K-Penalty. Se trova meno cluster delle specie conosciute, lo puniamo severamente!
                    specie_conosciute = len(labels_reali_ancore.unique())
                    k_penalty = 1.0
                    if n_clusters < specie_conosciute:
                        k_penalty = 0.5  # Taglia il punteggio a metà
                    
                    # Penalizziamo il rumore in modo bilanciato
                    penalty_rumore = max(0.01, 1.0 - (noise_ratio * 1.5))
                    score_finale = valore_metrica * penalty_rumore * k_penalty
                    
                else:
                    # TUNING CIECO CLASSICO
                    try:
                        valore_metrica = silhouette_score(X_compresso[mask_validi], labels[mask_validi], metric='euclidean')
                        nome_metrica = "Sil"
                        penalty_rumore = max(0.01, 1.0 - (noise_ratio * 3.0)) 
                        score_finale = valore_metrica * penalty_rumore
                    except:
                        continue
                        
                risultati.append({
                    'neigh': n_neigh, 'mcs': mcs, 'ms': ms, 'val': valore_metrica, 
                    'noise': noise_ratio, 'score': score_finale, 'k': n_clusters, 'metrica': nome_metrica
                })
                
                if score_finale > best_score:
                    best_score = score_finale
                    best_neigh = n_neigh
                    best_mcs = mcs
                    best_ms = ms
                    
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    if best_score == -2.0 or not top5:
        return 15, 15, 5, []
        
    return best_neigh, best_mcs, best_ms, top5

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