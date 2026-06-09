import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, no_update
from dash.exceptions import PreventUpdate
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, pairwise_distances
from sklearn.decomposition import PCA

# Assicurati di importare le variabili e le funzioni dal tuo file utils
from utils import (
    DF_GLOBALE, EMBEDDINGS_GLOBALI, 
    genera_grafico_3d, genera_tabella_crosstab, calcola_percorso_hover
)

# Creiamo i dataset Labeled filtrando i dati globali all'avvio della pagina
maschera_labeled = DF_GLOBALE['UnifiedCategory'] == 'Labeled Set'
X_labeled = EMBEDDINGS_GLOBALI[maschera_labeled]
df_labeled = DF_GLOBALE[maschera_labeled].copy()

dash.register_page(__name__, path='/affinity', name='Affinity Propagation')

ORDINE_CATEGORIE = ['Labeled Set', 'Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others']

# =========================================================
# LAYOUT
# =========================================================

layout = html.Div([
    html.H2("Affinity Propagation Clustering", className="mb-3", style={"color": "#430783"}),
    html.Hr(),
    
    # ---------------------------------------------------------
    # FASE 1: LABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 1: Calibrazione delle Votazioni su Labeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚙️ Parametri Affinity", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Div([
                        html.Label("Preferenza (Quantile %):", className="fw-bold mt-2"),
                        dbc.Button("✨ Auto", id="aff-btn-opt-lab", size="sm", color="warning", outline=True, className="float-end")
                    ], className="d-flex justify-content-between align-items-center mb-2"),
                    html.Div(
                        "Alto = Più leader (più cluster). Basso = Meno leader (meno cluster).", 
                        className="text-muted mb-2", style={"fontSize": "12px"}
                    ),
                    dcc.Slider(id='aff-slider-pref-lab', min=5, max=95, step=5, value=50, marks={i: str(i) for i in range(10, 100, 20)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Damping (Smorzamento):", className="fw-bold mt-2"),
                    html.Div(
                        "Evita oscillazioni se l'algoritmo non riesce a decidere i leader.", 
                        className="text-muted mb-2", style={"fontSize": "12px"}
                    ),
                    dcc.Slider(id='aff-slider-damping-lab', min=0.5, max=0.95, step=0.05, value=0.5, marks={round(i*0.1, 1): str(round(i*0.1, 1)) for i in range(5, 10)}, className="mb-4", tooltip={"always_visible": False}),

                    html.Hr(),
                    html.Div(id='aff-metriche-box-lab', className="mt-3"),
                    dcc.Store(id='store-top5-aff-lab', data=[])
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='aff-grafico-3d-lab', style={'height': '420px'}, clear_on_unhover=True),
                            className="p-0" 
                        ), 
                        style={'height': '420px'},
                        className="shadow-sm border-0"
                    ), width=9
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Preview", className="text-center fw-bold", style={"backgroundColor": "#C499F9", "fontSize": "12px", "height": "40px", "padding": "8px"}),
                        dbc.CardBody([
                            html.H6(id='aff-hover-text-lab', className="text-center text-secondary mb-2", style={'fontSize': '11px', 'height': '15px'}),
                            html.Img(
                                id='aff-hover-image-lab', 
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
                        dbc.CardBody(id='aff-tabella-crosstab-lab')
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
                dbc.CardHeader("⚙️ Parametri Affinity", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    dbc.Button("🔄 Usa parametri Labeled Set", id="btn-sync-params-aff", color="success", outline=True, className="w-100 mb-4 shadow-sm"),
                    
                    html.Label("Preferenza (Quantile %):", className="fw-bold mt-2"),
                    dcc.Slider(id='aff-slider-pref-ted', min=5, max=95, step=5, value=50, marks={i: str(i) for i in range(10, 100, 20)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Damping:", className="fw-bold mt-2"),
                    dcc.Slider(id='aff-slider-damping-ted', min=0.5, max=0.95, step=0.05, value=0.5, marks={round(i*0.1, 1): str(round(i*0.1, 1)) for i in range(5, 10)}, className="mb-4", tooltip={"always_visible": False}),

                    html.Hr(),
                    html.Div(id='aff-metriche-box-ted', className="mt-3")
                ])
            ], className="shadow-sm border-0 h-auto mb-3"),
            dbc.Card([
                dbc.CardHeader("🔍 Filtri Dataset", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Div([
                        dbc.Checklist(
                            id='filter-main-aff',
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
                            dcc.Graph(id='aff-grafico-3d-ted', style={'height': '420px'}, clear_on_unhover=True),
                            className="p-0"
                        ), style={'height': '420px'}, className="shadow-sm border-0"
                    ), width=9
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader("Preview", className="text-center fw-bold", style={"backgroundColor": "#C499F9", "fontSize": "12px", "height": "40px", "padding": "8px"}),
                        dbc.CardBody([
                            html.H6(id='aff-hover-text-ted', className="text-center text-secondary mb-2", style={'fontSize': '11px'}),
                            html.Img(
                                id='aff-hover-image-ted', 
                                style={'maxWidth': '100%', 'maxHeight': '330px', 'objectFit': 'contain', 'borderRadius': '5px'}
                            )
                        ], className="d-flex flex-column align-items-center justify-content-center", style={'height': '380px'}) 
                    ], style={'height': '420px'}, className="shadow-sm border-0"), width=3
                )
            ], className="border-0 mb-3 align-items-stretch"),

            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Matrice di Distribuzione (Unlabeled Set)", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                    dbc.CardBody(id='aff-tabella-crosstab-ted')
                ], className="shadow-sm border-0"))
            ])
        ], width=9)
    ])
], className="mb-5")

# =========================================================
# FUNZIONE HELPER PER IL CALCOLO DELLA PREFERENZA
# =========================================================
def calcola_preferenza(X, quantile):
    # Affinity Propagation usa le negative squared euclidean distances come similarità
    S = -pairwise_distances(X, squared=True)
    pref = float(np.percentile(S, quantile))
    
    # OPZIONE NUCLEARE: Se il quantile è 0, abbassiamo artificialmente 
    # la preferenza moltiplicandola, per forzare la creazione di pochissimi cluster
    if quantile == 0:
        return pref * 3.0  
        
    return pref

# =========================================================
# CALLBACKS: FASE 1 (Auto-Tuning)
# =========================================================

@callback(
    [Output('aff-slider-pref-lab', 'value', allow_duplicate=True),
     Output('aff-slider-damping-lab', 'value', allow_duplicate=True),
     Output('store-top5-aff-lab', 'data')],
    Input('aff-btn-opt-lab', 'n_clicks'),
    prevent_initial_call=True
)
def auto_ottimizza_aff_labeled(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    risultati = []
    best_score = -1
    best_q = 50
    best_damp = 0.5
    
    classi_uniche = df_labeled['Specie Predetta'].unique()
    
    # 🌟 NOVITÀ: Comprimiamo lo spazio a 20 dimensioni per rimuovere il rumore
    pca = PCA(n_components=20, random_state=42)
    X_compresso = pca.fit_transform(X_labeled)
    
    # Grid Search ragionevole (aggiunto lo 0 tra i test)
    quantili_da_testare = [0, 10, 30, 50, 70]
    damping_da_testare = [0.5, 0.7, 0.9]
    
    for q in quantili_da_testare:
        pref = calcola_preferenza(X_compresso, q)
        for damp in damping_da_testare:
            # Fit sui dati compressi!
            clusterer = AffinityPropagation(damping=damp, preference=pref, random_state=42)
            labels = clusterer.fit_predict(X_compresso)
            
            # Se trova un solo cluster o non converge, il punteggio è zero
            if len(set(labels)) <= 1:
                continue
                
            scores_simulati = []
            
            # LOCO Validation
            for classe_da_ignorare in classi_uniche:
                maschera_loco = df_labeled['Specie Predetta'] != classe_da_ignorare
                labels_reali = df_labeled[maschera_loco]['Specie Predetta']
                labels_pred = labels[maschera_loco]
                
                fmi = fowlkes_mallows_score(labels_reali, labels_pred)
                scores_simulati.append(fmi)
                
            mean_fmi = np.mean(scores_simulati)
            fmi_base = fowlkes_mallows_score(df_labeled['Specie Predetta'], labels)
            
            risultati.append({
                'q': q, 'damp': damp, 'fmi_base': fmi_base, 'score': mean_fmi, 'n_clusters': len(set(labels))
            })
            
            if mean_fmi > best_score:
                best_score = mean_fmi
                best_q = q
                best_damp = damp
                
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    return best_q, best_damp, top5


@callback(
    [Output('aff-grafico-3d-lab', 'figure'), 
     Output('aff-tabella-crosstab-lab', 'children'), 
     Output('aff-metriche-box-lab', 'children')],
    [Input('aff-slider-pref-lab', 'value'), 
     Input('aff-slider-damping-lab', 'value'),
     Input('store-top5-aff-lab', 'data')]
)
def aggiorna_aff_labeled(q, damp, top5_data):
    # 🌟 Compressione PCA
    pca = PCA(n_components=20, random_state=42)
    X_compresso = pca.fit_transform(X_labeled)
    
    pref = calcola_preferenza(X_compresso, q)
    clusterer = AffinityPropagation(damping=damp, preference=pref, random_state=42)
    labels = clusterer.fit_predict(X_compresso)
    
    n_clusters = len(set(labels))
    
    df_plot = df_labeled.copy()
    df_plot['Cluster'] = [str(l) for l in labels]
    
    fmi = fowlkes_mallows_score(df_plot['Specie Predetta'], labels)
    
    fig = genera_grafico_3d(df_plot, "Spazio Latente Labeled Set (Affinity + PCA)")
    tabella = genera_tabella_crosstab(df_plot)
    
    elementi_box = [
        html.H6("Metriche Affinity:", className="text-success fw-bold"), 
        html.P(f"Isole Formate: {n_clusters}", className="mb-1 fw-bold"),
        html.P(f"FMI (Purezza): {fmi:.4f}", className="mb-1")
    ]
    
    if top5_data and q == top5_data[0]['q'] and damp == top5_data[0]['damp']:
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("🏆 Top 5 Parametri (Score LOCO):", className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Quantile: {res['q']}% | Damp: {res['damp']} | K: {res['n_clusters']} | LOCO: {res['score']:.3f}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
            
    return fig, tabella, html.Div(elementi_box)

# =========================================================
# GESTIONE FILTRI E SYNC
# =========================================================

@callback(
    [Output('filter-main-aff', 'value'),
     Output('aff-slider-pref-ted', 'value'),
     Output('aff-slider-damping-ted', 'value')],
    [Input('filter-main-aff', 'value'),
     Input('btn-sync-params-aff', 'n_clicks')],
    [State('aff-slider-pref-lab', 'value'),
     State('aff-slider-damping-lab', 'value'),
     State('aff-slider-pref-ted', 'value'),
     State('aff-slider-damping-ted', 'value')]
)
def gestisci_input_aff_ted(filtri_selezionati, n_clicks, lab_q, lab_damp, ted_q, ted_damp):
    triggered_id = ctx.triggered_id
    nuovi_filtri = filtri_selezionati
    
    if triggered_id == 'filter-main-aff':
        if 'ALL' in filtri_selezionati:
            nuovi_filtri = ['ALL'] + ORDINE_CATEGORIE
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    if triggered_id == 'btn-sync-params-aff':
        return nuovi_filtri, lab_q, lab_damp

    return nuovi_filtri, ted_q, ted_damp

# =========================================================
# FASE 2: UNLABELED SET
# =========================================================

@callback(
    [Output('aff-grafico-3d-ted', 'figure'), 
     Output('aff-tabella-crosstab-ted', 'children'), 
     Output('aff-metriche-box-ted', 'children')],
    [Input('aff-slider-pref-ted', 'value'), 
     Input('aff-slider-damping-ted', 'value'),
     Input('filter-main-aff', 'value')]
)
def aggiorna_aff_ted(q, damp, categorie):
    if not categorie:
        return dash.no_update, "Nessun dato", html.Div("Seleziona almeno un filtro")

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    df_ted = DF_GLOBALE[maschera_ted].copy()
    
    if len(X_ted) < 5:
        return dash.no_update, "Pochi dati", html.Div("Dati insufficienti.")

    # 🌟 Compressione PCA anche sul TED Set
    pca = PCA(n_components=20, random_state=42)
    X_compresso = pca.fit_transform(X_ted)

    pref = calcola_preferenza(X_compresso, q)
    clusterer = AffinityPropagation(damping=damp, preference=pref, random_state=42)
    labels = clusterer.fit_predict(X_compresso)
    
    n_clusters = len(set(labels))
    df_ted['Cluster'] = [str(l) for l in labels]
    
    try:
        if n_clusters > 1:
            # Calcoliamo il Silhouette sullo spazio originale per coerenza con le altre pagine
            sil_score = silhouette_score(X_ted, labels, metric='euclidean')
            sil_text = f"{sil_score:.3f}"
        else:
            sil_text = "N/A"
    except Exception:
        sil_text = "Errore Calcolo"

    fig = genera_grafico_3d(df_ted, "Scoperta Classi (Affinity + PCA)")
    tabella = genera_tabella_crosstab(df_ted)

    metriche = html.Div([
        html.H6(f"Immagini analizzate: {len(X_ted)}", className="text-secondary fw-bold"),
        html.P(f"Isole Formate: {n_clusters}", className="text-primary fw-bold mb-1"),
        html.P(f"Silhouette Score: {sil_text}", className="text-success fw-bold")
    ])
    
    return fig, tabella, metriche

# =========================================================
# CALLBACK HOVER
# =========================================================

@callback(
    [Output('aff-hover-image-lab', 'src'), Output('aff-hover-text-lab', 'children')],
    Input('aff-grafico-3d-lab', 'hoverData')
)
def hover_lab(hoverData): return calcola_percorso_hover(hoverData)

@callback(
    [Output('aff-hover-image-ted', 'src'), Output('aff-hover-text-ted', 'children')],
    Input('aff-grafico-3d-ted', 'hoverData')
)
def hover_ted(hoverData): return calcola_percorso_hover(hoverData)