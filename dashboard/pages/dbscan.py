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

# Assicurati di importare le variabili e le funzioni dal tuo file utils
from utils import (
    DF_GLOBALE, EMBEDDINGS_GLOBALI, 
    genera_grafico_3d, genera_tabella_crosstab, calcola_percorso_hover
)

# Creiamo i dataset Labeled filtrando i dati globali all'avvio della pagina
maschera_labeled = DF_GLOBALE['UnifiedCategory'] == 'Labeled Set'
X_labeled = EMBEDDINGS_GLOBALI[maschera_labeled]
df_labeled = DF_GLOBALE[maschera_labeled].copy()

dash.register_page(__name__, path='/dbscan', name='DBSCAN Clustering')

# Supponendo che ORDINE_CATEGORIE sia importato da utils
ORDINE_CATEGORIE = ['Labeled Set', 'Curated', 'Usable', 'Hardcore', 'Ruined Surface', 'Hands', 'Others']

layout = html.Div([
    html.H2("DBSCAN Clustering", className="mb-3", style={"color": "#430783"}),
    html.Hr(),
    
    # ---------------------------------------------------------
    # FASE 1: LABELED SET
    # ---------------------------------------------------------
    html.H4("Fase 1: Calibrazione della Distanza su Labeled Set", className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                # Selezione parametri Labeled 
                dbc.CardHeader("⚙️ Parametri DBSCAN", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Div([
                        html.Label("Epsilon (eps):", className="fw-bold mt-2"),
                        dbc.Button("✨ Auto", id="btn-opt-dbscan-lab", size="sm", color="warning", outline=True, className="float-end")
                    ], className="d-flex justify-content-between align-items-center mb-2"),
                    html.Div(
                        "Distanza massima per unire due foto nello stesso cluster.", 
                        className="text-muted mb-2", style={"fontSize": "12px"}
                    ),
                    # Nota: range e step sono impostati per distanze Coseno (0.0 a 1.0). Se usi Euclidea, potresti doverli alzare.
                    dcc.Slider(id='dbscan-slider-eps-lab', min=0.01, max=0.5, step=0.01, value=0.3, marks={round(i*0.1, 1): str(round(i*0.1, 1)) for i in range(1, 6)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Min Samples:", className="fw-bold mt-2"),
                    html.Div(
                        "Foto minime per formare un cluster (densità).", 
                        className="text-muted mb-2", style={"fontSize": "12px"}
                    ),
                    dcc.Slider(id='dbscan-slider-min-samples-lab', min=2, max=30, step=1, value=15, marks={i: str(i) for i in range(5, 31, 5)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    # Controllo per gestire gli outlier
                    dbc.Checklist(
                        id="dbscan-toggle-noise-lab",
                        options=[{"label": " Mostra Rumore (Outliers)", "value": True}],
                        value=[True],
                        switch=True,
                        className="fw-bold text-secondary mt-2 mb-3"
                    ),

                    html.Hr(),
                    html.Div(id='dbscan-metriche-box-lab', className="mt-3"),
                    dcc.Store(id='store-top5-dbscan-lab', data=[])
                ])
            ], className="shadow-sm border-0 h-100")
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                # Grafico 3D Labeled 
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id='dbscan-grafico-3d-lab', style={'height': '420px'}, clear_on_unhover=True),
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
                            html.H6(id='dbscan-hover-text-lab', className="text-center text-secondary mb-2", style={'fontSize': '11px', 'height': '15px'}),
                        
                            html.Img(
                                id='dbscan-hover-image-lab', 
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
                        dbc.CardBody(id='dbscan-tabella-crosstab-lab')
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
                # Selezione parametri Unlabeled 
                dbc.CardHeader("⚙️ Parametri DBSCAN", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    dcc.Store(id='store-top5-dbscan-ted'),
                    dbc.Button("🔄 Usa parametri Labeled Set", id="btn-sync-params-dbscan", color="success", outline=True, className="w-100 mb-4 shadow-sm"),
                    dbc.Button("✨ Auto-Tune (con Validazione Labeled)", id="btn-opt-dbscan-ted", color="primary", className="w-100 mb-4 shadow-sm"),

                    html.Label("Epsilon (eps):", className="fw-bold mt-2"),
                    dcc.Slider(id='dbscan-slider-eps-ted', min=0.01, max=0.5, step=0.01, value=0.3, marks={round(i*0.1, 1): str(round(i*0.1, 1)) for i in range(1, 6)}, className="mb-4", tooltip={"always_visible": False}),
                    
                    html.Label("Min Samples:", className="fw-bold mt-2"),
                    dcc.Slider(id='dbscan-slider-min-samples-ted', min=2, max=30, step=1, value=15, marks={i: str(i) for i in range(5, 31, 5)}, className="mb-4", tooltip={"always_visible": False}),

                    dbc.Checklist(
                        id="dbscan-toggle-noise-ted",
                        options=[{"label": " Mostra Rumore (Outliers)", "value": True}],
                        value=[True],
                        switch=True,
                        className="fw-bold text-secondary mt-2 mb-3"
                    ),

                    html.Hr(),
                    html.Div(id='dbscan-metriche-box-ted', className="mt-3")
                ])
            ], className="shadow-sm border-0 h-auto mb-3"),
            dbc.Card([
                dbc.CardHeader("🔍 Filtri Dataset", className="fw-bold", style={"backgroundColor": "#C499F9"}),
                dbc.CardBody([
                    html.Div([
                        dbc.Checklist(
                            id='filter-main-dbscan',
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
                            dcc.Graph(id='dbscan-grafico-3d-ted', style={'height': '420px'}, clear_on_unhover=True),
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
                            html.H6(id='dbscan-hover-text-ted', className="text-center text-secondary mb-2", style={'fontSize': '11px'}),
                            
                            html.Img(
                                id='dbscan-hover-image-ted', 
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
                    dbc.CardBody(id='dbscan-tabella-crosstab-ted')
                ], className="shadow-sm border-0"))
            ])
        ], width=9)
    ])
], className="mb-5")

# =========================================================
# FASE 1: LABELED SET (Auto-Tuning e Visualizzazione)
# =========================================================

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

    risultati = []
    best_score = -1
    best_eps = 0.3
    best_ms = 5
    
    classi_uniche = df_labeled['Specie Predetta'].unique()
    
    # Grid Search per DBSCAN classico (Distanza Euclidea)
    # Range Eps da 0.1 a 1.0 (step 0.05 per non appesantire il calcolo automatico)
    for eps in np.arange(0.1, 1.05, 0.05):
        for ms in range(5, 31, 2):
            # Riapplichiamo il DBSCAN come lo usavi tu (Euclidea sui dati grezzi)
            clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
            labels = clusterer.fit_predict(X_labeled)
            
            # Calcolo del rumore (-1)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            # Penalizziamo configurazioni che scartano più del 40%
            if noise_ratio > 0.4:
                continue 
                
            scores_simulati = []
            
            # LOCO Validation (Robustezza)
            for classe_da_ignorare in classi_uniche:
                maschera_loco = df_labeled['Specie Predetta'] != classe_da_ignorare
                labels_reali = df_labeled[maschera_loco]['Specie Predetta']
                labels_pred = labels[maschera_loco]
                
                fmi = fowlkes_mallows_score(labels_reali, labels_pred)
                scores_simulati.append(fmi)
                
            mean_fmi = np.mean(scores_simulati)
            
            # Formula: FMI scalato per penalizzare eccesso di rumore
            score_finale = mean_fmi * (1.0 - noise_ratio)
            
            fmi_base = fowlkes_mallows_score(df_labeled['Specie Predetta'], labels)
            
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
     Input('dbscan-toggle-noise-lab', 'value'),
     Input('store-top5-dbscan-lab', 'data')]
)
def aggiorna_dbscan_labeled(eps, ms, mostra_rumore, top5_data):
    clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
    labels = clusterer.fit_predict(X_labeled)
    
    df_plot = df_labeled.copy()
    
    df_plot['Cluster'] = [str(l) if l != -1 else 'Rumore' for l in labels]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    fmi = fowlkes_mallows_score(df_plot['Specie Predetta'], labels)
    
    if not mostra_rumore:
        df_plot = df_plot[df_plot['Cluster'] != 'Rumore']
        
    if df_plot.empty:
        return dash.no_update, "Nessun dato (Tutto rumore)", html.Div("Tutte le foto sono state etichettate come rumore.")

    fig = genera_grafico_3d(df_plot, "Spazio Latente Labeled Set (DBSCAN)")
    
    if 'Rumore' in df_plot['Cluster'].values:
        fig.update_traces(selector=dict(name="Rumore"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = genera_tabella_crosstab(df_plot)
    
    elementi_box = [
        html.H6("Metriche DBSCAN:", className="text-success fw-bold"), 
        html.P(f"FMI (Purezza): {fmi:.4f}", className="mb-1"),
        html.P(f"Rumore (Scartate): {noise_ratio:.1%}", className="mb-1 text-danger")
    ]
    
    if top5_data and eps == top5_data[0]['eps'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("🏆 Top 5 Parametri:", className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Eps: {res['eps']} | MS: {res['ms']} | FMI: {res['fmi']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
            
    return fig, tabella, html.Div(elementi_box)


# =========================================================
# GESTIONE FILTRI E SYNC
# =========================================================

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
            nuovi_filtri = ['ALL'] + ORDINE_CATEGORIE
        elif filtri_selezionati == ['ALL']:
             nuovi_filtri = []

    if triggered_id == 'btn-sync-params-dbscan':
        return nuovi_filtri, lab_eps, lab_ms

    return nuovi_filtri, ted_eps, ted_ms


# =========================================================
# FASE 2: UNLABELED SET
# =========================================================

@callback(
    [Output('dbscan-grafico-3d-ted', 'figure'), 
     Output('dbscan-tabella-crosstab-ted', 'children'), 
     Output('dbscan-metriche-box-ted', 'children')],
    [Input('dbscan-slider-eps-ted', 'value'), 
     Input('dbscan-slider-min-samples-ted', 'value'),
     Input('dbscan-toggle-noise-ted', 'value'),
     Input('filter-main-dbscan', 'value'),
     Input('store-top5-dbscan-ted', 'data')]
)
def aggiorna_dbscan_ted(eps, ms, mostra_rumore, categorie, top5_data):
    if not categorie:
        return dash.no_update, "Nessun dato", html.Div("Seleziona almeno un filtro")

    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_ted = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    X_ted = EMBEDDINGS_GLOBALI[maschera_ted]
    df_ted = DF_GLOBALE[maschera_ted].copy()
    
    if len(X_ted) < ms:
        return dash.no_update, "Pochi dati", html.Div("Dati insufficienti.")

    clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
    labels = clusterer.fit_predict(X_ted)
    
    df_ted['Cluster'] = [str(l) if l != -1 else 'Rumore' for l in labels]
    
    noise_ratio = np.sum(labels == -1) / len(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Calcolo Silhouette Score SOLO sui punti che non sono rumore
    mask_no_noise = labels != -1
    try:
        if num_clusters > 1:
            sil_score = silhouette_score(X_ted[mask_no_noise], labels[mask_no_noise], metric='euclidean')
            sil_text = f"{sil_score:.3f}"
        else:
            sil_text = "N/A (Serve >1 cluster)"
    except Exception:
        sil_text = "Errore Calcolo"

    if not mostra_rumore:
        df_ted = df_ted[df_ted['Cluster'] != 'Rumore']

    fig = genera_grafico_3d(df_ted, "Scoperta Classi (DBSCAN Unsupervised)")
    
    if 'Rumore' in df_ted['Cluster'].values:
        fig.update_traces(selector=dict(name="Rumore"), marker=dict(color='rgba(150, 150, 150, 0.3)'))

    tabella = genera_tabella_crosstab(df_ted)

    elementi_box = [
        html.H6(f"Immagini analizzate: {len(X_ted)}", className="text-secondary fw-bold"),
        html.P(f"Isole Trovate: {num_clusters}", className="text-primary fw-bold mb-1"),
        html.P(f"Rumore (Scartate): {noise_ratio:.1%}", className="text-danger fw-bold mb-1"),
        html.P(f"Silhouette (Senza rumore): {sil_text}", className="text-success fw-bold")
    ]
    
    # Aggiunta della Top 5 se l'auto-tuning è stato eseguito e siamo sui parametri migliori
    if top5_data and eps == top5_data[0]['eps'] and ms == top5_data[0]['ms']:
        elementi_box.append(html.Hr(className="my-2"))
        elementi_box.append(html.H6("🏆 Top 5 Parametri (Validati su Labeled):", className="text-info fw-bold mt-2", style={'fontSize': '13px'}))
        for res in top5_data:
            elementi_box.append(
                html.P(f"Eps: {res['eps']} | MS: {res['ms']} | Score: {res['score']:.3f} | Noise: {res['noise']:.1%}", 
                       className="mb-0 text-muted", style={'fontSize': '12px'})
            )
    
    return fig, tabella, html.Div(elementi_box)


# =========================================================
# CALLBACK HOVER (Identiche per le due fasi)
# =========================================================

@callback(
    [Output('dbscan-hover-image-lab', 'src'), Output('dbscan-hover-text-lab', 'children')],
    Input('dbscan-grafico-3d-lab', 'hoverData')
)
def hover_lab(hoverData): return calcola_percorso_hover(hoverData)

@callback(
    [Output('dbscan-hover-image-ted', 'src'), Output('dbscan-hover-text-ted', 'children')],
    Input('dbscan-grafico-3d-ted', 'hoverData')
)
def hover_ted(hoverData): return calcola_percorso_hover(hoverData)

# =========================================================
# AUTO-TUNING FASE 2 (Misto: Unlabeled/Curated + Labeled)
# =========================================================

@callback(
    [Output('dbscan-slider-eps-ted', 'value', allow_duplicate=True),
     Output('dbscan-slider-min-samples-ted', 'value', allow_duplicate=True),
     Output('store-top5-dbscan-ted', 'data')], # Necessario creare un dcc.Store(id='store-top5-dbscan-ted') nel layout
    Input('btn-opt-dbscan-ted', 'n_clicks'),   # Necessario un bottone dedicato nel layout della Fase 2
    State('filter-main-dbscan', 'value'),      # Leggiamo i filtri attuali (es. ['Labeled', 'Curated'])
    prevent_initial_call=True
)
def auto_ottimizza_dbscan_misto(n_clicks, categorie):
    if not n_clicks or not categorie:
        raise PreventUpdate

    # 1. Filtriamo i dati globali in base alle categorie selezionate nella UI
    categorie_attive = [c for c in categorie if c != 'ALL']
    maschera_totale = DF_GLOBALE['UnifiedCategory'].isin(categorie_attive)
    
    df_subset = DF_GLOBALE[maschera_totale].copy()
    X_subset = EMBEDDINGS_GLOBALI[maschera_totale]

    # 2. Identifichiamo DOVE si trovano i dati Labeled all'interno del subset
    # Sostituisci 'Labeled' con il nome esatto usato nel tuo dataset
    maschera_labeled_interna = df_subset['UnifiedCategory'] == 'Labeled Set'
    
    # Se non c'è nessun dato Labeled nel mix, non possiamo validare le metriche
    if not maschera_labeled_interna.any():
        return dash.no_update, dash.no_update, dash.no_update

    # Estraiamo la Ground Truth (le etichette vere) solo per i dati Labeled
    labels_reali_labeled = df_subset[maschera_labeled_interna]['Specie Predetta']

    risultati = []
    best_score = -1
    best_eps = 0.3
    best_ms = 5
    
    # 3. Grid Search
    for eps in np.arange(0.1, 1.05, 0.05):
        for ms in range(5, 31, 2):
            # Il clustering avviene su TUTTO il subset (Labeled + Unlabeled/Curated)
            clusterer = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
            labels_totali = clusterer.fit_predict(X_subset)
            
            # Calcolo del rumore GLOBALE sul subset
            noise_ratio_totale = np.sum(labels_totali == -1) / len(labels_totali)
            
            # Scartiamo configurazioni con troppo rumore globale (> 40%)
            #if noise_ratio_totale > 0.4:
            #    continue 
                
            # 4. Estraiamo SOLO le predizioni relative ai dati Labeled per validarle
            labels_pred_labeled = labels_totali[maschera_labeled_interna]
            
            # Calcolo delle Metriche richieste
            ari = adjusted_rand_score(labels_reali_labeled, labels_pred_labeled)
            ami = adjusted_mutual_info_score(labels_reali_labeled, labels_pred_labeled)
            fmi = fowlkes_mallows_score(labels_reali_labeled, labels_pred_labeled)
            
            # Creiamo uno score composito (Media delle 3 metriche penalizzata dal rumore globale)
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
            
            # Aggiornamento dei migliori parametri
            if score_finale > best_score:
                best_score = score_finale
                best_eps = round(eps, 2)
                best_ms = ms
                
    # Estraiamo la Top 5
    top5 = sorted(risultati, key=lambda x: x['score'], reverse=True)[:5]
    
    return best_eps, best_ms, top5