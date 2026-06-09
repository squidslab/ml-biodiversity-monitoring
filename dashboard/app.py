import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Inizializzazzione Dash con funzionalità Pages
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY])

# Barra di Navigazione
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/", active="exact")), 
        dbc.NavItem(dbc.NavLink("Spectral Clustering", href="/spectral", active="exact")),
        dbc.NavItem(dbc.NavLink("HBDBSCAN", href="/hdbscan", active="exact")),
        dbc.NavItem(dbc.NavLink("DBSCAN", href="/dbscan", active="exact")),
        dbc.NavItem(dbc.NavLink("Agglomerative Clustering", href="/agglomerative", active="exact")),
    ],
    brand=[
        html.Img(
            src="/assets/logo.png",
            height="70px", 
            className="me-3"
        ),
        html.Div([
            html.Div("OrchiData", className="brand-title"),
            html.Div("Clustering Workspace", className="brand-subtitle")
        ], className="d-flex flex-column justify-content-center")
    ],
    className="mb-4 shadow-sm",
    color="#25425B",
    dark=True,
    fluid=True, 
)

# Il layout globale della tua applicazione
app.layout = html.Div([
    navbar,
    
    # Questo è il "contenitore magico" dove Dash caricherà i contenuti delle varie pagine
    dbc.Container([
        dash.page_container
    ], fluid=True)
])

if __name__ == '__main__':
    app.run(debug=True)