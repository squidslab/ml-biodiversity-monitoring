import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name="Home")

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            
            html.Div([
                html.H1("Discover Patterns in Your Image Archive", className="text-primary mb-3"),
                html.P(
                    "This tool is designed to help you analyze and discover patterns within large photo archives. "
                    "The system uses various Clustering algorithms to automatically "
                    "group images sharing similar visual features.",
                    className="lead mb-0"
                ),
            ], className="mb-4 mt-4"),
            
            # STEP 1
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        html.H3("Step 1: Tool Calibration on Labeled Data", className="step-title"),
                        dbc.Col([
                            html.P([
                                "The goal of this first step is simply to find the right settings to group the images correctly.",
                                html.Br(),
                                html.Br(), 
                                "We start by working with a labeled dataset (images where the species is already known). ",
                                "You can tweak the parameters, as shown in this example, until each species is neatly separated into its own distinct cluster.",
                                html.Br(),
                                "Once these optimal parameters are set, the system essentially 'learns' how to look at the data. "
                                "It now knows exactly which visual features are the most important to tell one image apart from another.",
                                html.Br(),
                                html.Br(),
                                html.Em("Please note: This is just an example. Graph colors do not indicate species distribution. "
                                "Use the dedicated distribution table for accurate details.")                              
                            ], className="lead mb-0")
                        ], md=5, className="pe-4"), 
                        
                        dbc.Col([
                            html.Div([
                                html.Iframe(
                                    src="/assets/_style/animazione_calibration.html",
                                    style={
                                        "width": "100%", 
                                        "height": "400px",
                                        "backgroundColor": "#ffffff"
                                    }
                                )
                            ])
                        ], md=7)
                        
                    ], align="center") 
                ]),
                className="mb-4 shadow-sm card-custom-sx"
            ),

            # INFO CARD
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Not sure how to adjust the values?", className="step-title-info mb-2"),
                            html.P(
                                "With just one click, you can let the system do the heavy lifting and mathematically calculate the perfect configuration for you!",
                                className="lead mb-0 pe-md-4" 
                            )
                        ], md=7, className="text-center text-md-center"), 
                        dbc.Col([
                            dbc.Button(
                                "Optimize ✨", 
                                id="btn-opt-example", 
                                size="lg", 
                                color="warning", 
                                outline=True,
                                className="fw-bold px-4 rounded-pill shadow-sm magic-btn" 
                            )
                        ], md=5, className="d-flex align-items-center justify-content-center justify-content-md-center mt-3 mt-md-0") 
                    ], align="center") 
                ]),
                className="mb-4 shadow-sm card-info border-0"
            ),

            # STEP 2
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        html.H3("Step 2: Analyzing Unlabeled Data", className="step-title-dx"),
                       
                       dbc.Col([
                            html.Div([
                                html.Iframe(
                                    src="/assets/_style/animazione_clustering.html",
                                    style={
                                        "width": "100%", 
                                        "height": "400px",
                                        "backgroundColor": "#ffffff"
                                    }
                                )
                            ])
                        ], md=7),

                        dbc.Col([
                            html.P([
                                "Now that the tool is calibrated, you are ready to analyze unknown samples. In this section, "
                                "you can apply the exact settings discovered in the first step to your unlabeled dataset.",
                                html.Br(),
                                html.Br(), 
                                "Examine the resulting clusters and visually inspect the images to discover the shared patterns connecting them.",
                                html.Br(),
                                html.Br(),
                                "You can also apply quality filters or run a mixed clustering analysis "
                                "(combining both labeled and unlabeled datasets) for quality control purposes."                           
                            ], className="lead mb-0")
                        ], md=5, className="pe-4"), 
                        
                    ], align="center") 
                ]),
                className="mb-4 shadow-sm card-custom-dx"
            ),
            
            # GLOSSARY
            dbc.Card([
                dbc.CardBody([
                    html.H3("Essential Glossary", className="glossary-title mb-2"),
                    html.P(
                        "Machine learning terminology can be dense. Here is a quick translation guide designed to help you interpret the algorithm's behavior.",
                        className="text-muted mb-4 lead"
                    ),
                    
                    html.Ul([
                        html.Li([
                            html.B("Clustering: ", className="fw-bold"), 
                            "The automated grouping of plant specimens based entirely on visual and morphological similarities. Clustering independently discovers natural taxonomic groupings within your data."
                        ], className="mb-3 lead"),
                        
                        html.Li([
                            html.Strong("Labeled and Unlabeled Datasets: ", className="fw-bold"), 
                            "A ", html.I("Labeled Dataset"), " consists of images already identified and verified by a taxonomist. An ", html.I("Unlabeled Dataset"), " consists of unknown field samples or raw data that the algorithm needs to group and analyze."
                        ], className="mb-3 lead"),
                        
                        html.Li([
                            html.Strong("Hyperparameters: ", className="fw-bold"), 
                            "The adjustable 'tuning knobs' of the algorithm. Just as you adjust the focus and contrast on a microscope to see more clearly, tweaking hyperparameters changes how strictly or loosely the algorithm groups similar plants together."
                        ], className="mb-3 lead"),
                        
                        html.Li([
                            html.Strong("Noise / Outliers: ", className="fw-bold"), 
                            "Specimens that the algorithm cannot confidently assign to any main cluster. In a botanical context, these are often natural hybrids, highly atypical phenotypic variations, or simply poor-quality photographs that lack distinct morphological features."
                        ], className="mb-3 lead"),
                        
                        # The Evaluation
                        html.Li([
                            html.Strong("Validation Metrics (AMI, ARI, FMI): ", className="fw-bold"), 
                            "Statistical scores ranging from 0.0 to 1.0 that measure the algorithm's accuracy. They evaluate how perfectly the machine's automated groupings match your established human-verified taxonomy. A score close to 1.0 indicates near-perfect biological alignment."
                        ], className="mb-0 lead")
                        
                    ], ) # Removes default bullet points for a cleaner look
                ])
            ], className="mb-5 shadow-sm border-0 card-glossary")
            
        ], width=12, lg=10, className="mx-auto")
    ])
], fluid=True, className="pb-5")