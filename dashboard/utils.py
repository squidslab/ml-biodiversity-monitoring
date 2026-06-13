"""
=============================================================================
GENERIC DATA LOADING UTILITIES FOR CLUSTERING PIPELINE
=============================================================================

This module provides a generic, plug-and-play way to load features (.npz) 
and metadata (.xlsx) into the interactive clustering dashboard. 

HOW TO USE THIS MODULE FOR YOUR OWN DATA:
Modify the `DATASET_CONFIG` dictionary below to point to your files.
The dashboard operates dynamically in one of three scenarios based on what 
you provide here:

1. HYBRID MODE (Default):
   - Provide paths for both Labeled and Unlabeled features/metadata.
   - The pipeline will combine them, allowing you to calibrate algorithms 
     on the Labeled data (Step 1) and apply them to the Unlabeled data (Step 2).

2. ONLY UNLABELED DATA:
   - Provide paths for `UNLABELED_FEATURES_PATH` and `UNLABELED_METADATA_PATH`.
   - Set the Labeled paths to `None`. 
   - The dashboard will hide the calibration step and only show the Unlabeled set.

3. ONLY LABELED DATA:
   - Provide paths for `LABELED_FEATURES_PATH` and `LABELED_METADATA_PATH`.
   - Set the Unlabeled paths to `None`.
   - Useful for pure benchmarking and exploring ground-truth distributions.
     
*** REQUIRED EXCEL COLUMNS ***
Ensure your metadata files contain the columns specified in `IMAGE_ID_COL`,
`CATEGORY_COL`, and `PREDICTION_COL` within the configuration dictionary. 
If they are missing, the script will safely create placeholders (e.g., 'Unknown'), 
but validation metrics (AMI, ARI, FMI) will not compute correctly without 
true labels in the `PREDICTION_COL`.

*** IMPORTANT NOTE ON IMAGE PREVIEWS ***
To see specimen previews when hovering over the 3D plot points, the images 
must be physically located inside the `dashboard/assets/` directory.
If you used the `run_pipeline.py` orchestrator, this was done automatically! 
If you are loading data manually, ensure your folders exactly match the 
`UNLABELED_IMAGES_DIR` and `LABELED_IMAGES_DIR` variables configured below.
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import dash
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# ==========================================
# DATASET CONFIGURATION
# ==========================================
DATASET_CONFIG = {
    # --- File Paths ---
    # Set to None or "" if a specific subset is not available
    'UNLABELED_FEATURES_PATH': '../data/03_features/features_unlabeled.npz',
    'UNLABELED_METADATA_PATH': '../data/00_metadata/dataset_unlabeled.xlsx',
    
    'LABELED_FEATURES_PATH': '../data/03_features/features_labeled.npz',
    'LABELED_METADATA_PATH': '../data/00_metadata/dataset_labeled.xlsx',

    # --- Image Folders (Dash Assets) ---
    # These folders must be placed inside your dashboard/assets/ directory
    'UNLABELED_IMAGES_DIR': 'unlabeled_images',
    'LABELED_IMAGES_DIR': 'labeled_images',
    
    # --- Column Names in Metadata (Excel) ---
    'IMAGE_ID_COL': 'image name',           # Column used to match .npz names with Excel rows
    'CATEGORY_COL': 'Categoria',            # Column used for the UI filtering system
    'PREDICTION_COL': 'Specie Predetta'     # Column containing the classification label
}

# ==========================================
# MAIN LOADING FUNCTION
# ==========================================
def load_features_and_metadata():
    """
    Loads .npz features and Excel metadata based on DATASET_CONFIG.
    Handles scenarios with missing labeled or unlabeled sets dynamically.
    
    Returns:
        normalized_embeddings (numpy.ndarray): L2 normalized feature matrix.
        global_dataframe (pandas.DataFrame): Combined metadata with 3D PCA coordinates.
    """
    try:
        embeddings_list = []
        image_names_list = []
        metadata_dfs = []
        labeled_image_names = set()

        # ---------------------------------------------------------
        # 1. PROCESS UNLABELED DATA (If available)
        # ---------------------------------------------------------
        if DATASET_CONFIG['UNLABELED_FEATURES_PATH'] and os.path.exists(DATASET_CONFIG['UNLABELED_FEATURES_PATH']):
            unlabeled_npz = np.load(DATASET_CONFIG['UNLABELED_FEATURES_PATH'])
            embeddings_list.append(unlabeled_npz['embeddings'])
            image_names_list.append(unlabeled_npz['names'])
            
            if DATASET_CONFIG['UNLABELED_METADATA_PATH'] and os.path.exists(DATASET_CONFIG['UNLABELED_METADATA_PATH']):
                unlabeled_metadata_df = pd.read_excel(DATASET_CONFIG['UNLABELED_METADATA_PATH'])
                
                # Safe column checking
                if DATASET_CONFIG['CATEGORY_COL'] not in unlabeled_metadata_df.columns:
                    unlabeled_metadata_df[DATASET_CONFIG['CATEGORY_COL']] = 'Not Specified'
                if DATASET_CONFIG['PREDICTION_COL'] not in unlabeled_metadata_df.columns:
                    unlabeled_metadata_df[DATASET_CONFIG['PREDICTION_COL']] = 'Unknown'
                    
                clean_unlabeled_df = unlabeled_metadata_df[[
                    DATASET_CONFIG['IMAGE_ID_COL'], 
                    DATASET_CONFIG['CATEGORY_COL'], 
                    DATASET_CONFIG['PREDICTION_COL']
                ]].copy()
                metadata_dfs.append(clean_unlabeled_df)

        # 2. PROCESS LABELED DATA
        if DATASET_CONFIG['LABELED_FEATURES_PATH'] and os.path.exists(DATASET_CONFIG['LABELED_FEATURES_PATH']):
            labeled_npz = np.load(DATASET_CONFIG['LABELED_FEATURES_PATH'])
            embeddings_list.append(labeled_npz['embeddings'])
            image_names_list.append(labeled_npz['names'])
            
            labeled_image_names = set(labeled_npz['names'])
            
            if DATASET_CONFIG['LABELED_METADATA_PATH'] and os.path.exists(DATASET_CONFIG['LABELED_METADATA_PATH']):
                labeled_metadata_df = pd.read_excel(DATASET_CONFIG['LABELED_METADATA_PATH'])
                
                if 'Classe' in labeled_metadata_df.columns:
                    if DATASET_CONFIG['PREDICTION_COL'] in labeled_metadata_df.columns:
                        labeled_metadata_df = labeled_metadata_df.drop(columns=[DATASET_CONFIG['PREDICTION_COL']])
                    labeled_metadata_df = labeled_metadata_df.rename(columns={'Classe': DATASET_CONFIG['PREDICTION_COL']})
                elif DATASET_CONFIG['PREDICTION_COL'] not in labeled_metadata_df.columns:
                    labeled_metadata_df[DATASET_CONFIG['PREDICTION_COL']] = 'Unknown'
                    
                if 'Nome File' in labeled_metadata_df.columns:
                    if DATASET_CONFIG['IMAGE_ID_COL'] in labeled_metadata_df.columns:
                        labeled_metadata_df = labeled_metadata_df.drop(columns=[DATASET_CONFIG['IMAGE_ID_COL']])
                    labeled_metadata_df = labeled_metadata_df.rename(columns={'Nome File': DATASET_CONFIG['IMAGE_ID_COL']})
                
                if DATASET_CONFIG['PREDICTION_COL'] in labeled_metadata_df.columns:
                    labeled_metadata_df[DATASET_CONFIG['PREDICTION_COL']] = 'labeled_' + labeled_metadata_df[DATASET_CONFIG['PREDICTION_COL']].astype(str)
                    
                clean_labeled_df = labeled_metadata_df[[
                    DATASET_CONFIG['IMAGE_ID_COL'], 
                    DATASET_CONFIG['PREDICTION_COL']
                ]].copy()
                
                clean_labeled_df[DATASET_CONFIG['CATEGORY_COL']] = 'Labeled Set'
                metadata_dfs.append(clean_labeled_df)

        # ---------------------------------------------------------
        # 3. COMBINE AND NORMALIZE
        # ---------------------------------------------------------
        if not embeddings_list:
            raise ValueError("No data loaded. Check your file paths in DATASET_CONFIG.")

        combined_embeddings = np.vstack(embeddings_list)
        combined_image_names = np.concatenate(image_names_list)
        
        normalized_embeddings = normalize(combined_embeddings, norm='l2')

        # ---------------------------------------------------------
        # 4. COMPUTE 3D PCA FOR VISUALIZATION
        # ---------------------------------------------------------
        pca_3d_coordinates = PCA(n_components=3).fit_transform(normalized_embeddings)
        global_dataframe = pd.DataFrame(pca_3d_coordinates, columns=['x', 'y', 'z'])
        global_dataframe[DATASET_CONFIG['IMAGE_ID_COL']] = combined_image_names
        
        # Boolean mask for UI logic
        global_dataframe['is_labeled_set'] = global_dataframe[DATASET_CONFIG['IMAGE_ID_COL']].apply(
            lambda img_name: img_name in labeled_image_names
        )

        # ---------------------------------------------------------
        # 5. MERGE METADATA AND CREATE UNIFIED CATEGORY
        # ---------------------------------------------------------
        if metadata_dfs:
            combined_metadata_df = pd.concat(metadata_dfs, ignore_index=True)
            global_dataframe = global_dataframe.merge(
                combined_metadata_df, 
                on=DATASET_CONFIG['IMAGE_ID_COL'], 
                how='left'
            )
        
        # Fill NaN values resulting from the merge
        global_dataframe[DATASET_CONFIG['PREDICTION_COL']] = global_dataframe[DATASET_CONFIG['PREDICTION_COL']].fillna('Unknown')
        global_dataframe[DATASET_CONFIG['CATEGORY_COL']] = global_dataframe[DATASET_CONFIG['CATEGORY_COL']].fillna('Unknown')

        def assign_unified_category(row):
            """Helper function to clean up and standardize category names for the UI."""
            if row['is_labeled_set']:
                return 'Labeled Set'
            
            raw_category = str(row[DATASET_CONFIG['CATEGORY_COL']]).strip()
            if raw_category.lower() in ['nan', 'none', '']:
                return 'Unknown'
                
            return raw_category.title()
            
        global_dataframe['UnifiedCategory'] = global_dataframe.apply(assign_unified_category, axis=1)

        return normalized_embeddings, global_dataframe

    except Exception as e:
        print(f"CRITICAL ERROR in generic_utils.py: {e}")
        return np.array([]), pd.DataFrame()
    
# ==========================================
# SHARED UI COMPONENTS (Charts and Tables)
# ==========================================
def generate_3d_scatter_plot(df_plot, title="3D Latent Space"):
    """
    Takes a DataFrame and returns a standardized Plotly figure,
    dynamically reading column names from DATASET_CONFIG.
    """
    fig = px.scatter_3d(
        df_plot, 
        x='x', y='y', z='z', 
        color='Cluster',
        hover_name=DATASET_CONFIG['IMAGE_ID_COL'],
        custom_data=[DATASET_CONFIG['PREDICTION_COL'], DATASET_CONFIG['IMAGE_ID_COL']],
        title=title,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_traces(
        marker=dict(size=5, opacity=0.8),
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Label: %{customdata[0]}<br>"
            "<extra></extra>"
        )
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40), 
        scene=dict(xaxis_title='', yaxis_title='', zaxis_title='')
    )
    return fig

def generate_crosstab_table(df_plot):
    """
    Takes a DataFrame, calculates the crosstab using the configured
    prediction column, and returns a Dash Bootstrap Table.
    """
    target_column = DATASET_CONFIG['PREDICTION_COL']
    
    ct = pd.crosstab(df_plot[target_column], df_plot['Cluster'])
    table = dbc.Table.from_dataframe(
        ct.reset_index(), 
        striped=True, bordered=True, hover=True, responsive=True, size="sm"
    )
    return table

def get_hover_image_path(hover_data):
    """
    Extracts the image name from Plotly hover_data, checks which configured 
    folder it belongs to, and returns the proper Dash asset URL.
    
    NOTE: Users must ensure their image folders are placed inside the 
    'dashboard/assets/' directory and match the names in DATASET_CONFIG.
    """
    if hover_data is None:
        return "", "Move the mouse over a point."

    # Index 1 corresponds to IMAGE_ID_COL in the 3D plot's custom_data
    image_name = hover_data['points'][0]['customdata'][1]
    
    current_page_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_page_path)
    
    dir_labeled = DATASET_CONFIG['LABELED_IMAGES_DIR']
    dir_unlabeled = DATASET_CONFIG['UNLABELED_IMAGES_DIR']
    
    # Check if the file exists in the labeled folder
    labeled_check_path = os.path.join(project_root, "dashboard", "assets", dir_labeled, image_name)
    
    if os.path.exists(labeled_check_path):
        subfolder = dir_labeled
        status_tag = "Labeled"
    else:
        subfolder = dir_unlabeled
        status_tag = "Unlabeled"
        
    image_url = dash.get_asset_url(f"{subfolder}/{image_name}")
    return image_url, f"[{status_tag}] File: {image_name}"

# ==========================================
# INITIALIZATION ON LOAD (In-Memory Cache)
# ==========================================
# Executed once when the module is imported to keep the app fast.
GLOBAL_EMBEDDINGS, GLOBAL_DF = load_features_and_metadata()    