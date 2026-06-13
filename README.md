# OrchiData: Ophrys Sphegodes Clustering and Analysis Pipeline

This repository contains a complete, end-to-end Machine Learning pipeline designed to assist botanical researchers in categorizing and clustering specimen images. 

The pipeline is highly modular by design. You can run the full process, or easily swap out the preprocessing/feature extraction models with your own, as long as you respect the input/output folder structures.

---

## 📂 Project Structure

~~~text
orchID_project/
├── data/                       # Dataset directories 
│   ├── 00_metadata/            # Place your .xlsx files here 
│   ├── 01_raw/                 
   │   │   ├── unlabeled/          # Raw images of unknown species
   │   │   └── labeled/            # Raw images of known species 
│   ├── 02_preprocessed/        # Output of Unit 1 (cropped images)
   │   │   ├── unlabeled/          # Cropped unlabeled images
   │   │   └── labeled/            # Cropped labeled images
│   └── 03_features/            # Output of Unit 2 (.npz embeddings)
│       ├── features_unlabeled.npz
│       └── features_labeled.npz
├── src/                        # Pipeline source code
│   ├── step1_preprocessing.py  
│   └── step2_extraction.py     
├── dashboard/                  # Interactive Plotly/Dash Web UI
│   ├── app.py
│   ├── utils.py                # UI data loading and configuration
│   ├── pages/                  # Clustering algorithm pages
│   └── assets/                 # CSS and image previews
├── run_pipeline.py             # Main orchestrator script
└── requirements.txt            # Python dependencies
~~~

---

## ⚙️ 1. Setup & Installation

Before preparing your data, clone the repository and set up your Python environment:

~~~bash
git clone <repository_url>
cd orchID_project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
~~~

---

## 💾 2. Data Preparation

**💡 Note on Provided Artifacts:** If you have requested and received access to our proprietary *Ophrys sphegodes* datasets and model weights (see *Data and Code Availability*), you can use them as a ready-to-go baseline. Because the system processes **Unlabeled Data** (for discovery) and **Labeled Data** (for tool calibration and benchmarking) independently, you have complete flexibility:
* **Test out of the box:** Run the pipeline or launch the dashboard immediately using our provided datasets.
* **Replace both:** Clear the folders and supply your own images and metadata for a completely custom analysis.
* **Replace only one:** Mix and match. For example, you can supply your own unlabeled images to discover new clusters, while keeping our provided labeled set to calibrate the algorithms.

If you choose to use your own dataset(s), you must place your raw images and metadata (Excel files) in the correct folders:

### 🖼️ Images
* **Unlabeled Images:** Place your raw `.jpg` or `.png` files in `data/01_raw/unlabeled/`.
* **Labeled Images (Ground Truth):** Place the images of known species in `data/01_raw/labeled/`.

### 📊 Metadata (Excel files)
Place your `.xlsx` files in the `data/00_metadata/` folder. The dashboard relies on these files to display image names, filter datasets, and calculate validation metrics.

* **For the LABELED SET (Required for tool calibration):**
  Provide an Excel file containing at least two columns:
  1. **Image Name:** The exact filename of the image (e.g., `Ophrys_001.jpg`).
  2. **Species / Prediction:** The actual ground-truth class. The dashboard needs this to calculate metrics like AMI, ARI, and FMI.

* **For the UNLABELED SET (Optional but recommended):**
  Provide an Excel file to enable data filtering in the dashboard:
  1. **Image Name:** The exact filename.
  2. **Category:** A custom category (e.g., *Curated*, *Usable*). This allows you to filter out noisy data dynamically within the 3D visualizer.

*(Note: The exact column names for "Image Name", "Species", and "Category" can be configured in `dashboard/utils.py`).*

---

## 🚀 3. How to Run the Pipeline

Once your data is in place, you can run the project in two ways:

### Option A: Full End-to-End Execution
If you added new raw images, the orchestrator script will process the unlabeled and labeled folders separately to avoid mixing them up. 

~~~bash
python run_pipeline.py
~~~
This script will:
1. Run Preprocessing (crop & resize) on both raw folders.
2. Run Feature Extraction to generate `.npz` embeddings in `data/03_features/`.
3. Automatically copy the preprocessed images to `dashboard/assets/` to enable hover previews in the UI.

### Option B: Launching the Dashboard 
This is the final step of the pipeline. You will run this after completing Option A to visualize your results. 
Alternatively, if you are just testing the requested datasets (which are already processed) or already have your `.npz` and `.xlsx` files prepared, you can skip Option A entirely and launch the interface directly:

~~~bash
cd dashboard
python app.py
~~~

To connect your specific `.npz` and `.xlsx` files to the interface, open `dashboard/utils.py` and modify the `DATASET_CONFIG` dictionary. You can operate the dashboard in three modes:
1. **Hybrid Mode:** Provide paths for both Labeled and Unlabeled files.
2. **Unlabeled Only:** Provide new paths only for the Unlabeled files.
3. **Labeled Only:** Provide new paths only for the Labeled files.

---

## 🛠️ 4. Advanced Customization & Modularity

The OrchiData pipeline is built to be highly adaptable to your specific research needs. You do not have to use our specific Object Detection (Preprocessing) or ResNet (Extraction) models. You can easily swap one or both units by respecting the **I/O Contracts**:

* **Customizing Preprocessing (Unit 1):** Write a script that takes images from `data/01_raw/` and saves the processed/aligned images to `data/02_preprocessed/`. The rest of the pipeline won't notice the difference.
  
* **Customizing Feature Extraction (Unit 2):** If you want to use a Vision Transformer, CLIP, or a different embedding model, write a script that reads images from `data/02_preprocessed/` and outputs a `.npz` file into `data/03_features/`. 
  *The `.npz` file MUST contain two arrays:*
  * `embeddings`: A 2D numpy array of shape `(N, Feature_Dimension)`.
  * `names`: A 1D numpy array of strings containing the exact filenames corresponding to the embeddings.

---

## 📜 5. Data and Code Availability

The source code of the pipeline and dashboard is freely available in this repository. Due to academic restrictions, the *Ophrys sphegodes* dataset and the custom trained models (`fasterrcnn_orchid3.pth`, `model.pt`) are not included here. However, they are available upon request. Please contact the author directly for access.