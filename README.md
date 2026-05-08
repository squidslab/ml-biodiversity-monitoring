# Orchid Clustering & Visual Explorer

This project provides a complete pipeline for analyzing orchid images using Deep Learning features and unsupervised clustering (DBSCAN). It includes a set of utility scripts for data preparation and a web-based dashboard for interactive exploration of the feature space.

---

## 📂 Project Structure

*   **`util/`**: Contains utility scripts for data management:
    *   Excel database manipulation and cleaning.
    *   Feature extraction (generating `.npz` files from image datasets).
    *   DBSCAN testing and parameter fine-tuning.
*   **`dashboard/`**: The core interactive application:
    *   `app.py`: The main Dash application.
    *   `assets/`: UI elements (logos) and image previews.
    *   `dataset.xlsx`: The Excel database with every image documentation.

> **Note**: Due to size constraints, the `.jpg` image dataset and the `.npz` feature vector files are ignored by git. They can be provided upon request.

---

## 🚀 Getting Started

Follow these steps to run the Visual Explorer on your local machine.

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
### 2. Installation
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Data Setup
Since large files are not included in the repository:

*   Request the features vector file and the image dataset.
*   Place the `.npz` file inside the `dashboard/` folder.
*   Place the orchid images inside the `dashboard/assets/` folder.

### 4. Running the Dashboard
Navigate to the application folder and start the server:

```bash
cd dashboard
python app.py
```
Once the script is running, open your browser and go to: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

## 🛠 Features

*   **Filtering**: Filter data through a list of categories (Curated, Usable, Hardcore, etc.).
*   **Dynamic Clustering**: Adjust DBSCAN parameters (`EPS`, `Min Samples`) in real-time and see results instantly.
*   **3D Visualization**: Explore the feature space in an interactive 3D scatter plot with image preview on hover.
*   **Contingency Table**: Real-time cross-tabulation (Species vs. Cluster ID) to evaluate clustering performance dynamically.