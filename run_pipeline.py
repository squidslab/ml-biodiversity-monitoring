import subprocess
import sys
import shutil
import os

def run_step(command, description):
    """Executes a shell command and stops the pipeline if an error occurs."""
    print(f"\n🚀 Starting: {description}")
    print("-" * 50)
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"❌ ERROR during: {description}")
        sys.exit(1)
        
    print(f"✅ Completed: {description}")

def copy_assets_to_dashboard():
    """Copies the preprocessed images to the dashboard's assets folder."""
    print("\n📦 Preparing Assets for the Dashboard...")
    
    folders_to_copy = {
        "data/02_preprocessed/unlabeled": "dashboard/assets/unlabeled_images",
        "data/02_preprocessed/labeled": "dashboard/assets/labeled_images"
    }
    
    for source_dir, dest_dir in folders_to_copy.items():
        if os.path.exists(source_dir) and os.listdir(source_dir):
            try:
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                shutil.copytree(source_dir, dest_dir)
                print(f"✅ Images successfully copied from {source_dir} to {dest_dir}")
            except Exception as e:
                print(f"⚠️ WARNING: Could not copy images to {dest_dir}. Error: {e}")
        else:
            print(f"ℹ️ SKIP: Source folder {source_dir} is empty or missing.")

def main():
    print("=========================================")
    print("       ORCHID PIPELINE INITIALIZATION    ")
    print("=========================================")
    
    # --- UNLABELED DATA PIPELINE ---
    if os.path.exists("data/01_raw/unlabeled") and os.listdir("data/01_raw/unlabeled"):
        run_step(
            ["python", "src/step1_preprocessing.py", "--input", "data/01_raw/unlabeled", "--output", "data/02_preprocessed/unlabeled", "--discard", "data/02_discarded/unlabeled"],
            "[UNLABELED] Preprocessing (Cropping & Resizing)"
        )
        run_step(
            ["python", "src/step2_extraction.py", "--input", "data/02_preprocessed/unlabeled", "--output_npz", "data/03_features/features_unlabeled.npz"],
            "[UNLABELED] Feature Extraction"
        )
    else:
        print("\nℹ️ No Unlabeled data found in data/01_raw/unlabeled. Skipping.")

    # --- LABELED DATA PIPELINE ---
    if os.path.exists("data/01_raw/labeled") and os.listdir("data/01_raw/labeled"):
        run_step(
            ["python", "src/step1_preprocessing.py", "--input", "data/01_raw/labeled", "--output", "data/02_preprocessed/labeled", "--discard", "data/02_discarded/labeled"],
            "[LABELED] Preprocessing (Cropping & Resizing)"
        )
        run_step(
            ["python", "src/step2_extraction.py", "--input", "data/02_preprocessed/labeled", "--output_npz", "data/03_features/features_labeled.npz"],
            "[LABELED] Feature Extraction"
        )
    else:
        print("\nℹ️ No Labeled data found in data/01_raw/labeled. Skipping.")
    
    # --- MOVE ASSETS TO DASHBOARD ---
    copy_assets_to_dashboard()
    
    print("\n🎉 Pipeline completed successfully!")
    print("To start the interface, run the following commands:")
    print("  cd dashboard")
    print("  python app.py\n")

if __name__ == "__main__":
    main()