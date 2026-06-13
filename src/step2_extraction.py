"""
=============================================================================
UNIT 2: FEATURE EXTRACTION
=============================================================================
This module loads a custom trained ResNet18 model, removes the final 
classification layer, and extracts numerical embeddings (features) for each 
preprocessed image. The outputs are saved as a .npz file, ready to be 
consumed by the clustering dashboard (Unit 3).
=============================================================================
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from dotenv import dotenv_values

# --- CONFIGURATION ---
config = dotenv_values(".env")
DEFAULT_MODEL_PATH = config.get("FEATURE_EXTRACTOR_MODEL_PATH", "models/model.pt")
NUM_CLASSES = 6

def get_custom_extractor(model_path, device):
    """Loads the custom ResNet18 model and strips the classification head."""
    print(f"Loading Custom ResNet as feature extractor from: {model_path}...")
    
    # Initialize base ResNet18
    model = models.resnet18()
    
    # Adjust final layer to match the classes of the custom trained model
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # Handle cases where the saved dictionary has a 'model' key
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print("[✓] Model weights successfully loaded.")
    except Exception as e:
        print(f"[!] CRITICAL ERROR loading model weights: {e}")
        print("Please ensure the model weights are located in the correct directory.")
        return None
        
    # Replace the classification head with an Identity layer to extract embeddings
    model.fc = nn.Identity()
    
    model.to(device)
    model.eval()
    
    return model

def get_transforms():
    """Returns the exact transformations used during the custom model training."""
    WIDTH = 256
    HEIGHT = 512
    MEAN = [0.5414286851882935, 0.5396731495857239, 0.3529253602027893]
    STD = [0.2102500945329666, 0.23136012256145477, 0.19928686320781708]

    return transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def extract_features(input_folder, output_path, model_path, valid_names=None):
    """Iterates through images, extracts embeddings, and saves them to a .npz file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_custom_extractor(model_path, device)
    if model is None:
        return
        
    transform = get_transforms()
    
    features_list = []
    names_list = []
    
    print(f"\nStarting feature extraction for images in: {input_folder}")
    
    count = 0
    valid_extensions = (".jpg", ".png", ".jpeg")
    
    for file in sorted(os.listdir(input_folder)):
        if not file.lower().endswith(valid_extensions):
            continue

        if valid_names is not None and file not in valid_names:
            continue

        path = os.path.join(input_folder, file)

        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = model(tensor)
                
            # Move back to CPU and convert to numpy array
            feature_array = embedding.squeeze().cpu().numpy()
            
            features_list.append(feature_array)
            names_list.append(file)
            
            count += 1
            if count % 20 == 0:
                print(f"Extracted features for {count} images...")
                
        except Exception as e:
            print(f"[SKIP] Error processing {file}: {e}")

    if len(features_list) > 0:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to compressed numpy format
        np.savez(output_path, embeddings=np.array(features_list), names=np.array(names_list))
        print(f"\n[✓] Extraction completed successfully!")
        print(f"  - Extracted feature shape: {np.array(features_list).shape}")
        print(f"  - File saved to: {output_path}")
    else:
        print("\n[!] No images were processed. Ensure the input directory contains valid images.")

if __name__ == "__main__":
    # Setup Argument Parser for pipeline orchestration
    parser = argparse.ArgumentParser(description="Step 2: Feature Extraction using Custom ResNet18")
    parser.add_argument("--input", type=str, required=True, help="Path to the preprocessed images folder")
    parser.add_argument("--output_npz", type=str, required=True, help="Path to save the output .npz file (or directory)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to the custom model weights (.pt)")
    
    args = parser.parse_args()
    
    # Check if the output argument is a directory. If so, append a default filename.
    if os.path.isdir(args.output_npz) or not args.output_npz.endswith('.npz'):
        final_output_path = os.path.join(args.output_npz, "features_extracted.npz")
    else:
        final_output_path = args.output_npz
    
    print(f"Input Directory: {args.input}")
    print(f"Output File: {final_output_path}")
    
    extract_features(args.input, final_output_path, args.model)