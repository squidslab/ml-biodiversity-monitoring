import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import os
import numpy as np

# ------------------------
# LOAD MODEL
# ------------------------
def get_model():
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    
    model.heads = nn.Identity()
    model.eval()
    return model

# ------------------------
# IMAGE TRANSFORM
# ------------------------
def get_transforms():
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    return weights.transforms()

# ------------------------
# EXTRACT FEATURES
# ------------------------
def run_extraction(folder, model, transform, valid_names=None):
    embeddings = []
    image_names = []

    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
            
        if valid_names is not None and file not in valid_names:
            continue

        path = os.path.join(folder, file)
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                feat = model(tensor)
            
            embeddings.append(feat.squeeze().numpy())
            image_names.append(file)
            
        except Exception as e:
            print(f"[SKIP] Errore su {file}: {e}")

    return np.array(embeddings), image_names