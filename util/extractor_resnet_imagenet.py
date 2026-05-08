import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# ------------------------
# LOAD MODEL
# ------------------------
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
   
    model.fc = nn.Identity()
    return model

# ------------------------
# IMAGE TRANSFORM
# ------------------------
def get_transforms():
    # trasformazioni standard per ImageNet
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ------------------------
# EXTRACT FEATURES
# ------------------------
def run_extraction(folder, model, transform, valid_names=None):
    embeddings   = []
    image_names  = []

    for file in sorted(os.listdir(folder)):  # sorted per riproducibilità
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        if valid_names is not None and file not in valid_names:
            continue

        path = os.path.join(folder, file)

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Immagine non leggibile: {file} — {e}")
            continue

        tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = model(tensor)            # (1, 512) grazie a Identity
        
        embeddings.append(features.squeeze().numpy())
        image_names.append(file)

    return np.array(embeddings), image_names