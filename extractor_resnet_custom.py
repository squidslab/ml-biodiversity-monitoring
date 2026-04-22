import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

MODEL_PATH = "models/model.pt"
NUM_CLASSES = 6

def get_custom_extractor():
    print(f"Caricamento ResNet Custom come estrattore da: {MODEL_PATH}...")
    model = models.resnet18()
    
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        print(f"[!] Errore critico nel caricamento dei pesi: {e}")
        return None
        
    model.fc = nn.Identity()
    
    model.eval()
    return model

def get_transforms():
    WIDTH = 256
    HEIGHT = 512
    MEAN = [0.5414286851882935, 0.5396731495857239, 0.3529253602027893]
    STD = [0.2102500945329666, 0.23136012256145477, 0.19928686320781708]

    return transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def extract_features(folder, valid_names, output_npz):
    model = get_custom_extractor()
    if model is None:
        return
        
    transform = get_transforms()
    
    features_list = []
    names_list = []
    
    print(f"Inizio estrazione feature sulle immagini in: {folder}")
    
    count = 0
    
    for file in sorted(os.listdir(folder)):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        if valid_names is not None and file not in valid_names:
            continue

        path = os.path.join(folder, file)

        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                embedding = model(tensor)
                
            feature_array = embedding.squeeze().numpy()
            
            features_list.append(feature_array)
            names_list.append(file)
            
            count += 1
            if count % 20 == 0:
                print(f"Estratte feature per {count} immagini...")
                
        except Exception as e:
            print(f"[SKIP] Errore su {file}: {e}")

    if len(features_list) > 0:
        np.savez(output_npz, embeddings=np.array(features_list), names=np.array(names_list))
        print(f"\n[✓] Estrazione completata! File salvato: {output_npz}")
        print(f"Formato feature estratte: {np.array(features_list).shape}")
    else:
        print("[!] Nessuna immagine processata.")