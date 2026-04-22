import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd

# --- CONFIGURAZIONE ---
MODEL_PATH = "models/model.pt"
NUM_CLASSES = 6
CLASS_NAMES = [
    'O. exaltata', 'O. garganica', 'O. incubacea',
    'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena'
]

def get_model():
    print(f"\nCaricamento modello di classificazione da: {MODEL_PATH}...")
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        print(f"[!] Errore critico nel caricamento dei pesi: {e}")
        return None
        
    model.eval()
    return model

def get_transforms():
    # Valori estratti da config originale
    WIDTH = 256
    HEIGHT = 512
    MEAN = [0.5414286851882935, 0.5396731495857239, 0.3529253602027893]
    STD = [0.2102500945329666, 0.23136012256145477, 0.19928686320781708]

    return transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def run_classification(folder, model, transform, valid_names, output_xlsx, excel_path):
    results = []
    print(f"Inizio classificazione sulle immagini in: {folder}")

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
                logits = model(tensor)
                probabilities = F.softmax(logits, dim=1)[0]
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            results.append({
                "Nome File": file,
                "Specie Predetta": CLASS_NAMES[predicted_idx.item()],
                "Confidenza (%)": round(confidence.item() * 100, 2)
            })
        except Exception as e:
            print(f"[SKIP] Errore su {file}: {e}")

    if results:
        df_results = pd.DataFrame(results)
        try:
            print("Integrazione con i metadati originali in corso...")
            df_original = pd.read_excel(excel_path)
            
            df_merged = pd.merge(
                df_original, 
                df_results, 
                left_on="image name", 
                right_on="Nome File", 
                how="inner"
            )
            
            df_merged = df_merged.drop(columns=["Nome File"])
            df_merged.to_excel(output_xlsx, index=False)
            print(f"[✓] Classificazione completata! File salvato: {output_xlsx}")
            
        except Exception as e:
            print(f"[!] Errore unione: {e}")
            df_results.to_excel(output_xlsx, index=False)
            print(f"[✓] Salvati solo risultati base in: {output_xlsx}")
            
    else:
        print("[!] Nessun risultato da salvare.")