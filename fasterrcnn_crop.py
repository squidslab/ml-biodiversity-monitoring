import os
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from dotenv import dotenv_values

# --- CONFIGURAZIONE ---
config = dotenv_values(".env")
DETECTION_MODEL_PATH = config.get("DETECTION_MODEL_PATH", "models/fasterrcnn_orchid3.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cropping_model():
    print("Caricamento modello Faster R-CNN...")
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2 # Background + Orchid
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=device))
        model.to(device).eval()
        print(f"[✓] Modello di cropping caricato con successo su: {device}")
        return model
    except Exception as e:
        print(f"[!] ERRORE CARICAMENTO MODELLO DETECTOR: {e}")
        return None

def process_single_crop(image: Image.Image, model):
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    scores = predictions['scores'].cpu().numpy().tolist()
    boxes = predictions['boxes'].cpu().numpy().astype(int).tolist()
    
    cropped_img = image # Fallback nel caso in cui non trovi nulla

    if len(boxes) > 0:
        x_min, y_min, x_max, y_max = boxes[0]
        
        # Padding del 10%
        padding_w = int((x_max - x_min) * 0.10)
        padding_h = int((y_max - y_min) * 0.10)
        
        crop_coords = (
            max(0, x_min - padding_w),
            max(0, y_min - padding_h),
            min(image.width, x_max + padding_w),
            min(image.height, y_max + padding_h)
        )
        cropped_img = image.crop(crop_coords)
        
    return cropped_img

def run_smart_cropping(input_folder, output_folder, valid_names=None):
    model = load_cropping_model()
    if model is None:
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"\nInizio ritaglio intelligente...")
    print(f"Cartella destinazione: {output_folder}")
    
    count = 0
    for file in sorted(os.listdir(input_folder)):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        if valid_names is not None and file not in valid_names:
            continue

        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)

        # Se l'immagine è già stata ritagliata in passato, saltala per risparmiare tempo
        if os.path.exists(out_path):
            continue

        try:
            img = Image.open(in_path).convert("RGB")
            cropped_img = process_single_crop(img, model)
            cropped_img.save(out_path)
            count += 1
            
            if count % 20 == 0:
                print(f"Ritagliate {count} immagini...")
        except Exception as e:
            print(f"[SKIP] Errore su {file}: {e}")

    print(f"\n[✓] Fase 0 completata! Elaborate {count} nuove immagini.")