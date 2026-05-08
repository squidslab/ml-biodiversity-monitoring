import os
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
import cv2
import numpy as np
from dotenv import dotenv_values

# --- CONFIGURAZIONE ---
config = dotenv_values(".env")
DETECTION_MODEL_PATH = config.get("DETECTION_MODEL_PATH", "models/fasterrcnn_orchid3.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CARTELLA_TEST_ORIGINALI = "C:\\Users\\aless\\OneDrive - Università di Napoli Federico II\\OrchID\\cropper_fallito" 
CARTELLA_OUTPUT_DIAGNOSTICA = "risultati_diagnostica"
# ----------------------

def load_cropping_model():
    print("Caricamento modello Faster R-CNN per Diagnostica...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

def analizza_e_disegna(image_path, model, output_path):
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions['boxes'].cpu().numpy().astype(int)
    scores = predictions['scores'].cpu().numpy()

    print(f"\nAnalisi di {os.path.basename(image_path)}:")
    print(f"Trovati {len(boxes)} potenziali bounding box.")

    # Disegna tutti i box 
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < 0.10: 
            continue
            
        x_min, y_min, x_max, y_max = box
        
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        thickness = 3 if i == 0 else 1
        
        cv2.rectangle(img_cv2, (x_min, y_min), (x_max, y_max), color, thickness)
        
        testo = f"Conf: {score:.2f}"
        cv2.putText(img_cv2, testo, (x_min, max(20, y_min - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        print(f"  Box {i+1}: Confidenza {score:.2f} -> Coordinate: {box}")

    img_final_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    img_final_pil.save(output_path) 

def main():
    os.makedirs(CARTELLA_OUTPUT_DIAGNOSTICA, exist_ok=True)
    model = load_cropping_model()
    
    for file in os.listdir(CARTELLA_TEST_ORIGINALI):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")): continue
        
        in_path = os.path.join(CARTELLA_TEST_ORIGINALI, file)
        out_path = os.path.join(CARTELLA_OUTPUT_DIAGNOSTICA, "diagnostica_" + file)
        
        analizza_e_disegna(in_path, model, out_path)
        
    print("\n[✓] Diagnostica completata! Controlla la cartella di output.")

if __name__ == "__main__":
    main()