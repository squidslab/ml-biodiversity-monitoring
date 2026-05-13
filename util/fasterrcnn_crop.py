import os
import torch
import shutil
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

def process_single_crop(image: Image.Image, model, target_size=(256, 512), device='cpu'):
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions['boxes'].cpu().numpy().astype(int).tolist()
    
    x_min, y_min, x_max, y_max = 0, 0, 0, 0
    found_box = False

    if len(boxes) > 0:
        x_min, y_min, x_max, y_max = boxes[0]
        found_box = True
    else:
        # --- INIZIO FALLBACK: SLIDING WINDOW ---
        # 1/6 dell'area totale
        win_w = max(1, image.width // 2)
        win_h = max(1, image.height // 3)
        
        # step (stride) pari alla metà della finestra per sovrapposizione
        step_x = max(1, win_w // 2)
        step_y = max(1, win_h // 2)

        for y in range(0, image.height - win_h + 1, step_y):
            for x in range(0, image.width - win_w + 1, step_x):
                window_crop = image.crop((x, y, x + win_w, y + win_h))
                win_tensor = F.to_tensor(window_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    win_preds = model(win_tensor)[0]

                win_boxes = win_preds['boxes'].cpu().numpy().astype(int).tolist()
                
                if len(win_boxes) > 0:
                    # è stato trovato un fiore nella sottofinestra
                    lx_min, ly_min, lx_max, ly_max = win_boxes[0]
                    
                    x_min = lx_min + x
                    y_min = ly_min + y
                    x_max = lx_max + x
                    y_max = ly_max + y
                    
                    found_box = True
            
            if found_box:
                break 
        # --- FINE FALLBACK ---

    if not found_box:
        return None

    pw = int((x_max - x_min) * 0.10)
    ph = int((y_max - y_min) * 0.10)
    crop_coords = (
        max(0, x_min - pw), max(0, y_min - ph),
        min(image.width, x_max + pw), min(image.height, y_max + ph)
    )
    
    cropped_img = image.crop(crop_coords)
    
    # Ruota se il formato è orizzontale
    if cropped_img.width > cropped_img.height:
        cropped_img = cropped_img.rotate(90, expand=True)

    # Ridimensiona mantenendo l'aspect ratio
    cropped_img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
    
    # Applica padding nero per raggiungere target_size
    final_img = Image.new("RGB", target_size, (0, 0, 0))
    upper_left = (
        (target_size[0] - cropped_img.width) // 2,
        (target_size[1] - cropped_img.height) // 2
    )
    final_img.paste(cropped_img, upper_left)
    
    return final_img

def run_smart_cropping(input_folder, output_folder, discard_folder, valid_names=None):
    model = load_cropping_model()
    if model is None: return

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(discard_folder, exist_ok=True)
    
    count_ok = 0
    count_fail = 0

    for file in sorted(os.listdir(input_folder)):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")): continue
        if valid_names is not None and file not in valid_names: continue

        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        fail_path = os.path.join(discard_folder, file)

        try:
            img = Image.open(in_path).convert("RGB")
            result_img = process_single_crop(img, model)

            if result_img is not None:
                result_img.save(out_path)
                count_ok += 1
            else:
                shutil.copy(in_path, fail_path)
                count_fail += 1

            if (count_ok + count_fail) % 20 == 0:
                print(f"Processati {count_ok + count_fail} file...")

        except Exception as e:
            print(f"Errore su {file}: {e}")

    print(f"\n[✓] Completato!")
    print(f"- Immagini ritagliate (512x256): {count_ok}")
    print(f"- Immagini non rilevate (spostate): {count_fail}")