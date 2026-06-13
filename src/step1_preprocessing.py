"""
=============================================================================
UNIT 1: SMART IMAGE PREPROCESSING
=============================================================================
This module uses a custom Faster R-CNN model to detect orchids in raw images.
It applies a dynamic bounding box, pads it, and crops the image.
If the primary detection fails, it falls back to a sliding window approach.
Output images are rotated (if horizontal), resized, and padded with a 
black background to a fixed target size (default 256x512).
=============================================================================
"""

import os
import torch
import shutil
import argparse
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from dotenv import dotenv_values

# --- CONFIGURATION ---
config = dotenv_values(".env")
DETECTION_MODEL_PATH = config.get("DETECTION_MODEL_PATH", "models/fasterrcnn_orchid3.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_cropping_model():
    """Loads the Faster R-CNN model for orchid detection."""
    print("Loading Faster R-CNN model...")
    try:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        num_classes = 2  # Background + Orchid
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        model.load_state_dict(torch.load(DETECTION_MODEL_PATH, map_location=device))
        model.to(device).eval()
        print(f"[✓] Cropping model successfully loaded on: {device}")
        return model
    except Exception as e:
        print(f"[!] ERROR LOADING DETECTOR MODEL: {e}")
        print("Please ensure the model weights are located in the 'models/' directory.")
        return None

def process_single_crop(image: Image.Image, model, target_size=(256, 512)):
    """
    Detects the flower, applies smart cropping, and resizes to target_size
    with a black background padding.
    """
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)[0]

    boxes = predictions['boxes'].cpu().numpy().astype(int).tolist()
    scores = predictions['scores'].cpu().numpy().tolist()
    
    x_min, y_min, x_max, y_max = 0, 0, 0, 0
    found_box = False

    if len(boxes) > 0:
        found_box = True

        if len(boxes) == 1:
            x_min, y_min, x_max, y_max = boxes[0]
        else:
            box1, box2 = boxes[0], boxes[1]
            score1, score2 = scores[0], scores[1]
            
            # Micro-tolerance to choose the largest bounding box if scores are nearly identical
            if abs(score1 - score2) < 0.01:
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                if area2 > area1:
                    x_min, y_min, x_max, y_max = box2
                else:
                    x_min, y_min, x_max, y_max = box1
            else:
                x_min, y_min, x_max, y_max = box1
    else:
        # --- START FALLBACK: SLIDING WINDOW ---
        # 1/6 of the total area
        win_w = max(1, image.width // 2)
        win_h = max(1, image.height // 3)
        
        # Step (stride) equal to half the window for overlap
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
                    # A flower was found in the sub-window
                    lx_min, ly_min, lx_max, ly_max = win_boxes[0]
                    
                    x_min = lx_min + x
                    y_min = ly_min + y
                    x_max = lx_max + x
                    y_max = ly_max + y
                    
                    found_box = True
            
            if found_box:
                break 
        # --- END FALLBACK ---

    if not found_box:
        return None

    # Add 10% padding around the detected bounding box
    pw = int((x_max - x_min) * 0.10)
    ph = int((y_max - y_min) * 0.10)
    crop_coords = (
        max(0, x_min - pw), max(0, y_min - ph),
        min(image.width, x_max + pw), min(image.height, y_max + ph)
    )
    
    cropped_img = image.crop(crop_coords)
    
    # Rotate if the format is horizontal
    if cropped_img.width > cropped_img.height:
        cropped_img = cropped_img.rotate(90, expand=True)

    # Resize while maintaining aspect ratio
    cropped_img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
    
    # Apply black padding to reach target_size strictly
    final_img = Image.new("RGB", target_size, (0, 0, 0))
    upper_left = (
        (target_size[0] - cropped_img.width) // 2,
        (target_size[1] - cropped_img.height) // 2
    )
    final_img.paste(cropped_img, upper_left)
    
    return final_img

def run_smart_cropping(input_folder, output_folder, discard_folder, valid_names=None):
    """Processes all images in the input folder and routes them based on detection success."""
    model = load_cropping_model()
    if model is None: 
        return

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(discard_folder, exist_ok=True)
    
    count_ok = 0
    count_fail = 0

    valid_extensions = (".jpg", ".png", ".jpeg")
    files_to_process = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(valid_extensions)]

    for file in files_to_process:
        if valid_names is not None and file not in valid_names: 
            continue

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

            total_processed = count_ok + count_fail
            if total_processed % 20 == 0:
                print(f"Processed {total_processed} files...")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"\n[✓] Preprocessing Completed!")
    print(f"  - Successfully cropped (512x256): {count_ok}")
    print(f"  - Not detected (moved to discard): {count_fail}")

if __name__ == "__main__":
    # Setup Argument Parser for pipeline orchestration
    parser = argparse.ArgumentParser(description="Step 1: Smart Image Cropping and Resizing")
    parser.add_argument("--input", type=str, required=True, help="Path to the raw images folder")
    parser.add_argument("--output", type=str, required=True, help="Path to save successfully cropped images")
    parser.add_argument("--discard", type=str, default="data/02_discarded", help="Path to save images where no flower was detected")
    
    args = parser.parse_args()
    
    print(f"Input Directory: {args.input}")
    print(f"Output Directory: {args.output}")
    print(f"Discard Directory: {args.discard}")
    
    run_smart_cropping(args.input, args.output, args.discard)