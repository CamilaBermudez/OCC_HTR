import xml.etree.ElementTree as ET
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import os
import ast
import datetime
from dotenv import load_dotenv
import sys

sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import fixed_file_naming


def build_mask_yolo(model_path, images_path, output_path):
    model = YOLO(model_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    masks_dir = os.path.join(output_path, "masks", timestamp)
    os.makedirs(masks_dir, exist_ok=True)
    
    images_output_dir = os.path.join(output_path, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    load_dotenv()
    SEGMONTO_LIST = ast.literal_eval(os.environ.get("SEGMONTO_LIST"))
    target_idx = SEGMONTO_LIST.index("MainZone")

    results = model.predict(
        images_path,
        imgsz=640,
        save=True,
        project=images_output_dir,
        name=f'predict_run-{timestamp}',
        exist_ok=True
    )

    for result in tqdm(results, desc="Processing masks"):
        img_path = result.path  
        img_name = fixed_file_naming(Path(img_path).stem)
        mask_full_path = os.path.join(masks_dir, f"{img_name}.png")
        
        h, w = result.orig_shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        if result.boxes is None or len(result.boxes) == 0:
            cv2.imwrite(mask_full_path, mask)
            continue
        
        cls_mask = result.boxes.cls == target_idx
        if not cls_mask.any():
            cv2.imwrite(mask_full_path, mask)
            continue

        zones = result.boxes.xyxy[cls_mask]
        zones_np = zones.cpu().numpy()
        
        for x1, y1, x2, y2 in zones_np:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
        
        cv2.imwrite(mask_full_path, mask)


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "layout", "y8_YALTAi_15_best_annotated.pt")
    IMAGES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "original_manuscript", "reproduction14453_100")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "img_layout")

    build_mask_yolo(model_path=MODEL_PATH, images_path=IMAGES_PATH, output_path=OUTPUT_PATH)