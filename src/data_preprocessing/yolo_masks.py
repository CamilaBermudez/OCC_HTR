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
import json
import logging
import subprocess

sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import fixed_file_naming

def setup_processing_logging(logs_dir, run_name=None, extra_config=None):
    """
    Setup logging for mask generation run
    Returns: (log_file_path, metrics_file_path, logger)
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_mask_generation.log"
    config_file = Path(logs_dir) / f"{run_name}_config.json"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # Reset any existing handlers
    )
    
    logger = logging.getLogger(__name__)
    
    return str(log_file), str(config_file), logger


def setup_processing_logging(logs_dir, run_name=None, extra_config=None):
    """
    Setup logging for mask generation run
    Returns: (log_file_path, metrics_file_path, logger)
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_mask_generation.log"
    config_file = Path(logs_dir) / f"{run_name}_config.json"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # Reset any existing handlers
    )
    
    logger = logging.getLogger(__name__)
    
    return str(log_file), str(config_file), logger



def build_mask_yolo(model_path, images_path, output_path, logs_dir=None, run_name=None):
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"mask_gen_{timestamp}"
    
    # Setup logging if requested
    if logs_dir:
        log_file, config_file, logger = setup_processing_logging(logs_dir, run_name)
        logger.info(f"=== Mask Generation Run Started ===")
        logger.info(f"Run name: {run_name}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Config file: {config_file}")
    else:
        logger = None
    
    model_name = Path(model_path).stem
    model_info = {
        "model_path": model_path,
        "model_name": model_name,
        "model_file_size_mb": round(Path(model_path).stat().st_size / (1024*1024), 2) if Path(model_path).exists() else None,
        "timestamp": timestamp,
        "images_path": str(images_path),
        "output_path": str(output_path),
        "imgsz": 640,
        "target_class": "MainZone",
        "environment": {
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT"),
            "SEGMONTO_LIST": os.environ.get("SEGMONTO_LIST")
        }
    }
    
    try:
        model_info["git_commit"] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.environ.get("PROJECT_ROOT", ".")
        ).decode('ascii').strip()
    except:
        model_info["git_commit"] = None
    
    if logger:
        logger.info(f"Model Configuration: {json.dumps(model_info, indent=2)}")
    
    # Load model
    if logger:
        logger.info(f"Loading YOLO model: {model_name}")

    model = YOLO(model_path)
    
    masks_dir = os.path.join(output_path, "masks", timestamp)
    os.makedirs(masks_dir, exist_ok=True)
    
    images_output_dir = os.path.join(output_path, "images")
    os.makedirs(images_output_dir, exist_ok=True)

    if logger:
        logger.info(f"Masks output directory: {masks_dir}")
        logger.info(f"Images output directory: {images_output_dir}")
    

    load_dotenv()
    SEGMONTO_LIST = ast.literal_eval(os.environ.get("SEGMONTO_LIST"))
    target_idx = SEGMONTO_LIST.index("MainZone")

    if logger:
        logger.info(f"Target class: 'MainZone' (index: {target_idx})")
        logger.info(f"Starting prediction on: {images_path}")

    results = model.predict(
        images_path,
        imgsz=640,
        save=True,
        project=images_output_dir,
        name=f'predict_run-{timestamp}',
        exist_ok=True,
        verbose=False
    )

    total_images = len(results)
    processed_count = 0
    no_detections_count = 0
    mainzone_found_count = 0
    
    for result in tqdm(results, desc="Processing masks"):
        img_path = result.path  
        img_name = fixed_file_naming(Path(img_path).stem)
        mask_full_path = os.path.join(masks_dir, f"{img_name}.png")
        
        h, w = result.orig_shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        if result.boxes is None or len(result.boxes) == 0:
            cv2.imwrite(mask_full_path, mask)
            no_detections_count += 1
            if logger and processed_count % 50 == 0:
                logger.debug(f"No detections: {img_name}")
            processed_count += 1
            continue
        
        cls_mask = result.boxes.cls == target_idx
        if not cls_mask.any():
            cv2.imwrite(mask_full_path, mask)
            no_detections_count += 1
            processed_count += 1
            continue

        zones = result.boxes.xyxy[cls_mask]
        zones_np = zones.cpu().numpy()
        
        for x1, y1, x2, y2 in zones_np:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
        
        cv2.imwrite(mask_full_path, mask)
        mainzone_found_count += 1
        processed_count += 1
        
        # Log progress periodically
        if logger and processed_count % 100 == 0:
            logger.info(f"Progress: {processed_count}/{total_images} images processed")
    
    summary = {
        "total_images": total_images,
        "processed_successfully": processed_count,
        "images_with_mainzone": mainzone_found_count,
        "images_no_detections": no_detections_count,
        "masks_saved_to": masks_dir,
        "masks_format": "PNG (0=MainZone, 255=background)"
    }
    
    if logger:
        logger.info("=== Processing Summary ===")
        logger.info(f"Total images: {summary['total_images']}")
        logger.info(f"Images with MainZone detected: {summary['images_with_mainzone']}")
        logger.info(f"Images with no detections: {summary['images_no_detections']}")
        logger.info(f"Masks saved to: {summary['masks_saved_to']}")
        logger.info(f"=== Mask Generation Run Completed ===")
    
    
    if logger and config_file:
        output_config = {**model_info, "processing_summary": summary}
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(output_config, f, indent=2, default=str)
        logger.info(f"Configuration and summary saved to: {config_file}")
    
    return {
        "masks_dir": masks_dir,
        "summary": summary,
        "config_file": config_file if logger else None
    }


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "layout", "y8_YALTAi_15_best_+9annotated_fix50.pt")
    IMAGES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "original_manuscript", "reproduction14453_100")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "img_layout")

    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "mask_generation")  # NEW
    RUN_NAME = f"mask_{Path(IMAGES_PATH).name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"  # NEW


    result = build_mask_yolo(model_path=MODEL_PATH, images_path=IMAGES_PATH, output_path=OUTPUT_PATH,
        logs_dir=LOGS_DIR,run_name=RUN_NAME)
    
    print(f"Masks generated: {result['summary']['images_with_mainzone']}/{result['summary']['total_images']} images")
    print(f"Masks saved to: {result['masks_dir']}")
    if result['config_file']:
        print(f"Config logged to: {result['config_file']}")