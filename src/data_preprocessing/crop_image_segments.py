import os
import sys
import numpy as np
import logging
import json
import subprocess
from pathlib import Path
from PIL import Image
import datetime
from tqdm import tqdm  
from dotenv import load_dotenv
from typing import Union, Optional

sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import format_filename

def setup_simple_logging(logs_dir: str, run_name: Optional[str] = None):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_cropping.log"
    
    logger = logging.getLogger("cropping")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File + console handlers (same format)
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    
    return logger, str(log_file)

def crop_image_segments( img_path: Path, kraken_output_path: Path, processed_name: str, output_path: Path) -> bool:
    try:
        img = Image.open(img_path).convert("RGB")

        with open(kraken_output_path, "r", encoding="utf-8") as f:
            kraken_output = json.load(f)

        for id, line in enumerate(kraken_output.get("lines", [])):
            if "boundary" not in line:
                continue

            xs, ys = zip(*line["boundary"])
            left, upper = int(min(xs)), int(min(ys))
            right, lower = int(max(xs)), int(max(ys))
            img.crop((left, upper, right, lower)).save(f"{output_path}/{processed_name}_line_{id}.png", optimize=True)
        return True
    except Exception:
        return False


def crop_all_images(input_folder: Union[str, Path],output_kraken_path: Union[str, Path],output_folder: Union[str, Path],logs_dir: Optional[str] = None,run_name: Optional[str] = None) -> dict:
    nput_folder = Path(input_folder)
    output_kraken_path = Path(output_kraken_path)
    output_folder = Path(output_folder)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if run_name is None:
        run_name = f"crop_{timestamp}"

    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Cropping Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("cropping")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        log_file = None

    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],  # Short hash only
            cwd=os.environ.get("PROJECT_ROOT", "."),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        git_commit = "unknown"

    config_summary = {"run": run_name,"git": git_commit,"input": str(input_folder.name),"output": str(output_folder.name),"images_count": None}
    
    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return {"success": False}

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions])

    config_summary["images_count"] = len(image_files)
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")
    
    logger.info(f"Starting cropping for {len(image_files)} images")
    run_output_folder = output_folder / f"extraction_{timestamp}"
    run_output_folder.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for idx, img_path in enumerate(tqdm(image_files, desc="Cropping", unit="file"), 1):
        output_path_str, _, processed_name = format_filename(base_name=img_path.stem, output_folder=output_kraken_path)
        json_path = Path(output_path_str)
        file_output_dir = run_output_folder / processed_name
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not json_path.exists():
            continue
        
        if crop_image_segments(img_path, json_path, processed_name, file_output_dir):
            success_count += 1
        
        if idx % 20 == 0 or idx == len(image_files):
            logger.info(f"Progress: {idx}/{len(image_files)} | OK: {success_count}")
    
    logger.info(f"Cropping complete: {success_count}/{len(image_files)} succeeded")
    logger.info(f"Output: {run_output_folder}")
    if log_file:
        logger.info(f"📄 Log file: {log_file}")
    
    return {"total": len(image_files), "success": success_count, "output": str(run_output_folder)}

if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    input_folder = project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    output_kraken_path = project_root / "data" / "processed" / "segmented_images"  / "segmentation_20260427_181912"
    output_folder = project_root / "data" / "processed" / "extracted_lines"

    logs_dir = project_root / "logs" / "cropping"
    run_name = f"crop_{input_folder.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    result = crop_all_images(input_folder = input_folder,output_kraken_path = output_kraken_path,output_folder=output_folder,logs_dir=str(logs_dir),run_name=run_name)

    print(f"\n Done: {result['success']}/{result['total']} images | Output: {result['output']}\n")
