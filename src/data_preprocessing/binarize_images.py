import os
import cv2 as cv
import numpy as np
import logging
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys
import datetime
from dotenv import load_dotenv
from typing import Tuple, Union, List, Optional
from argparse import ArgumentParser

load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from src.utils.path_utils import format_filename


def setup_simple_logging(logs_dir: str, run_name: Optional[str] = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_binarization.log"
    
    logger = logging.getLogger("binarization")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    
    return logger, str(log_file)


def binarize_image(input_img: Union[str, Path], gaussian_filter: Tuple[int, int] = (5, 5), method: str = "otsu_gaussian") -> Tuple[float, np.ndarray]:
    img = cv.imread(str(input_img), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {input_img}")
    
    if method == "otsu":
        threshold, img_thr = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    elif method == "otsu_gaussian":
        blur = cv.GaussianBlur(img, gaussian_filter, 0)
        threshold, img_thr = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    
    return float(threshold), img_thr


def binarize_and_save(input_path: Union[str, Path], output_path: Union[str, Path], gaussian_filter: Tuple[int, int] = (5, 5), 
    method: str = "otsu_gaussian",dry_run: bool = False) -> dict:
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    image_folders = [f for f in input_path.iterdir() if f.is_dir()]
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    
    stats = {"total_folders": len(image_folders), "total_images": 0, "success_count": 0, "failed_count": 0, "thresholds_summary": {}}
    
    for img_folder in tqdm(image_folders, desc="Processing folders", unit="folder"):
        image_files = [f for f in img_folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            continue
            
        stats["total_images"] += len(image_files)
    
        # Create matching subfolder in output_path
        folder_output_path = output_path / img_folder.name
        if not dry_run:
            folder_output_path.mkdir(parents=True, exist_ok=True)
        
        thresholds: List[float] = []
        folder_success_count = 0
        
        for img_path in tqdm(image_files, desc=f"Images in {img_folder.name}", unit="file", leave=False):
            try:
                _, _, processed_name = format_filename(base_name=img_path.stem, output_folder=folder_output_path)
                
                threshold, img_thr = binarize_image(input_img=img_path, gaussian_filter=gaussian_filter, method=method)
                
                if not dry_run:
                    output_file = folder_output_path / f"{processed_name}.png"
                    cv.imwrite(str(output_file), img_thr)
                
                thresholds.append(float(threshold))
                folder_success_count += 1
                
            except Exception:
                continue
        
        stats["success_count"] += folder_success_count
        stats["failed_count"] += len(image_files) - folder_success_count
        
        if not dry_run and thresholds:
            summary_file = folder_output_path / f"{img_folder.stem}_thresholds.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Folder: {img_folder.name}\n")
                f.write(f"Processed: {folder_success_count}/{len(image_files)}\n")
                f.write(f"Thresholds: min={min(thresholds):.2f} / max={max(thresholds):.2f} / avg={np.mean(thresholds):.2f}\n")
            
            stats["thresholds_summary"][img_folder.name] = {
                "min": round(min(thresholds), 2),
                "max": round(max(thresholds), 2),
                "avg": round(np.mean(thresholds), 2)
            }
    
    return stats

def run_binarization_pipeline( input_path: Union[str, Path], output_base_dir: Union[str, Path], logs_dir: Optional[str] = None, run_name: Optional[str] = None, gaussian_filter: Tuple[int, int] = (3, 3), method: str = "otsu_gaussian", dry_run: bool = False) -> dict:
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"bin_{timestamp}"
    
    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Binarization Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("binarization")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        log_file = None
    
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.environ.get("PROJECT_ROOT", "."),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        git_commit = "unknown"
    
    input_path = Path(input_path)
    output_base_dir = Path(output_base_dir)
    
    if not input_path.is_dir():
        logger.error(f"Input folder not found: {input_path}")
        return {"success": False}
    
    image_folders = [f for f in input_path.iterdir() if f.is_dir()]
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    total_images = sum(
        len([f for f in folder.iterdir() if f.suffix.lower() in image_extensions])
        for folder in image_folders)
    
    # Log essential config only
    config_summary = {
        "run": run_name,
        "git": git_commit,
        "input": str(input_path.name),
        "output_base": str(output_base_dir.name),
        "timestamp": timestamp,
        "method": method,
        "gaussian_filter": gaussian_filter,
        "folders_count": len(image_folders),
        "images_count": total_images,
        "dry_run": dry_run
    }
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")
    
    run_output_dir = output_base_dir / timestamp
    if not dry_run:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {run_output_dir}")
    
    logger.info(f"Starting binarization for {total_images} images across {len(image_folders)} folders...")
    
    stats = binarize_and_save(input_path=input_path, output_path=run_output_dir, gaussian_filter=gaussian_filter, method=method, dry_run=dry_run)
    
    if len(image_folders) >= 20:
        for idx in range(20, len(image_folders) + 1, 20):
            logger.info(f"📊 Progress: {idx}/{len(image_folders)} folders processed")
    
    success_rate = stats["success_count"] / stats["total_images"] * 100 if stats["total_images"] > 0 else 0
    
    logger.info(f"Binarization complete: {stats['success_count']}/{stats['total_images']} succeeded ({success_rate:.1f}%)")
    logger.info(f"Failed: {stats['failed_count']}")
    logger.info(f"Results: {run_output_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return {
        "total_folders": stats["total_folders"],
        "total_images": stats["total_images"],
        "success": stats["success_count"],
        "failed": stats["failed_count"],
        "success_rate_percent": round(success_rate, 1),
        "output_dir": str(run_output_dir),
        "thresholds_summary": stats["thresholds_summary"] if not dry_run else {}
    }

if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    
    INPUT_PATH = project_root / "data" / "processed" / "extracted_lines" / "extraction_20260427_221639"
    OUTPUT_BASE_DIR = project_root / "data" / "processed" / "binarized_images"
    
    LOGS_DIR = project_root / "logs" / "binarization"
    RUN_NAME = f"bin_{INPUT_PATH.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    GAUSSIAN_FILTER = (3, 3)
    METHOD = "otsu_gaussian"
    DRY_RUN = False  # Set True to test without writing files
    
    result = run_binarization_pipeline(
        input_path=INPUT_PATH,
        output_base_dir=OUTPUT_BASE_DIR,
        logs_dir=str(LOGS_DIR),
        run_name=RUN_NAME,
        gaussian_filter=GAUSSIAN_FILTER,
        method=METHOD,
        dry_run=DRY_RUN
    )
    
    print(f"\n{'='*50}")
    print(f"BINARIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"Folders:        {result.get('total_folders', 0)}")
    print(f"Total images:   {result.get('total_images', 0)}")
    print(f"Success:     {result.get('success', 0)}")
    print(f"Failed:      {result.get('failed', 0)}")
    if result.get('success_rate_percent') is not None:
        print(f"Success rate: {result['success_rate_percent']:.1f}%")
    print(f"Output:      {result.get('output_dir')}")
    print(f"{'='*50}\n")