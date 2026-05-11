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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
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

def get_image_groups(input_path: Path, image_extensions):
    # Case 1: subfolders exist
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]

    if subfolders:
        return {
    folder.name: sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in image_extensions ]) 
        for folder in sorted(subfolders)}

    # Case 2: flat structure (images directly in root)
    return {
        "root": [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
    }

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
    
    image_groups = get_image_groups(input_path, IMAGE_EXTENSIONS)
    
    
    stats = {"total_folders": len(image_groups), "total_images": 0, "success_count": 0, "failed_count": 0, "thresholds_summary": {}}
    
    for group_name, image_files in tqdm(image_groups.items(), desc="Processing groups", unit="group"):
        if not image_files:
            continue
            
        stats["total_images"] += len(image_files)
    
        # Create matching subfolder in output_path
        folder_output_path = output_path / group_name
        if not dry_run:
            folder_output_path.mkdir(parents=True, exist_ok=True)
        
        thresholds: List[float] = []
        folder_success_count = 0
        
        for img_path in tqdm(image_files, desc=f"Images in {group_name}", unit="file", leave=False):
            try:
                _, _, processed_name = format_filename(base_name=img_path.stem, output_folder=folder_output_path)
                
                threshold, img_thr = binarize_image(input_img=img_path, gaussian_filter=gaussian_filter, method=method)
                
                if not dry_run:
                    output_file = folder_output_path / f"{processed_name}.png"
                    cv.imwrite(str(output_file), img_thr)
                
                thresholds.append(float(threshold))
                folder_success_count += 1
                
            except Exception as e:
                logging.getLogger("binarization").warning(f"Failed {img_path}: {e}")
                continue
        
        stats["success_count"] += folder_success_count
        stats["failed_count"] += len(image_files) - folder_success_count
        
        if not dry_run and thresholds:
            summary_file = folder_output_path / f"{group_name}_thresholds.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"Folder: {group_name}\n")
                f.write(f"Processed: {folder_success_count}/{len(image_files)}\n")
                f.write(f"Thresholds: min={min(thresholds):.2f} / max={max(thresholds):.2f} / avg={np.mean(thresholds):.2f}\n")
            
            stats["thresholds_summary"][group_name] = {
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
    
    
    image_groups = get_image_groups(input_path, IMAGE_EXTENSIONS)
    
    total_images = sum(len(files) for files in image_groups.values())
    
    # Log essential config only
    config_summary = {
        "run": run_name,
        "git": git_commit,
        "input": str(input_path.name),
        "output_base": str(output_base_dir.name),
        "timestamp": timestamp,
        "method": method,
        "gaussian_filter": gaussian_filter,
        "folders_count": len(image_groups),
        "images_count": total_images,
        "dry_run": dry_run
    }
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")
    
    run_output_dir = output_base_dir / timestamp
    if not dry_run:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_output_dir}")
    
    logger.info(f"Starting binarization for {total_images} images across {len(image_groups)} folders...")
    
    stats = binarize_and_save(input_path=input_path, output_path=run_output_dir, gaussian_filter=gaussian_filter, method=method, dry_run=dry_run)
    
    if len(image_groups) >= 20:
        for idx in range(20, len(image_groups) + 1, 20):
            logger.info(f"Progress: {idx}/{len(image_groups)} folders processed")
    
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