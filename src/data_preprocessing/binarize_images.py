import os
import cv2 as cv
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import sys
from dotenv import load_dotenv
from typing import Tuple, Union, List, Optional
from argparse import ArgumentParser

load_dotenv()

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from src.utils.path_utils import format_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger(__name__)


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
    logger.info(f"Found {len(image_folders)} image folders to process")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    
    stats = {"total_folders": len(image_folders), "total_images": 0, "success_count": 0, "failed_count": 0}
    
    for img_folder in tqdm(image_folders, desc="Processing folders", unit="folder"):
        image_files = [f for f in img_folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            logger.warning(f"No valid images found in {img_folder}")
            continue
            
        stats["total_images"] += len(image_files)
        logger.info(f"Starting binarization for {len(image_files)} images in {img_folder.name}")
        
        # Create matching subfolder in output_path
        folder_output_path = output_path / img_folder.name
        if not dry_run:
            folder_output_path.mkdir(parents=True, exist_ok=True)
        
        thresholds: List[float] = []
        failed_files: List[str] = []
        folder_success_count = 0
        
        for img_path in tqdm(image_files, desc=f"Images in {img_folder.name}", unit="file", leave=False):
            try:
                # Pass the subfolder path to format_filename so it generates names relative to that folder
                _, _, processed_name = format_filename(
                    base_name=img_path.stem, 
                    output_folder=folder_output_path  # ← Use subfolder, not root output_path
                )
                
                threshold, img_thr = binarize_image(
                    input_img=img_path, 
                    gaussian_filter=gaussian_filter, 
                    method=method
                )
                
                if not dry_run:
                    #  Save to the subfolder
                    output_file = folder_output_path / f"{processed_name}.png"
                    cv.imwrite(str(output_file), img_thr)
                
                thresholds.append(float(threshold))
                folder_success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                failed_files.append(img_path.name)
                continue
        
        stats["success_count"] += folder_success_count
        stats["failed_count"] += len(failed_files)
        
        if not dry_run:
            summary_file = folder_output_path / f"{img_folder.stem}_thresholds.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                if thresholds:
                    f.write(f"Folder: {img_folder.name}\n")
                    f.write(f"Processed: {folder_success_count}/{len(image_files)} images\n")
                    f.write(f"Thresholds (min/max/avg): {min(thresholds):.2f} / {max(thresholds):.2f} / {np.mean(thresholds):.2f}\n")
                    f.write("Individual thresholds:\n")
                    f.write(", ".join(f"{t:.2f}" for t in thresholds) + "\n")
                else:
                    f.write("No thresholds recorded - processing failed.\n")
        
        if failed_files:
            logger.warning(f"Failed files in {img_folder.name}: {failed_files}")
        
        logger.info(f"Completed {img_folder.name}: {folder_success_count}/{len(image_files)} successful")
    
    return stats



if __name__ == "__main__":
    base_root = Path(PROJECT_ROOT)
    IMAGES_PATH = base_root / "data" / "processed" / "extracted_lines"
    OUTPUT_PATH = base_root / "data" / "processed" / "binarized_images"
    
    results = binarize_and_save(input_path=IMAGES_PATH, output_path=OUTPUT_PATH, gaussian_filter=(3,3), method="otsu_gaussian",dry_run=False)
    
    logger.info(f"Summary: {results['success_count']}/{results['total_images']} images succeeded, "
                f"{results['failed_count']} failed across {results['total_folders']} folders.")