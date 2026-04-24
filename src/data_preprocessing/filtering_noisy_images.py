import os
import cv2 as cv
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Tuple, List, Optional
from tqdm import tqdm
import shutil
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True 
)
logger = logging.getLogger(__name__)


def calculate_text_density(image: Union[np.ndarray, str, Path]) -> Tuple[float, int]:
    if isinstance(image, (str, Path)):
        img = cv.imread(str(image), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        img = image.copy()

    total_pixels = img.size
    if total_pixels == 0:
        return 0.0, 0

    text_pixels = np.count_nonzero(img == 0)
    return float(text_pixels / total_pixels), total_pixels


def process_image_folder(input_path: Union[str, Path]) -> List[dict]:
    """
    Returns: List of dicts with folder name, total lines and list of (density, size) tuples
    """
    input_path = Path(input_path)
    image_folders = [f for f in input_path.iterdir() if f.is_dir()]
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    glb_density = []
    
    for img_folder in tqdm(image_folders, desc="Processing folders", unit="folder"):
        image_files = [f for f in img_folder.iterdir() if f.suffix.lower() in image_extensions]
        folder_density = []
        
        for img_path in tqdm(image_files, desc=f"Images in {img_folder.name}", unit="file", leave=False):
            try:
                density, size = calculate_text_density(image=img_path)
                folder_density.append((density, size))
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                continue
        
        glb_density.append({'folder': img_folder.name,'total_lines': len(image_files),'density': folder_density})
    
    return glb_density


def descr_statistics_list(density_size_data: List[dict], type: str, percentiles: List[float]) -> pd.Series:
    """Extract either density or size values and compute descriptive statistics."""
    idx = 0 if type == "density" else 1
    full_list = [item[idx] for sublist in density_size_data for item in sublist['density']]
    return pd.Series(full_list).describe(percentiles=percentiles)


def filter_based_on_thresholds(density_size_data: List[dict], type: str, descr_stats: pd.Series, 
                               threshold: str, upper: bool, src: Path) -> List[Tuple[int, int, str, float, int]]:
    """
    Returns list of (folder_idx, file_idx, stem, density, size) for images meeting threshold condition.
    """
    idx = 0 if type == "density" else 1
    list_filtered = []
    
    folder_stems = {}
    for sublist in density_size_data:
        folder_name = sublist['folder']
        path_ = Path(src) / folder_name
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        folder_stems[folder_name] = sorted([f.stem for f in path_.iterdir() if f.suffix.lower() in image_extensions])
    
    for i, sublist in enumerate(density_size_data):
        folder_name = sublist['folder']
        stems = folder_stems.get(folder_name, [])
        
        for j, (density, size) in enumerate(sublist['density']):
            if j >= len(stems):
                continue  
            
            stem = stems[j]
            value = density if type == "density" else size
            
            if upper:
                if value >= descr_stats[threshold]:
                    list_filtered.append((i, j, stem, density, size))
            else:
                if value <= descr_stats[threshold]:
                    list_filtered.append((i, j, stem, density, size))
    
    return list_filtered

def _threshold_to_percentile_key(threshold: float) -> str:
    pct = threshold * 100
    if pct == int(pct):
        return f"{int(pct)}%"
    return f"{pct}%"


def filter_noisy_lines(src_dir: Path, dst_dir: Path, size_thresholds: List[float] = [0.05], density_thresholds: List[float] = [0.001, 0.99]) -> None:
    density_size_data = process_image_folder(src_dir)
    
    stats_density = descr_statistics_list(density_size_data, "density", density_thresholds)
    stats_size = descr_statistics_list(density_size_data, "size", size_thresholds)
    
    bounds_density_down = []
    bounds_density_up = []
    bounds_size_down = []
    bounds_size_up = []
    
    # Density filters
    if len(density_thresholds) >= 1:
        key = _threshold_to_percentile_key(density_thresholds[0])
        bounds_density_down = filter_based_on_thresholds(density_size_data, "density", stats_density, key, upper=False, src=src_dir)
    if len(density_thresholds) == 2:
        key = _threshold_to_percentile_key(density_thresholds[1])
        bounds_density_up = filter_based_on_thresholds(density_size_data, "density", stats_density, key, upper=True, src=src_dir)
    
    # Size filters
    if len(size_thresholds) >= 1:
        key = _threshold_to_percentile_key(size_thresholds[0])
        bounds_size_down = filter_based_on_thresholds(density_size_data, "size", stats_size, key, upper=False, src=src_dir)
    if len(size_thresholds) == 2:
        key = _threshold_to_percentile_key(size_thresholds[1])
        bounds_size_up = filter_based_on_thresholds(density_size_data, "size", stats_size, key, upper=True, src=src_dir)



    set_density_down = set(bounds_density_down)
    set_density_up = set(bounds_density_up)
    
    bounds_size_density_ = [
        item for item in bounds_size_down 
        if item in set_density_down or item in set_density_up]
    
    image_folders = [f for f in src_dir.iterdir() if f.is_dir()]
    exclude_keys = {(image_folders[item[0]].name, item[2]) for item in bounds_size_density_}
    
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    src_files = [f for f in src_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts]
    
    copied_count = 0
    excluded_count = 0
    
    for src_file in tqdm(src_files, desc="Copying filtered images", unit="file"):
        file_key = (src_file.parent.name, src_file.stem)
        
        if file_key in exclude_keys:
            excluded_count += 1
            continue
            
        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied_count += 1

    logger.info(f"Copy complete: {copied_count} copied, {excluded_count} excluded")
    logger.info(f"Output: {dst_dir}")


if __name__ == "__main__":
    src_dir = Path(PROJECT_ROOT) / "data" / "processed" / "binarized_images"
    dst_dir = Path(PROJECT_ROOT) / "data" / "processed" / "binarized_images_filtered"
    
    filter_noisy_lines(src_dir=src_dir, dst_dir=dst_dir, size_thresholds=[0.05], density_thresholds=[0.001, 0.99])