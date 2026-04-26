import os
import cv2 as cv
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Union, Tuple, List, Optional
from collections import defaultdict
from tqdm import tqdm
import shutil
from datetime import datetime
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


def calculate_text_density(image: Union[np.ndarray, str, Path]) -> Tuple[float, int, int, int]:
    if isinstance(image, (str, Path)):
        img = cv.imread(str(image), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        img = image.copy()

    total_pixels = img.size
    height, width = img.shape
    if total_pixels == 0:
        return 0.0, 0, height, width
    
    text_pixels = np.count_nonzero(img == 0)
    return float(text_pixels / total_pixels), total_pixels, height, width


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
                density, size, height, width = calculate_text_density(image=img_path)
                folder_density.append((density, size, height, width))
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


def _threshold_to_percentile_key(threshold: float) -> str:
    pct = threshold * 100
    if pct == int(pct):
        return f"{int(pct)}%"
    return f"{pct}%"



def filter_based_on_thresholds(density_size_data: List[dict], density_thresholds: List[float] ,size_thresholds: List[float] ,stats_density: pd.Series,stats_size: pd.Series,src_dir: Path)-> Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple], List[Tuple], List[Tuple]]:
    bounds_density_down = []
    bounds_density_up = []
    bounds_size_down = []
    bounds_size_up = []
    long_imgs = []
    clean_img =[]


    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    folder_files = {}
    for sublist in density_size_data:
        folder_name = sublist['folder']
        folder_path = src_dir / folder_name
        folder_files[folder_name] = sorted(
            [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions],
            key=lambda x: x.stem)

    density_keys = {}
    if len(density_thresholds) >= 1:
        density_keys['low'] = _threshold_to_percentile_key(density_thresholds[0])
    if len(density_thresholds) == 2:
        density_keys['high'] = _threshold_to_percentile_key(density_thresholds[1])
    
    size_keys = {}
    if len(size_thresholds) >= 1:
        size_keys['low'] = _threshold_to_percentile_key(size_thresholds[0])
    if len(size_thresholds) == 2:
        size_keys['high'] = _threshold_to_percentile_key(size_thresholds[1])

    for i, sublist in enumerate(density_size_data):
        folder_name = sublist['folder']
        image_files = folder_files.get(folder_name, [])  # O(1) lookup
        
        # Safety guard: skip if folder has no valid images
        if not image_files:
            logger.warning(f"Folder '{folder_name}' has no valid image files")
            continue
            
        for j, (density, size, height, width) in enumerate(sublist['density']):
            
            if j >= len(image_files):
                logger.warning(f"Index mismatch in folder '{folder_name}': j={j} >= {len(image_files)} files")
                continue
            
            stem = image_files[j].stem
            _tuple_values = (i, j, stem, density, size, height, width)

            if 'low' in density_keys and density <= stats_density[density_keys['low']]:
                bounds_density_down.append(_tuple_values)
            if 'high' in density_keys and density >= stats_density[density_keys['high']]:
                bounds_density_up.append(_tuple_values)
            
            if 'low' in size_keys and size <= stats_size[size_keys['low']]:
                bounds_size_down.append(_tuple_values)
            if 'high' in size_keys and size >= stats_size[size_keys['high']]:
                bounds_size_up.append(_tuple_values)
            
            if height > width:
                long_imgs.append(_tuple_values)
            else:
                clean_img.append(_tuple_values)
    
    return bounds_density_down, bounds_density_up, bounds_size_down, bounds_size_up, long_imgs, clean_img
    
    
def move_files(src_dir: Path, dst_dir: Path, list_img_remove,type):
    
    image_folders = [f for f in src_dir.iterdir() if f.is_dir()]
    exclude_keys = {(image_folders[item[0]].stem, item[2]) for item in list_img_remove}

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    src_files = [f for f in src_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts]

    copied_count = 0
    for src_file in tqdm(src_files, desc="Copying filtered images", unit="file"):
        file_key = (src_file.parent.name, src_file.stem)
        
        folder = 'removed' if file_key in exclude_keys else 'kept'
            
        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_dir / type /folder / rel_path
        
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_file, dst_file)
        copied_count += 1

    print(f"Successfully copied {copied_count} images to:\n  {dst_dir}")

def get_key(t):
        return t[0], t[1], t[2],t[3],t[4],t[5],t[6]


def generate_analysis_df(dst_dir: Path,bounds_density_down,bounds_density_up,bounds_size_down,bounds_size_up,long_imgs,clean_img):
    
    doc_reasons = defaultdict(set)
    for t in bounds_density_down:
        doc_reasons[get_key(t)].add("density_down")
    for t in bounds_density_up:
        doc_reasons[get_key(t)].add("density_up")
    for t in bounds_size_down:
        doc_reasons[get_key(t)].add("size_down")
    for t in bounds_size_up:
        doc_reasons[get_key(t)].add("size_up")
    for t in long_imgs:
        doc_reasons[get_key(t)].add("long_imgs")
    for t in clean_img:
        doc_reasons[get_key(t)].add("clean")

    density_keys = set(get_key(t) for t in bounds_density_down) | set(get_key(t) for t in bounds_density_up)
    size_keys = set(get_key(t) for t in bounds_size_down) | set(get_key(t) for t in bounds_size_up)
    bounds_density_size_keys = density_keys & size_keys
    long_imgs_keys = set(get_key(t) for t in long_imgs)
    clean_img_keys = set(get_key(t) for t in clean_img)  

    img_remove_keys = long_imgs_keys | bounds_density_size_keys
    df_all = pd.DataFrame([
        {   "i": k[0], "j": k[1], "stem": k[2],
            "density": k[3], "size": k[4], "height": k[5], "width": k[6],
            "reasons": list(v),
            "was_removed": k in img_remove_keys} for k, v in doc_reasons.items()])

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = dst_dir / f"filter_tracking_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)
    logger.info(f"Tracking DataFrame saved to: {output_path}")
    
    logger.info(f"Total tracked images: {len(df_all)}")
    logger.info(f"Marked as removed (original logic): {df_all['was_removed'].sum()}")
    logger.info(f"Clean/kept images: {(~df_all['was_removed']).sum()}")




def filter_noisy_lines(src_dir: Path, dst_dir: Path,size_thresholds: List[float] = [0.05], density_thresholds: List[float] = [0.001, 0.99],export_tracking =True) -> None:
    
    density_size_data = process_image_folder(src_dir)
    
    stats_density = descr_statistics_list(density_size_data, "density", density_thresholds)
    stats_size = descr_statistics_list(density_size_data, "size", size_thresholds)
    
    
    bounds_density_down,bounds_density_up,bounds_size_down,bounds_size_up,long_imgs,clean_img = filter_based_on_thresholds(density_size_data=density_size_data, 
                                        density_thresholds= density_thresholds ,size_thresholds=size_thresholds ,stats_density = stats_density,stats_size = stats_size,src_dir=src_dir)

    bounds_density_ = set(bounds_density_down).union(set(bounds_density_up))
    bounds_size_ = set(bounds_size_down).union(set(bounds_size_up))
    bounds_density_size_ = set(bounds_size_).intersection(set(bounds_density_))
    img_remove = set([(i,j,stem,density,size,height,width)for i,j,stem,density,size,height,width in long_imgs]).union(set([(i,j,stem,density,size,height,width)for i,j,stem,density,size,height,width in bounds_density_size_]))

    if export_tracking:
        generate_analysis_df(dst_dir = dst_dir, bounds_density_down = bounds_density_down,
                             bounds_density_up = bounds_density_up,bounds_size_down = bounds_size_down,
                             bounds_size_up = bounds_size_up,long_imgs = long_imgs,clean_img = clean_img)
    
    return img_remove



if __name__ == "__main__":
    src_dir = Path(PROJECT_ROOT) / "data" / "processed" / "binarized_images"
    dst_dir = Path(PROJECT_ROOT) / "data" / "processed" / "filtered_images"
    
    img_remove = filter_noisy_lines(src_dir=src_dir, dst_dir=dst_dir, size_thresholds=[0.03], density_thresholds=[0.001, 0.997])
    
    # Moving binarized files
    move_files(src_dir = src_dir, dst_dir = dst_dir, list_img_remove = img_remove,type = "binarized")

    # Moving original images
    src_dir = Path(PROJECT_ROOT) / "data" / "processed" / "extracted_lines"
    move_files(src_dir = src_dir, dst_dir = dst_dir, list_img_remove = img_remove, type = "original")