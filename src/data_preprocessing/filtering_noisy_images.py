import os
import cv2 as cv
import numpy as np
import pandas as pd
import logging
import json
import subprocess
import shutil
from pathlib import Path
from typing import Union, Tuple, List, Optional, Set
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv


def setup_simple_logging(logs_dir: str, run_name: Optional[str] = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_filtering.log"
    
    logger = logging.getLogger("filtering")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    
    return logger, str(log_file)

def log_filter_funnel(
    logger, bounds_density_down, bounds_density_up, bounds_size_down, bounds_size_up,
    long_imgs, clean_img, img_remove_keys):
    def get_key(t):
        return (t[0], t[1])  # t = (folder, stem, density, size, height, width)
    
    density_low = {get_key(t) for t in bounds_density_down}
    density_high = {get_key(t) for t in bounds_density_up}
    density_all = density_low | density_high
    
    size_low = {get_key(t) for t in bounds_size_down}
    size_high = {get_key(t) for t in bounds_size_up}
    size_all = size_low | size_high
    
    long_set = {get_key(t) for t in long_imgs}
    clean_set = {get_key(t) for t in clean_img}
    
    density_size = density_all & size_all
    density_long = density_all & long_set
    size_long = size_all & long_set
    all_three = density_all & size_all & long_set
    
    to_remove = img_remove_keys
    
    logger.info("Filter Breakdown:")
    logger.info(f"    Density outliers: {len(density_all)} (low: {len(density_low)}, high: {len(density_high)})")
    logger.info(f"    Size outliers:    {len(size_all)} (low: {len(size_low)}, high: {len(size_high)})")
    logger.info(f"    Long images:      {len(long_set)}")
    logger.info(f"    Clean/kept:       {len(clean_set)}")
    
    logger.info(" Intersections:")
    logger.info(f"   Density ∩ Size:     {len(density_size)}")
    logger.info(f"   Density ∩ Long:     {len(density_long)}")
    logger.info(f"   Size ∩ Long:        {len(size_long)}")
    logger.info(f"   All three:          {len(all_three)}")
    
    logger.info(" Final Removal:")
    logger.info(f"   Unique to remove:   {len(to_remove)}")
    
    # ASCII funnel visualization
    total = len(density_all | size_all | long_set | clean_set)
    if total > 0:
        funnel = [
            (" Total analyzed", total),
            (" Density filter", len(density_all)),
            (" Size filter", len(size_all)),
            (" Long filter", len(long_set)),
            (" Final removal", len(to_remove)),
            (" Kept", total - len(to_remove))
        ]
        max_val = max(v for _, v in funnel)
        logger.info(" Funnel:")
        for label, count in funnel:
            bar = "█" * int(40 * count / max_val) if max_val > 0 else ""
            logger.info(f"   {label:20s} {count:5d} {bar}")

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
            except Exception:
                continue
        if folder_density:
            glb_density.append({'folder': img_folder.name,'total_lines': len(folder_density),'density': folder_density})
    
    return glb_density


def descr_statistics_list(density_size_data: List[dict], type: str, percentiles: List[float]) -> pd.Series:
    """Extract either density or size values and compute descriptive statistics."""
    idx = 0 if type == "density" else 1
    full_list = [item[idx] for sublist in density_size_data for item in sublist['density']]
    return pd.Series(full_list).describe(percentiles=percentiles)


def _threshold_to_percentile_key(threshold: float) -> str:
    pct = threshold * 100
    return f"{int(pct)}%" if pct == int(pct) else f"{pct}%"


def filter_based_on_thresholds(density_size_data: List[dict], density_thresholds: List[float] ,size_thresholds: List[float] ,stats_density: pd.Series,stats_size: pd.Series,src_dir: Path)-> Tuple[List[Tuple], List[Tuple], List[Tuple], List[Tuple], List[Tuple], List[Tuple]]:
    bounds_density_down, bounds_density_up = [], []
    bounds_size_down, bounds_size_up = [], []
    long_imgs, clean_img = [], []


    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    folder_files = {}
    for sublist in density_size_data:
        folder_path = src_dir / sublist['folder']
        if folder_path.is_dir():
            folder_files[sublist['folder']] = sorted(
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

    for sublist in density_size_data:
        folder_name = sublist['folder']
        image_files = folder_files.get(folder_name, [])
        
        if not image_files:
            continue
            
        for j, (density, size, height, width) in enumerate(sublist['density']):
            if j >= len(image_files):
                continue
            
            stem = image_files[j].stem
            _tuple = (folder_name, stem, density, size, height, width)

            if 'low' in density_keys and density <= stats_density[density_keys['low']]:
                bounds_density_down.append(_tuple)
            if 'high' in density_keys and density >= stats_density[density_keys['high']]:
                bounds_density_up.append(_tuple)
            
            if 'low' in size_keys and size <= stats_size[size_keys['low']]:
                bounds_size_down.append(_tuple)
            if 'high' in size_keys and size >= stats_size[size_keys['high']]:
                bounds_size_up.append(_tuple)
            
            if height > width:
                long_imgs.append(_tuple)
            else:
                clean_img.append(_tuple)
    
    return bounds_density_down, bounds_density_up, bounds_size_down, bounds_size_up, long_imgs, clean_img
   
    
def move_files_to_timestamped_folder(src_dir: Path, dst_base_dir: Path, list_img_remove: Set[Tuple], file_type: str, timestamp: str) -> Tuple[int, int]:
    
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    exclude_keys = {(item[0], item[1]) for item in list_img_remove}
    
    copied, removed = 0, 0
    src_files = [f for f in src_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts]

    for src_file in src_files:
        file_key = (src_file.parent.name, src_file.stem)
        folder = 'removed' if file_key in exclude_keys else 'kept'
        
        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_base_dir / timestamp / file_type / folder / rel_path
        
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        
        if folder == 'removed':
            removed += 1
        else:
            copied += 1

    return copied, removed


def generate_analysis_df( dst_dir: Path, timestamp: str, bounds_density_down, bounds_density_up, bounds_size_down, bounds_size_up, long_imgs, clean_img) -> pd.DataFrame:
    
    def get_key(t):
        return t[0], t[1], t[2], t[3], t[4], t[5]
    
    doc_reasons = defaultdict(set)
    for t in bounds_density_down:
        doc_reasons[get_key(t)].add("density_low")
    for t in bounds_density_up:
        doc_reasons[get_key(t)].add("density_high")
    for t in bounds_size_down:
        doc_reasons[get_key(t)].add("size_low")
    for t in bounds_size_up:
        doc_reasons[get_key(t)].add("size_high")
    for t in long_imgs:
        doc_reasons[get_key(t)].add("long_img")
    for t in clean_img:
        doc_reasons[get_key(t)].add("clean")

    density_keys = set(get_key(t) for t in bounds_density_down) | set(get_key(t) for t in bounds_density_up)
    size_keys = set(get_key(t) for t in bounds_size_down) | set(get_key(t) for t in bounds_size_up)
    bounds_density_size_keys = density_keys & size_keys
    long_imgs_keys = set(get_key(t) for t in long_imgs)
    
    img_remove_keys = long_imgs_keys | bounds_density_size_keys
    
    df_all = pd.DataFrame([
        {   "folder": k[0], "stem": k[1],
            "density": k[2], "size": k[3], "height": k[4], "width": k[5],
            "reasons": list(v),
            "was_removed": k in img_remove_keys
        } for k, v in doc_reasons.items()])
    
    output_path = dst_dir / timestamp / f"filter_tracking.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)
    
    return df_all


def filter_noisy_lines( src_dir: Path, dst_dir: Path, size_thresholds: List[float] = [0.05], 
    density_thresholds: List[float] = [0.001, 0.99], export_tracking: bool = True, timestamp: Optional[str] = None) -> Tuple[Set[Tuple], pd.DataFrame]:
   
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    density_size_data = process_image_folder(src_dir)
    stats_density = descr_statistics_list(density_size_data, "density", density_thresholds)
    stats_size = descr_statistics_list(density_size_data, "size", size_thresholds)
    
    bounds = filter_based_on_thresholds(
        density_size_data=density_size_data,
        density_thresholds=density_thresholds,
        size_thresholds=size_thresholds,
        stats_density=stats_density,
        stats_size=stats_size,
        src_dir=src_dir
    )
    bounds_density_down, bounds_density_up, bounds_size_down, bounds_size_up, long_imgs, clean_img = bounds
    
    def get_key(t): return (t[0], t[1]) 

    bounds_density_ = {get_key(t) for t in bounds_density_down} | {get_key(t) for t in bounds_density_up}
    bounds_size_ = {get_key(t) for t in bounds_size_down} | {get_key(t) for t in bounds_size_up}
    bounds_density_size_ = bounds_size_ & bounds_density_
    long_imgs_set = {get_key(t) for t in long_imgs}
    
    img_remove = long_imgs_set | bounds_density_size_
    
    filter_counts = {
        "bounds_density_down": bounds_density_down,
        "bounds_density_up": bounds_density_up,
        "bounds_size_down": bounds_size_down,
        "bounds_size_up": bounds_size_up,
        "long_imgs": long_imgs,
        "clean_img": clean_img
    }

    df_tracking = pd.DataFrame()
    if export_tracking:
        df_tracking = generate_analysis_df(
            dst_dir=dst_dir,
            timestamp=timestamp,
            bounds_density_down=bounds_density_down,
            bounds_density_up=bounds_density_up,
            bounds_size_down=bounds_size_down,
            bounds_size_up=bounds_size_up,
            long_imgs=long_imgs,
            clean_img=clean_img
        )
    
    return img_remove, df_tracking, filter_counts


def run_filtering_pipeline( binarized_src: Path, extracted_src: Path, dst_base_dir: Path,
    logs_dir: Optional[str] = None, run_name: Optional[str] = None, size_thresholds: List[float] = [0.03], density_thresholds: List[float] = [0.001, 0.997]) -> dict:
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"filter_{timestamp}"
    
    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Filtering Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("filtering")
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
    
    if not binarized_src.is_dir():
        logger.error(f"Binarized source folder not found: {binarized_src}")
        return {"success": False}
    
    config_summary = {
        "run": run_name,
        "git": git_commit,
        "binarized_input": str(binarized_src.name),
        "extracted_input": str(extracted_src.name),
        "output_base": str(dst_base_dir.name),
        "timestamp": timestamp,
        "density_thresholds": density_thresholds,
        "size_thresholds": size_thresholds
    }
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")
    
    run_output_dir = dst_base_dir / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_output_dir}")
    
    logger.info("Analyzing images for filtering...")
    img_remove, df_tracking, filter_counts = filter_noisy_lines(
        src_dir=binarized_src,
        dst_dir=dst_base_dir, 
        size_thresholds=size_thresholds,
        density_thresholds=density_thresholds,
        export_tracking=True,
        timestamp=timestamp
    )
    
    total_analyzed = len(df_tracking) if not df_tracking.empty else 0
    
    if not df_tracking.empty:
        log_filter_funnel(
        logger=logger,
        bounds_density_down=filter_counts["bounds_density_down"],
        bounds_density_up=filter_counts["bounds_density_up"],
        bounds_size_down=filter_counts["bounds_size_down"],
        bounds_size_up=filter_counts["bounds_size_up"],
        long_imgs=filter_counts["long_imgs"],
        clean_img=filter_counts["clean_img"],
        img_remove_keys=img_remove 
    )
    
    marked_removed = df_tracking['was_removed'].sum() if not df_tracking.empty else 0
    logger.info(f"Analysis complete: {total_analyzed} images | Marked for removal: {marked_removed}")
    
    logger.info("Moving binarized images...")
    kept_bin, removed_bin = move_files_to_timestamped_folder(
        src_dir=binarized_src,
        dst_base_dir=dst_base_dir,
        list_img_remove=img_remove,
        file_type="binarized",
        timestamp=timestamp
    )
    logger.info(f"Binarized: {kept_bin} kept | {removed_bin} removed")
    
    # Step 3: Move original extracted lines to timestamped folders
    if extracted_src.is_dir():
        logger.info("Moving original extracted lines...")
        kept_orig, removed_orig = move_files_to_timestamped_folder(
            src_dir=extracted_src,
            dst_base_dir=dst_base_dir,
            list_img_remove=img_remove,
            file_type="original",
            timestamp=timestamp
        )
        logger.info(f"   ✓ Original: {kept_orig} kept | {removed_orig} removed")
    else:
        logger.warning(f"⚠ Extracted source folder not found: {extracted_src}")
        kept_orig = removed_orig = 0
    
    # Final summary
    total_kept = kept_bin + kept_orig
    total_removed = removed_bin + removed_orig
    
    logger.info(f"Filtering complete")
    logger.info(f"Kept: {total_kept} | Removed: {total_removed}")
    logger.info(f"Results: {run_output_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return {
        "total_analyzed": total_analyzed,
        "marked_removed": int(marked_removed),
        "kept": total_kept,
        "removed": total_removed,
        "output_dir": str(run_output_dir),
        "tracking_csv": str(run_output_dir / "filter_tracking.csv")
    }


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    
    BINARIZED_SRC = project_root / "data" / "processed" / "binarized_images" / "20260427_233547"
    EXTRACTED_SRC = project_root / "data" / "processed" / "extracted_lines" / "extraction_20260427_221639"
    DST_BASE_DIR = project_root / "data" / "processed" / "filtered_images"
    
    LOGS_DIR = project_root / "logs" / "filtering"
    RUN_NAME = f"filter_{BINARIZED_SRC.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    SIZE_THRESHOLDS = [0.03]
    DENSITY_THRESHOLDS = [0.001, 0.997]
    
    result = run_filtering_pipeline(
        binarized_src=BINARIZED_SRC,
        extracted_src=EXTRACTED_SRC,
        dst_base_dir=DST_BASE_DIR,
        logs_dir=str(LOGS_DIR),
        run_name=RUN_NAME,
        size_thresholds=SIZE_THRESHOLDS,
        density_thresholds=DENSITY_THRESHOLDS
    )
    
    print(f"\n{'='*50}")
    print(f"FILTERING SUMMARY")
    print(f"{'='*50}")
    print(f"Analyzed:     {result.get('total_analyzed', 0)}")
    print(f"Kept:      {result.get('kept', 0)}")
    print(f"Removed:   {result.get('removed', 0)}")
    print(f"Output:    {result.get('output_dir')}")
    print(f"Tracking:  {result.get('tracking_csv')}")
    print(f"{'='*50}\n")

