import os
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import datetime
import subprocess
from typing import Union, Optional
from dotenv import load_dotenv
import sys

sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import format_filename


def setup_simple_logging(logs_dir: str, run_name: Optional[str] = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_plotting.log"
    
    logger = logging.getLogger("plotting")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    
    return logger, str(log_file)

def get_config_summary(run_name: str,timestamp: str,input_dir: Path,kraken_output_path: Path,output_dir: Path,font_size: int,logs_dir: Optional[str],git_commit: str) -> dict:
    return {
        "run_name": run_name,
        "timestamp": timestamp,
        "input_dir": str(input_dir),
        "kraken_output_path": str(kraken_output_path),
        "output_dir": str(output_dir),
        "font_size": font_size,
        "logs_dir": str(logs_dir) if logs_dir else None,
        "environment": {
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT", "."),
            "PYTHON_VERSION": sys.version.replace("\n", " ")
        },
        "git_commit": git_commit
    }


def get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        return ImageFont.load_default()
    

def plot_image_with_bounds(image_file: Path, json_path: Path, output_path: Path, font_size: int = 20) -> bool:
    """Plot bounds on image - NO internal logging, just return success/failure"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            kraken_data = json.load(f)

        with Image.open(image_file).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            font = get_font(font_size)

            for idx, line in enumerate(kraken_data.get("lines", []), start=0):
                baseline = line.get("baseline", [])
                
                # Baseline (red)
                if len(baseline) == 2:
                    draw.line([tuple(baseline[0]), tuple(baseline[1])], fill="red", width=2)

                # Boundary polygon (blue)
                boundary = line.get("boundary", [])
                if boundary:
                    draw.polygon([tuple(pt) for pt in boundary], outline="blue")

                # Line ID (green)
                if baseline:
                    x, y = baseline[0]
                    text_pos = (max(0, int(x) - 20), max(0, int(y) - 20))
                    draw.text(text_pos, str(idx), fill="green", font=font)
            
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, optimize=True)
        
        return True
    except Exception:
        return False 


def plot_all_images_with_bounds(input_dir: Union[str, Path], kraken_output_path: Union[str, Path], output_dir: Union[str, Path],font_size: int = 20,logs_dir: Optional[str] = None,run_name: Optional[str] = None) -> dict:

    input_dir = Path(input_dir)
    kraken_output_path = Path(kraken_output_path)
    output_dir = Path(output_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"plot_{timestamp}"
    
    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Plotting Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("plotting")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        log_file = None
    
    # Basic reproducibility: short git commit only
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.environ.get("PROJECT_ROOT", "."),
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        git_commit = "unknown"

    config = get_config_summary(
        run_name=run_name,
        timestamp=timestamp,
        input_dir=input_dir,
        kraken_output_path=kraken_output_path,
        output_dir=output_dir,
        font_size=font_size,
        logs_dir=logs_dir,
        git_commit=git_commit
    )
    
    if logger:
        logger.info("Configuration: " + json.dumps(config, indent=2, default=str))
    
    config_summary = {"run": run_name,"git": git_commit,"input": str(input_dir.name),"output": str(output_dir.name),"images_count": None}
    
    if not input_dir.is_dir():
        logger.error(f"Input folder not found: {input_dir}")
        return {"success": False}
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
    
    config_summary["images_count"] = len(image_files)
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")
    
    run_output_dir = output_dir / f'plotting_{timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting annotation for {len(image_files)} images...")
    
    processed_count = 0
    error_count = 0
    
    for idx, image_file in enumerate(image_files, start=1):
        base_name = image_file.stem
        json_path_str, _, processed_name = format_filename(base_name, kraken_output_path)
        json_path = Path(json_path_str)
        output_path = run_output_dir / f"{processed_name}.png"

        if not json_path.exists():
            logger.warning(f"JSON not found for {image_file.name}, skipping.")
            error_count += 1
            continue

        success = plot_image_with_bounds(image_file, json_path, output_path, font_size)
        
        if success:
            processed_count += 1
        else:
            error_count += 1
        
        if idx % 20 == 0 or idx == len(image_files):
            logger.info(f"Progress: {idx}/{len(image_files)} | OK: {processed_count}")

    logger.info(f"Done. Processed: {processed_count} | Errors/Skipped: {error_count}")
    logger.info(f"Output: {run_output_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return {"total": len(image_files),"processed": processed_count,"errors": error_count,"output": str(run_output_dir)}

