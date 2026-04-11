import os
import sys
import numpy as np
import logging
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm  
from dotenv import load_dotenv
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import format_filename

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True 
)
logger = logging.getLogger(__name__)


def crop_image_segments( img_path: Path, kraken_output_path: Path, processed_name: str, output_path: Path) -> bool:
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


def crop_all_images() -> None:
    load_dotenv()
    project_root = Path(os.environ["PROJECT_ROOT"])

    input_folder = project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    output_kraken_path = project_root / "data" / "processed" / "segmented_images"
    output_folder = project_root / "data" / "processed" / "extracted_lines"
    output_folder.mkdir(parents=True, exist_ok=True)

    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]

    logger.info(f"Starting cropping for {len(image_files)} images...")
    success_count = 0

    for img_path in tqdm(image_files, desc="Processing images", unit="file"):
        output_path,_,processed_name= format_filename(base_name=img_path.stem, output_folder = output_kraken_path)

        file_output_dir = Path(output_folder) / processed_name
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        if crop_image_segments(img_path, kraken_output_path = output_path,processed_name=processed_name,output_path= file_output_dir):
            success_count += 1

    logger.info(f"Cropping complete: {success_count}/{len(image_files)} succeeded.")

if __name__ == "__main__":
    crop_all_images()
