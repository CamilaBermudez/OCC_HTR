import os
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import format_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (IOError, OSError):
        return ImageFont.load_default()
    

def plot_image_with_bounds(image_file: Path, json_path: Path, output_path: Path, font_size: int = 20) -> None:
    
    with open(json_path, "r", encoding="utf-8") as f:
        kraken_data = json.load(f)

    with Image.open(image_file).convert("RGB") as img:
        draw = ImageDraw.Draw(img)

    for idx, line in enumerate(kraken_data.get("lines", []), start=1):
        baseline = line.get("baseline", [])
        
        # Baseline (red)
        draw.line([tuple(baseline[0]), tuple(baseline[1])], fill="red", width=2)

        # Boundary polygon (blue)
        boundary = line.get("boundary", [])
        if boundary:
            draw.polygon([tuple(pt) for pt in boundary], outline="blue")

        # Line ID (green) - clamp to avoid negative coordinates
        x, y = baseline[0]
        text_pos = (max(0, x - 20), max(0, y - 20))
        draw.text(text_pos, str(idx - 1), fill="green", font=get_font())
    img.save(output_path, optimize=True)
    return None


def plot_all_images_with_bounds( input_dir: str | Path, kraken_output_path: str | Path, output_dir: str | Path,font_size: int = 20) -> None:
    
    input_dir, kraken_output_path, output_dir = Path(input_dir), Path(kraken_output_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    font = get_font(font_size)
    processed_count = 0
    error_count = 0

    logger.info(f"Starting image annotation for: {input_dir}")

    for image_file in sorted(input_dir.iterdir()):
        if not image_file.is_file() or image_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue

        base_name = image_file.stem
        json_path_str, _, processed_name = format_filename(base_name, kraken_output_path)
        json_path = Path(json_path_str) 
        output_path = output_dir / f"{processed_name}.png"

        if not json_path.exists():
            logger.warning(f"JSON not found for {image_file.name}, skipping.")
            error_count += 1
            continue

        try:
            plot_image_with_bounds(image_file, json_path, output_path)
            
            processed_count += 1
            logger.info(f"{processed_count:>3} | {image_file.name}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path.name}: {e}")
            error_count += 1
        except Exception as e:
            logger.exception(f"Failed to process {image_file.name}: {e}")
            error_count += 1

    logger.info(f"Done. Processed: {processed_count} | Errors/Skipped: {error_count}")

if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    
    plot_all_images_with_bounds(
        input_dir=project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100",
        kraken_output_path= project_root / "data" / "processed" / "segmented_images",
        output_dir=project_root / "data" / "processed" / "plotted_bounds",
    )
