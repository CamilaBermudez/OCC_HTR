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


def setup_plotting_logging(logs_dir: str, run_name: Optional[str] = None) -> tuple[str, str, logging.Logger]:
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_plotting.log"
    config_file = Path(logs_dir) / f"{run_name}_config.json"
    
    logger = logging.getLogger("image_plotting")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers to avoid duplicates
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"=== Plotting Run Started ===")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Config file: {config_file}")
    
    return str(log_file), str(config_file), logger


def get_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [ "arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",]
    
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()
    

def plot_image_with_bounds(image_file: Path, json_path: Path, output_path: Path, font_size: int = 20, logger: Optional[logging.Logger] = None) -> bool:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            kraken_data = json.load(f)

        with Image.open(image_file).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            font = get_font(font_size)
            
            lines = kraken_data.get("lines", [])

            for idx, line in enumerate(lines, start=1):
                    # Draw baseline (red line)
                    baseline = line.get("baseline", [])
                    if len(baseline) == 2:
                        draw.line([tuple(baseline[0]), tuple(baseline[1])], fill="red", width=2)

                    # Draw boundary polygon (blue outline)
                    boundary = line.get("boundary", [])
                    if boundary and len(boundary) >= 3:
                        draw.polygon([tuple(pt) for pt in boundary], outline="blue",width=1)

                    # Draw line index label (green text)
                    if baseline:
                        x, y = baseline[0]
                        text_pos = (max(0, int(x) - 20), max(0, int(y) - 20))
                        draw.text(text_pos, str(idx - 1), fill="green", font=font)
            

            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, optimize=True, quality=95)

        logger.debug(f"✓ Saved: {output_path.name}")
        return True
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path.name}: {e}")
        return False
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        return False
    except Exception as e:
        logger.exception(f"Failed to plot {image_file.name}: {type(e).__name__}: {e}")
        return False


def plot_all_images_with_bounds(input_dir: Union[str, Path], kraken_output_path: Union[str, Path], output_dir: Union[str, Path],font_size: int = 20,logs_dir: Optional[str] = None,run_name: Optional[str] = None) -> dict:

    input_dir, kraken_output_path, output_dir = Path(input_dir), Path(kraken_output_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"plot_{timestamp}"

    if logs_dir:
        log_file, config_file, logger = setup_plotting_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("image_plotting")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                force=True
            )
        log_file = config_file = None


    run_output_dir = output_dir / f'plotting_{timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration for reproducibility
    config_info = {
        "run_name": run_name,
        "timestamp": timestamp,
        "input_dir": str(input_dir),
        "kraken_output_path": str(kraken_output_path),
        "output_dir": str(run_output_dir),
        "font_size": font_size,
        "logs_dir": logs_dir,
        "environment": {
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT"),
            "PYTHON_VERSION": sys.version
        }
    }

    try:
        config_info["git_commit"] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            cwd=os.environ.get("PROJECT_ROOT", ".")
        ).decode('ascii').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        config_info["git_commit"] = None
    
    if log_file:
        logger.info(f"Configuration: {json.dumps(config_info, indent=2)}")
    
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return {"success": False, "error": f"Input directory not found: {input_dir}"}
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions])
    
    if not image_files:
        logger.warning(f"No valid images found in: {input_dir}")
        return {"success": True, "processed": 0, "message": "No images to process"}
    
    logger.info(f"Starting annotation for {len(image_files)} images")
    logger.info(f"Output directory: {run_output_dir}")
    
    # Tracking metrics
    metrics = {"total_images": len(image_files),"processed_successfully": 0, "skipped_no_json": 0, "failed_errors": 0, "errors_detail": []}


    for idx, image_file in enumerate(image_files, start=1):
        base_name = image_file.stem
        json_path_str, _, processed_name = format_filename(base_name, kraken_output_path)
        json_path = Path(json_path_str) 
        output_path = run_output_dir / f"{processed_name}.png"

        if not json_path.exists():
            logger.warning(f"JSON not found for {image_file.name}, skipping.")
            metrics["skipped_no_json"] += 1
            continue

        success = plot_image_with_bounds(image_file, json_path, output_path, font_size=font_size, logger=logger)
    
        if success:
            metrics["processed_successfully"] += 1
            if idx % 10 == 0 or idx == len(image_files):
                logger.info(f"Progress: {idx}/{len(image_files)} | ✓ {metrics['processed_successfully']}")
        else:
            metrics["failed_errors"] += 1
            metrics["errors_detail"].append({"image": image_file.name,"json": json_path.name})
     
    success_rate = (metrics["processed_successfully"] / metrics["total_images"] * 100 
                   if metrics["total_images"] > 0 else 0)
    
    summary = {
        **metrics,
        "success_rate_percent": round(success_rate, 2),
        "output_directory": str(run_output_dir),
        "completed_at": datetime.datetime.now().isoformat()
    }
    
    
    logger.info("=== Plotting Complete ===")
    logger.info(f"Successfully processed: {metrics['processed_successfully']}/{metrics['total_images']} ({success_rate:.1f}%)")
    logger.info(f"Skipped (no JSON): {metrics['skipped_no_json']}")
    logger.info(f"Failed (errors): {metrics['failed_errors']}")
    logger.info(f"Results saved to: {run_output_dir}")
    
    if metrics["failed_errors"] > 0:
        failed_names = [e["image"] for e in metrics["errors_detail"][:5]]
        logger.warning(f"Failed images: {failed_names}")
        if len(metrics["errors_detail"]) > 5:
            logger.warning(f"... and {len(metrics['errors_detail']) - 5} more")
    
    # Save config + metrics to JSON file
    if config_file:
        output_config = {**config_info, "results": summary}
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(output_config, f, indent=2, default=str)
        logger.info(f"Full config + metrics saved to: {config_file}")
    
    logger.info(f"=== Run '{run_name}' Finished ===")
    
    return summary

    

if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    
    input_dir = project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    kraken_output_path = project_root / "data" / "processed" / "segmented_images" / "segmentation_20260427_181912"
    output_dir = project_root / "data" / "processed" / "plotted_bounds"
    
    logs_dir = project_root / "logs" / "image_plotting"
    
    run_name = f"plot_{input_dir.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"  # NEW

    result = plot_all_images_with_bounds( input_dir=input_dir,kraken_output_path=kraken_output_path,
        output_dir=output_dir,font_size=20,logs_dir=str(logs_dir), run_name=run_name )


    print(f"\n{'='*60}")
    print(f"ANNOTATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images:      {result.get('total_images', 0)}")
    print(f"Processed:      {result.get('processed_successfully', 0)}")
    print(f"Skipped (no JSON): {result.get('skipped_no_json', 0)}")
    print(f"Failed:          {result.get('failed_errors', 0)}")
    if result.get('success_rate_percent'):
        print(f"Success Rate:   {result['success_rate_percent']:.1f}%")
    if result.get('output_directory'):
        print(f"Output:         {result['output_directory']}")
    print(f"{'='*60}\n")