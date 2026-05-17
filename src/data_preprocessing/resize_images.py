import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

import torchvision.transforms as T
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")


def setup_simple_logging(logs_dir: str, run_name: str | None = None):
    """Create logger with file + console handlers. Returns logger and log path."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(logs_dir) / f"{run_name}.log"

    logger = logging.getLogger("resize_images")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Prevent duplicate handlers on re-runs

    # Single formatter for both handlers
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, str(log_file)


def pad_center_img(img: Image.Image, target_size: int = 224, fill_color: int = 0) -> Image.Image:
    """
    Resize image keeping aspect ratio, then pad to target_size × target_size with centered padding.
    Returns a PIL Image in RGB mode.
    """
    # Ensure input is PIL Image in RGB
    if not isinstance(img, Image.Image):
        img = Image.open(img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size

    # Resize longest side to target_size, keep aspect ratio
    if w > h:
        new_w = target_size
        new_h = int(round(h * (target_size / w)))
    else:
        new_h = target_size
        new_w = int(round(w * (target_size / h)))

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Calculate padding needed
    pad_w = target_size - new_w
    pad_h = target_size - new_h

    # Centered padding: handles odd values properly
    # Format: (left, top, right, bottom)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

    return T.functional.pad(img_resized, padding, fill=fill_color)


def resize_all_images(
    input_dir: str | Path,
    output_dir: str | Path,
    target_size: int = 224,
    fill_color: int = 0,
    logs_dir: str | None = None,
    run_name: str | None = None,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"resize_{timestamp}"

    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Resizing Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("resize_images")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
            )
        log_file = None

    # Git commit for reproducibility (optional, like your cropping script)
    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT or ".",
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    # Config summary (matching your style)
    config_summary = {
        "run": run_name,
        "git": git_commit,
        "input": str(input_dir.resolve()),
        "output": str(output_dir.resolve()),
        "target_size": target_size,
        "images_count": None,
    }

    if not input_dir.is_dir():
        if logs_dir:
            logger.error(f"Input folder not found: {input_dir}")
        return {"success": False}

    run_output_dir = output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    success_count = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [
        f for f in input_dir.rglob("*") if f.suffix.lower() in image_extensions and f.is_file()
    ]
    config_summary["images_count"] = len(image_files)
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")

    logger.info(f"Starting resize for {len(image_files)} images")

    for image_file in tqdm(image_files, desc="Processing images", unit="img"):
        try:
            img = Image.open(image_file)
            original_size = img.size
            img_padded = pad_center_img(img, target_size=target_size, fill_color=0)

            rel_path = image_file.relative_to(input_dir)
            dst_file = run_output_dir / rel_path.parent / f"{image_file.stem}.png"
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            img_padded.save(dst_file, format="PNG")

            manifest.append(
                {
                    "source": str(image_file.relative_to(input_dir)),
                    "destination": str(dst_file.relative_to(run_output_dir)),
                    "original_size": original_size,
                    "final_size": img_padded.size,
                }
            )
            success_count += 1

        except Exception as e:
            if logs_dir:
                logger.error(f"✗ {image_file.name}: {type(e).__name__}: {e}")
            continue

    # Final summary (matching your style exactly)
    logger.info(f"Resizing complete: {success_count}/{len(image_files)} succeeded")
    logger.info(f"Output: {run_output_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return {
        "total": len(image_files),
        "success": success_count,
        "output": str(run_output_dir),
        "log": log_file,
    }
