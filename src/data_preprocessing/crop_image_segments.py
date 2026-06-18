import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.path_utils import format_filename


def setup_simple_logging(logs_dir: str, run_name: str | None = None):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(logs_dir) / f"{run_name}_cropping.log"

    logger = logging.getLogger("cropping")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File + console handlers (same format)
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger, str(log_file)


# Parchment background colour used when polygon-cropping a line. Matches
# the default `bg` in `medieval_text_generation.render_text_to_image`
# (240, 230, 200) so a cropped line on its parchment fill looks visually
# the same family as a synthetic line — useful both for downstream
# augmentation (no harsh black/parchment contrast) and for OCR models
# that expect a parchment-toned background outside the text region.
PARCHMENT_BG: tuple[int, int, int] = (240, 230, 200)


def extract_polygon(img_np, polygon, bg_color: tuple[int, int, int] = PARCHMENT_BG):
    """Crop the polygon region from ``img_np`` and replace outside-polygon
    pixels with ``bg_color`` (RGB).

    The crop is the axis-aligned bounding box of the polygon; pixels
    *inside* the polygon come from the source image, pixels *outside*
    are filled with the parchment colour.
    """
    pts = np.array(polygon, dtype=np.int32)
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    # Build a parchment-coloured canvas of the same shape, then composite
    # the image pixels where the polygon mask is set. Using np.where on
    # the boolean mask broadcast across channels avoids the black halo
    # that cv2.bitwise_and leaves outside the polygon.
    canvas = np.full_like(img_np, fill_value=0)
    canvas[:] = bg_color
    composed = np.where(mask[:, :, None].astype(bool), img_np, canvas)
    x, y, w, h = cv2.boundingRect(pts)
    cropped = composed[y : y + h, x : x + w]
    return cropped


def crop_image_segments(
    img_path: Path,
    kraken_output_path: Path,
    processed_name: str,
    output_path: Path,
    crop_type: str = "polygon",
) -> bool:
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        with open(kraken_output_path, encoding="utf-8") as f:
            kraken_output = json.load(f)

        for id, line in enumerate(kraken_output.get("lines", [])):
            if "boundary" not in line:
                continue

            save_path = output_path / f"{processed_name}_line_{id}.png"

            if crop_type == "rectangle":
                xs, ys = zip(*line["boundary"], strict=False)
                left, upper = int(min(xs)), int(min(ys))
                right, lower = int(max(xs)), int(max(ys))
                cropped = img_pil.crop((left, upper, right, lower))
                cropped.save(save_path, optimize=True)
            elif crop_type == "polygon":
                polygon = [(int(x), int(y)) for x, y in line["boundary"]]
                crop_np = extract_polygon(img_np, polygon)
                crop_pil = Image.fromarray(crop_np)
                crop_pil.save(save_path, optimize=True)
            else:
                raise ValueError(f"Unknown crop_type: {crop_type}")
        return True
    except Exception as e:
        print(f"Error in crop_image_segments: {e}")
        return False


def crop_all_images(
    input_folder: str | Path,
    output_kraken_path: str | Path,
    output_folder: str | Path,
    logs_dir: str | None = None,
    run_name: str | None = None,
    crop_type: str = "polygon",
) -> dict:
    input_folder = Path(input_folder)
    output_kraken_path = Path(output_kraken_path)
    output_folder = Path(output_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_name is None:
        run_name = f"crop_{timestamp}"

    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Cropping Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("cropping")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
            )
        log_file = None

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],  # Short hash only
                cwd=os.environ.get("PROJECT_ROOT", "."),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    config_summary = {
        "run": run_name,
        "git": git_commit,
        "input": str(input_folder.name),
        "output": str(output_folder.name),
        "images_count": None,
        "crop_type": str(crop_type),
    }

    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return {"success": False}

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]
    )

    config_summary["images_count"] = len(image_files)
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")

    logger.info(f"Starting cropping for {len(image_files)} images")
    run_output_folder = output_folder / f"extraction_{timestamp}"
    run_output_folder.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for img_path in tqdm(image_files, desc="Cropping", unit="file"):
        output_path_str, _, processed_name = format_filename(
            base_name=img_path.stem, output_folder=output_kraken_path
        )
        json_path = Path(output_path_str)
        file_output_dir = run_output_folder / processed_name
        file_output_dir.mkdir(parents=True, exist_ok=True)

        if not json_path.exists():
            continue

        if crop_image_segments(img_path, json_path, processed_name, file_output_dir, crop_type):
            success_count += 1

    logger.info(f"Cropping complete: {success_count}/{len(image_files)} succeeded")
    logger.info(f"Output: {run_output_folder}")
    if log_file:
        logger.info(f"📄 Log file: {log_file}")

    return {"total": len(image_files), "success": success_count, "output": str(run_output_folder)}
