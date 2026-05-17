import datetime
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from kraken import serialization
from kraken.containers import BaselineLine, BBoxLine, Segmentation
from PIL import Image
from tqdm import tqdm

from src.utils.path_utils import format_filename, format_for_cli


def setup_simple_logging(logs_dir: str, run_name: str | None = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(logs_dir) / f"{run_name}_segmentation.log"

    logger = logging.getLogger("segmentation")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger, str(log_file)


def segment_image(img_path: Path, output_path: Path, mask_path: Path | None = None) -> bool:
    try:
        load_dotenv()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = os.environ.get("PYTHON_IO_ENCODING", "utf-8")

        img_path = Path(img_path)
        output_path = Path(output_path)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)

        mask_cmd = None
        if mask_path is not None and Path(mask_path).exists():
            mask_cmd = format_for_cli(Path(mask_path))[0]

        kraken_bin = os.environ.get("KRAKEN_BIN") or shutil.which("kraken")
        if not kraken_bin:
            return False

        cmd = [kraken_bin, "-i", input_cmd, output_cmd, "segment", "-bl"]
        if mask_cmd:
            cmd.extend(["-m", mask_cmd])

        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, timeout=300)
        return True
    except Exception:
        import traceback

        traceback.print_exc()  # shows full error + line number
        return False


def format_from_JSON_to_ALTO_XML(input_json_path, input_img_path, output_alto_path):
    with open(input_json_path, encoding="utf-8") as f:
        data = json.load(f)
    im = Image.open(input_img_path)

    lines = []
    for line in data["lines"]:
        if line.get("type") == "baselines":
            lines.append(
                BaselineLine(
                    id=line["id"],
                    baseline=line["baseline"],
                    boundary=line["boundary"],
                    text=line.get("text"),
                    tags=line.get("tags"),
                )
            )
        else:
            lines.append(
                BBoxLine(
                    id=line["id"],
                    bbox=line["bbox"],
                    text=line.get("text"),
                    tags=line.get("tags"),
                    split=line.get("split"),
                )
            )

    seg = Segmentation(
        type=data["type"],
        imagename=data["imagename"],
        text_direction=data["text_direction"],
        script_detection=data["script_detection"],
        lines=lines,
        regions=data.get("regions", {}),
        line_orders=data.get("line_orders", []),
    )

    alto = serialization.serialize(seg, image_size=im.size)
    with open(output_alto_path, "w", encoding="utf-8") as f:
        f.write(alto)


def segment_all_images(
    input_folder: str | Path,
    output_folder: str | Path,
    masks_folder: str | Path,
    logs_dir: str | None = None,
    run_name: str | None = None,
) -> dict:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    masks_folder = Path(masks_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"seg_{timestamp}"

    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Segmentation Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("segmentation")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
            )
        log_file = None

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.environ.get("PROJECT_ROOT", "."),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return {"success": False}

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]
    )

    config_summary = {
        "run": run_name,
        "git": git_commit,
        "input": str(input_folder.name),
        "output": str(output_folder.name),
        "images_count": len(image_files),
    }
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")

    output_folder = Path(output_folder) / f"segmentation_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting segmentation for {len(image_files)} images...")

    alto_dir = output_folder / "alto_format"
    alto_dir.mkdir(parents=True, exist_ok=True)

    if not output_folder.exists():
        logger.error("Failed to create directory. Check permissions or parent path.")

    success_count = 0
    for idx, img_path in enumerate(tqdm(image_files, desc="Segmenting", unit="file"), 1):
        base_name = img_path.stem
        output_path, output_filename, processed_name = format_filename(base_name, output_folder)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)
        alto_path = alto_dir / f"{processed_name}.xml"
        mask_path = masks_folder / f"{processed_name}.png"
        if not mask_path.exists():
            mask_path = None

        if segment_image(input_cmd, output_cmd, mask_path=mask_path):
            success_count += 1
            format_from_JSON_to_ALTO_XML(
                input_json_path=output_path, input_img_path=img_path, output_alto_path=alto_path
            )

        if idx % 20 == 0 or idx == len(image_files):
            logger.info(f"Progress: {idx}/{len(image_files)} | OK: {success_count}")

    logger.info(f"Segmentation complete: {success_count}/{len(image_files)} succeeded")
    logger.info(f"Output: {output_folder}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return {"total": len(image_files), "success": success_count, "output": str(output_folder)}
