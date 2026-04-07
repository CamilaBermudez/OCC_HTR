import os
import logging
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm  # pip install tqdm
from src.utils.path_utils import format_filename, format_for_cli

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def segment_image( input_path_cmd: str, output_path_cmd: str, output_filename: str,
    kraken_bin: str) -> bool:
    """Run Kraken segmentation for a single image. Returns True on success."""
    load_dotenv()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = os.environ.get("PYTHON_IO_ENCODING", "utf-8")

    try:
        subprocess.run(
            [kraken_bin, "-i", input_path_cmd, output_path_cmd, "segment", "-bl"],
            check=True,
            capture_output=True,
            text=True,
            env=env
        )
        logger.info(f"Saved: {output_filename}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Kraken failed for {output_filename}: {e.stderr.strip()}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error for {output_filename}: {e}")
        return False
    

def segment_all_images() -> None:
    load_dotenv()
    project_root = Path(os.environ["PROJECT_ROOT"])

    input_folder = project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    output_folder = project_root / "data" / "processed" / "segmented_images"
    output_folder.mkdir(parents=True, exist_ok=True)

    kraken_bin = os.environ.get("KRAKEN_BIN") or shutil.which("kraken")
    if not kraken_bin:
        logger.error("Kraken executable not found. Set KRAKEN_BIN env var or add to PATH.")
        return

    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]

    logger.info(f"Starting segmentation for {len(image_files)} images...")
    success_count = 0

    for img_path in tqdm(image_files, desc="Processing images", unit="file"):
        base_name = img_path.stem
        output_path, output_filename, _ = format_filename(base_name, output_folder)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)

        if segment_image(input_cmd, output_cmd, output_filename, kraken_bin):
            success_count += 1

    logger.info(f"Segmentation complete: {success_count}/{len(image_files)} succeeded.")

if __name__ == "__main__":
    segment_all_images()