import os
import logging
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import format_filename, format_for_cli

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True 
)
logger = logging.getLogger(__name__)


def segment_image( img_path: Path, output_path: Path, mask_path: None) -> bool:
    """Run Kraken segmentation for a single image. Returns True on success."""
    load_dotenv()
    project_root = Path(os.environ["PROJECT_ROOT"])
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = os.environ.get("PYTHON_IO_ENCODING", "utf-8")

    img_path = Path(img_path)
    output_path = Path(output_path)
    input_cmd, output_cmd = format_for_cli(img_path, output_path)
    if mask_path is not None:
        mask_cmd = format_for_cli(Path(mask_path))[0]

    kraken_bin = os.environ.get("KRAKEN_BIN") or shutil.which("kraken")
    logger.info(f"KRAKEN_BIN resolved to: {kraken_bin}")
    
    if not kraken_bin:
        logger.error("Kraken executable not found. Set KRAKEN_BIN env var or add to PATH.")
        return


    try:

        cmd = [kraken_bin, "-i", input_cmd, output_cmd, "segment", "-bl"]
        if mask_cmd:
            cmd.extend(["-m", mask_cmd])
            
        logger.debug(f"Running: {' '.join(cmd)}")
        
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )

        logger.info(f"Saved: {output_cmd}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Kraken failed for {output_cmd}: {e.stderr.strip()}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error for {output_cmd}: {e}")
        return False
    

def segment_all_images() -> None:
    load_dotenv()
    project_root = Path(os.environ["PROJECT_ROOT"])

    input_folder = project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    output_folder = project_root / "data" / "processed" / "segmented_images"
    output_folder.mkdir(parents=True, exist_ok=True)

    masks_folder = project_root / "data" / "processed" / "img_layout" / "masks"

    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]

    logger.info(f"Starting segmentation for {len(image_files)} images")
    success_count = 0

    for img_path in tqdm(image_files, desc="Processing images", unit="file"):
        base_name = img_path.stem
        output_path, output_filename, processed_name = format_filename(base_name, output_folder)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)

        
        mask_path = masks_folder / f"{processed_name}.png"
        if segment_image(input_cmd, output_cmd, mask_path=mask_path):
            success_count += 1

    logger.info(f"Segmentation complete: {success_count}/{len(image_files)} succeeded.")

if __name__ == "__main__":
    segment_all_images()