import os
import logging
import shutil
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm  
import sys
import datetime
from typing import Optional

sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.utils.path_utils import format_filename, format_for_cli

def setup_simple_logging(logs_dir: str, run_name: Optional[str] = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_segmentation.log"
    
    logger = logging.getLogger("segmentation")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)
    
    return logger, str(log_file)


def segment_image(img_path: Path, output_path: Path, mask_path: Optional[Path] = None) -> bool:
    try:
        load_dotenv()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = os.environ.get("PYTHON_IO_ENCODING", "utf-8")

        img_path = Path(img_path)
        output_path = Path(output_path)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)
        
        mask_cmd = None
        if mask_path is not None and Path(mask_path).exists():
            mask_cmd = format_for_cli(Path(mask_path), None)[0]

        kraken_bin = os.environ.get("KRAKEN_BIN") or shutil.which("kraken")
        if not kraken_bin:
            return False

        cmd = [kraken_bin, "-i", input_cmd, output_cmd, "segment", "-bl"]
        if mask_cmd:
            cmd.extend(["-m", mask_cmd])
        
        subprocess.run(cmd,check=True,capture_output=True,text=True,env=env,timeout=300)
        return True
    except Exception:
        return False
    

def segment_all_images(input_folder: Union[str, Path],output_folder: Union[str, Path], masks_folder: Union[str, Path], logs_dir: Optional[str] = None, run_name: Optional[str] = None) -> dict:
    
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
    
    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return {"success": False}
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions])
    
    config_summary = {"run": run_name,"git": git_commit,"input": str(input_folder.name),"output": str(output_folder.name),"images_count": len(image_files)}
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting segmentation for {len(image_files)} images...")
    
    success_count = 0
    for idx, img_path in enumerate(tqdm(image_files, desc="Segmenting", unit="file"), 1):
        base_name = img_path.stem
        output_path, output_filename, processed_name = format_filename(base_name, output_folder)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)
        
        mask_path = masks_folder / f"{processed_name}.png"
        if not mask_path.exists():
            mask_path = None
        
        if segment_image(input_cmd, output_cmd, mask_path=mask_path):
            success_count += 1
        
        if idx % 20 == 0 or idx == len(image_files):
            logger.info(f"Progress: {idx}/{len(image_files)} | OK: {success_count}")

    logger.info(f"Segmentation complete: {success_count}/{len(image_files)} succeeded")
    logger.info(f"Output: {output_folder}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return {"total": len(image_files),"success": success_count,"output": str(output_folder)}

if __name__ == "__main__":
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    
    INPUT_FOLDER = project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    OUTPUT_FOLDER = project_root / "data" / "processed" / "segmented_images"
    MASKS_FOLDER = project_root / "data" / "processed" / "img_layout" / "masks" / "20260427_101946"
    
    LOGS_DIR = project_root / "logs" / "segmentation"
    RUN_NAME = f"seg_{INPUT_FOLDER.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    result = segment_all_images(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER, masks_folder=MASKS_FOLDER, logs_dir=str(LOGS_DIR), run_name=RUN_NAME)
    
    print(f"\n Done: {result['success']}/{result['total']} images | Output: {result['output']}\n")