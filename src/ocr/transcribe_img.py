import os
import sys
import json
import datetime
import subprocess
import logging
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
from kraken import rpred
from kraken.containers import BaselineLine, Segmentation
from kraken.lib import models

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
sys.path.insert(0, str(Path(PROJECT_ROOT)))


def setup_simple_logging(logs_dir: str, task_name: str = "transcription", run_name: Optional[str] = None):
    """Initialize logging with file + console handlers using consistent formatting."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_{task_name}.log"
    
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info("=== %s Run Started | Run: %s ===", task_name.upper(), run_name)
    logger.info("Log file: %s", log_file)
    
    return logger, str(log_file)


def bbox_of_polygon(pts):
    """Compute bounding box from polygon points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def fix_base_line(line_data, input_img_dir):
    """Convert JSON line data to Kraken BaselineLine with clipped coordinates."""
    x0, y0, _, _ = bbox_of_polygon(line_data['boundary'])
    im = Image.open(input_img_dir).convert("L")
    w, h = im.size
    raw_bl = line_data['baseline']
    local_bl = [(x - x0, y - y0) for x, y in raw_bl]
    clipped_bl = [(max(0, min(w - 1, x)), max(0, min(h - 1, y))) for x, y in local_bl]

    line = BaselineLine(
        id="0",
        baseline=clipped_bl,
        boundary=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    )
    return line, im


def transcribe_image(
    img_inventory: Union[str, Path],seg_path: Union[str, Path],output_dir: Union[str, Path],input_img_dir: Union[str, Path],model,
    logs_dir: Optional[Union[str, Path]] = None,task_name: str = "ocr_transcription",run_name: Optional[str] = None,log_config: bool = True) -> dict:
    if logs_dir is None:
        logs_dir = Path(PROJECT_ROOT) / "logs" / "transcription"
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger, log_file = setup_simple_logging(
        logs_dir=str(logs_dir),
        task_name=task_name,
        run_name=run_name
    )
    
    if log_config:
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_commit = "unknown"
        
        config = {
            "run_name": run_name,
            "git_commit": git_commit,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_repr": str(model) if hasattr(model, '__str__') else repr(model),
            "segmentation_path": str(seg_path),
            "input_img_dir": str(input_img_dir),
            "output_dir": str(output_dir),
            "img_inventory": str(img_inventory),
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")}
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))
    
    logger.info("Loading inventory CSV: %s", img_inventory)
    filt_inv = pd.read_csv(img_inventory)
    filt_inv = filt_inv[filt_inv["was_removed"] == False]
    logger.info("Found %d valid image entries", len(filt_inv))

    folders = sorted(set(filt_inv.folder))
    logger.info("Processing %d folders", len(folders))

    total_processed = 0
    total_failed = 0
    total_empty = 0
    global_idx = 0

    for folder_idx, img_ in enumerate(folders, 1):
        logger.info("[%d/%d] Processing folder: %s", folder_idx, len(folders), img_)
        json_name = f"{img_}.json"
        json_path = Path(seg_path) / json_name

        if not json_path.exists():
            logger.warning("Skipping folder %s: segmentation JSON not found at %s", img_, json_path)
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        json_data_lines = json_data.get('lines', [])

        text_lines_set = set(filt_inv[filt_inv['folder'] == img_]['stem'])
        text_lines = sorted(text_lines_set,key=lambda x: int(x.split('_')[-1]))
        
        
        id_text_lines = [int(stem.split('_')[-1]) for stem in text_lines]

        output_txt_dir = Path(output_dir) / run_name / img_
        output_txt_dir.mkdir(parents=True, exist_ok=True)
        consolidated_dict: dict[int, str] = {}
        for idx, img_nm in zip(id_text_lines, text_lines):
            global_idx += 1
            try:
                filtered_img_path = Path(input_img_dir) / img_ / f"{img_nm}.png"
                if not filtered_img_path.exists():
                    logger.warning("Skipping %s: image not found at %s", img_nm, filtered_img_path)
                    total_failed += 1
                    continue

                if idx < 0 or idx > len(json_data_lines):
                    logger.warning("Skipping %s: invalid json index (%d) for lines array of length %d", 
                                 img_nm, idx, len(json_data_lines))
                    total_failed += 1
                    continue

                line_data = json_data_lines[idx]
                line, im = fix_base_line(line_data=line_data, input_img_dir=filtered_img_path)
                
                seg = Segmentation(
                    type="baselines",
                    imagename=str(filtered_img_path),
                    text_direction="horizontal-lr",
                    script_detection=False,
                    lines=[line])

                preds = list(rpred.rpred(model, im, seg))
                
                output_txt_path = output_txt_dir / f"{img_nm}.txt"
                with open(output_txt_path, "w", encoding="utf-8") as f:
                    if preds:
                        p = preds[0]
                        pred = getattr(p, "prediction", "")
                        consolidated_dict[idx] = pred 
                        f.write(pred + "\n")
                    else:
                        f.write("\n")
                        total_empty += 1
                        logger.debug("Empty prediction for %s", img_nm)
                
                total_processed += 1
                
                if global_idx % 20 == 0:
                    logger.info("Progress: %d lines processed | OK: %d | Failed: %d | Empty: %d",
                              global_idx, total_processed, total_failed, total_empty)
                    
            except Exception as e:
                logger.exception("Error processing %s: %s", img_nm, str(e))
                total_failed += 1
        consolidated_path = output_txt_dir.parent / f"{img_}_full.txt"
        with open(consolidated_path, "w", encoding="utf-8") as f:
            for idx in sorted(consolidated_dict.keys()):
                f.write(consolidated_dict[idx] + "\n")

    logger.info("=== TRANSCRIPTION SUMMARY ===")
    logger.info("Total lines attempted: %d", total_processed + total_failed)
    logger.info("Successfully processed: %d", total_processed)
    logger.info("Failed: %d", total_failed)
    logger.info("Empty predictions: %d", total_empty)
    logger.info("Output directory: %s", output_dir)
    logger.info("Log file: %s", log_file)
    