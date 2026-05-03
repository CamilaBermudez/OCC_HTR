import os
from pathlib import Path
import re
import argparse
from datetime import datetime
from dotenv import load_dotenv
import sys
import logging
from typing import Union, Optional

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))

import logging
from pathlib import Path

def setup_logger(logs_dir: Union[str, Path], run_name: str):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / f"{run_name}.log"

    logger = logging.getLogger("corpus")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_file}")

    return logger


def unify_corpus(input_dir: Path, output_dir: Path,logs_dir: Optional[str] = None,run_name: Optional[str] = None) -> list[str]:
    texts = []
    total_chars = 0
    processed_files = 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if logs_dir:
        logger = setup_logger(logs_dir, run_name)
        logger.info(f"=== Plotting Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("plotting")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        
    logger.info(f"Starting corpus build from: {input_dir}")

    for txt_file in input_dir.glob("*.txt"):
        logger.info(f"Processing file: {txt_file}")

        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read()

            original_length = len(content)

            content = re.sub(r'<[^>]+>', '', content)  # Remove XML/HTML tags
            content = re.sub(r'\[.*?\]', '', content)   # Remove editorial notes
            content = re.sub(r'\s+', ' ', content).strip() # Remove white spaces

            cleaned_length = len(content)

            logger.info(f"{txt_file.name}: {original_length} -> {cleaned_length} chars after cleaning")

            if cleaned_length > 100:
                texts.append(content)
                total_chars += cleaned_length
                processed_files += 1
            else:
                logger.info(f"Skipped (too short): {txt_file.name}")

        except Exception as e:
            logger.error(f"Failed processing {txt_file}: {e}")

    output_file = output_dir / f"{run_name}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n<doc>\n\n".join(texts))

    logger.info("=== Corpus Summary ===")
    logger.info(f"Files included: {processed_files}")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Saved to: {output_file}")

    return texts

