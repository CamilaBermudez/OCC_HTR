from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
from datetime import datetime
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.ocr.transcribe_img import transcribe_image 
from kraken.lib import models


def main():
    load_dotenv()

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Transcribe segmented manuscript lines using Kraken OCR"
    )

    parser.add_argument("--seg-path", required=False)
    parser.add_argument("--input-img-dir", required=False)
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--img-inventory", required=False)
    parser.add_argument("--model-path", required=False)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)
    parser.add_argument("--device", required=False, default="cpu", choices=["cpu", "cuda"], help="Device for model inference")
    parser.add_argument("--no-config-log", action="store_true", help="Disable configuration logging inside the function")

    args = parser.parse_args()

    seg_path = (Path(args.seg_path) if args.seg_path else project_root / "data/processed/segmented_images/segmentation_20260505_172845")
    input_img_dir = (Path(args.input_img_dir) if args.input_img_dir else project_root / "data/processed/filtered_images/20260510_160650/original/kept")
    output_dir = (Path(args.output_dir) if args.output_dir else project_root / "data/processed/transcription")
    img_inventory = (Path(args.img_inventory) if args.img_inventory else project_root / "data/processed/filtered_images/20260510_160650/filter_tracking.csv")
    model_path = (Path(args.model_path) if args.model_path else project_root / "models/ocr/catmus-medieval.mlmodel")

    logs_dir = (Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "transcription")

    run_name = (args.run_name or f"ocr_{input_img_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model = models.load_any(str(model_path), device=args.device)

    transcribe_image(
        img_inventory=img_inventory,
        seg_path=seg_path,
        output_dir=output_dir,
        input_img_dir=input_img_dir,
        model=model,
        logs_dir=str(logs_dir),
        run_name=run_name,
        log_config=not args.no_config_log)

 

if __name__ == "__main__":
    main()