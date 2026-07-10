import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.transcribe_line_crops import transcribe_line_crops


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Run a kraken OCR model on a flat folder of pre-cropped line "
            "PNGs. Complementary to scripts/ocr/run_transcribe_img.py — "
            "that one is page-based (segmentation JSONs + per-page dirs), "
            "this one takes a flat folder such as the permanent "
            "validation set."
        ),
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Flat folder of *.png line crops (e.g. "
        "data/processed/annotated_samples/OCR/validation).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a kraken .mlmodel checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output root. Predictions land at output-dir/<run-name>/. "
        "Default: data/processed/transcription",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Run subdirectory name. Default: <model-stem>_line_crops_<TS>.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda:0", "cuda"],
        help="Device for kraken inference. Default: cpu.",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument(
        "--no-config-log",
        action="store_true",
        help="Disable configuration logging inside the function.",
    )

    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir else project_root / "data/processed/transcription"
    )
    logs_dir = (
        Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "transcribe_line_crops"
    )
    if args.run_name:
        run_name = args.run_name
    else:
        model_stem = Path(args.model_path).stem
        run_name = f"{model_stem}_line_crops_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    transcribe_line_crops(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        run_name=run_name,
        model_path=Path(args.model_path),
        device=args.device,
        logs_dir=logs_dir,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
