import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.trocr_transcribe import transcribe_trocr


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe pre-segmented line images with a fine-tuned Swin+BERT "
            "TrOCR checkpoint. Output layout mirrors the kraken/catmus + "
            "medusa transcription pipelines so the evaluation tooling can "
            "score it identically."
        ),
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Fine-tuned checkpoint directory — the best_model/ subfolder "
        "produced by run_trocr_finetune.py.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder of line images. Either a flat folder of *.png OR a parent "
        "folder containing per-page subdirs of *.png.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output root. Per-page subdirs land at output-dir/<run-name>/. "
        "Default: data/processed/transcription",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Run subdirectory name. Default: trocr_<timestamp>.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | mps | cuda | cpu. Default auto picks mps > cuda > cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Lines per forward pass. Default: 8.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation cap per line. Default: 128.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam-search width. Default: 4.",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("auto", "pad", "stretch"),
        default="auto",
        help="Line-resize mode. 'auto' (default) reads resize_mode.txt from the "
        "model; models without it (trained before this flag) fall back to "
        "'stretch'. Override only to force a mode. See spec §6.5.18.",
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
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "trocr_transcribe"
    run_name = args.run_name or f"trocr_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    transcribe_trocr(
        model_dir=Path(args.model_dir),
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        run_name=run_name,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        resize_mode=args.resize_mode,
        logs_dir=logs_dir,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
