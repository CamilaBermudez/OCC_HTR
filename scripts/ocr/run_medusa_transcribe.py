import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.medusa_transcribe import DEFAULT_MODEL_ID, DEFAULT_PROMPT, run_medusa_transcribe


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Run the ENC-PSL Medusa 0.2 medieval HTR VLM on a folder of "
            "pre-segmented line images. Output layout mirrors the kraken/"
            "catmus transcription pipeline so the evaluation tooling can "
            "compare predictions across models on identical inputs."
        ),
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder of line images. Either a flat folder of *.png OR a parent "
        "folder containing per-page subdirs of *.png (same layout as "
        "data/processed/extracted_lines/extraction_<ts>/).",
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
        help="Run subdirectory name. Default: medusa_<timestamp>.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id. Default: {DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | mps | cuda | cpu. Default auto picks mps > cuda > cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Lines per forward pass. Lower this if out-of-memory on MPS "
        "(BF16 inference of a 9B model is memory-heavy). Default: 4.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation cap per line (CATMuS lines are typically <100 chars). Default: 128.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="If set, only process the first N pages (subdirs). Useful for a "
        "smoke test before committing to a full-corpus run.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="The instruction passed to the VLM. The model card explicitly "
        "warns that results degrade if the prompt is modified, so override "
        "at your own risk.",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantize weights to fit in less RAM. 'none' loads in BF16 "
        "(~18 GB for Medusa 9B); '8bit' (~9 GB); '4bit' (~4.5 GB). "
        "Required on 16 GB Macs to avoid out-of-memory crashes. "
        "Uses bitsandbytes — best support on CUDA, partial on MPS, "
        "falls back to CPU compute on Apple Silicon if MPS quant kernels "
        "are unavailable (slower but won't crash).",
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
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "medusa_transcribe"
    run_name = args.run_name or f"medusa_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_medusa_transcribe(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        run_name=run_name,
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_pages=args.max_pages,
        prompt=args.prompt,
        quantization=args.quantization,
        logs_dir=logs_dir,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
