import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.evaluate_ocr import run_evaluate_ocr


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Compute CER/WER for one or more OCR prediction folders "
        "against a hand-verified ground-truth folder."
    )

    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Folder of <stem>.gt.txt reference files.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="A model's prediction folder, formatted as 'name=path'. "
        "Repeat once per model. Example: "
        "--pred catmus=./data/processed/transcription/ocr_kept_20260622_120413",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Root for per-run artefacts. Default: tests/ocr/evaluations",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Subdirectory + basename for this run. Default: eval_<timestamp>.",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument(
        "--no-config-log",
        action="store_true",
        help="Disable configuration logging inside the function.",
    )

    args = parser.parse_args()

    predictions: list[tuple[str, Path]] = []
    for spec in args.pred:
        assert "=" in spec, f"--pred expects 'name=path', got: {spec!r}"
        name, path = spec.split("=", 1)
        predictions.append((name, Path(path)))

    output_dir = (
        Path(args.output_dir) if args.output_dir else project_root / "tests/ocr/evaluations"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "evaluate_ocr"
    run_name = args.run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_evaluate_ocr(
        gt_dir=Path(args.gt_dir),
        predictions=predictions,
        output_dir=output_dir,
        logs_dir=logs_dir,
        run_name=run_name,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
