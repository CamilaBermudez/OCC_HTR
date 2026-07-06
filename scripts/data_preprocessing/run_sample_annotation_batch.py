import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.data_preprocessing.sample_annotation_batch import run_sample_annotation_batch


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Sample a fresh batch of line stems for hand-annotation. "
            "Copies each stem's PNG + a pre-filled .gt.txt (seeded from "
            "the catmus baseline OCR) into a new dated subfolder under "
            "--output-root. Excludes anything already in --exclude-folder."
        ),
    )

    parser.add_argument(
        "--source-lines-dir",
        required=False,
        help="Root of <page>/*.png line crops. Default: "
        "data/processed/filtered_images/20260618_160948/original/kept",
    )
    parser.add_argument(
        "--ocr-seed-dir",
        required=False,
        help="Root of <page>/*.txt OCR predictions to pre-fill .gt.txt. "
        "Default: data/processed/transcription/ocr_kept_20260622_120413",
    )
    parser.add_argument(
        "--exclude-folder",
        action="append",
        required=False,
        help="Folder whose <stem>.gt.txt files mark already-annotated or "
        "held-out stems. Repeat once per folder — every stem across all "
        "listed folders is excluded. Non-existent folders are skipped "
        "with a warning. Default: --exclude-folder for both "
        "data/processed/annotated_samples/OCR/full_annotated and "
        "data/processed/annotated_samples/OCR/validation.",
    )
    parser.add_argument(
        "--output-subfolder-prefix",
        default="real_val_sample",
        help="Prefix for the created batch subfolder — default "
        "'real_val_sample' matches historic training batches. Use "
        "'validation' when sampling the permanent held-out benchmark set.",
    )
    parser.add_argument(
        "--output-root",
        required=False,
        help="Where to create the batch subfolder. Default: tests/ocr",
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=100,
        help="Number of stems to pick (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42). Prior batches "
        "used seeds 42-48; increment for each new batch.",
    )
    parser.add_argument(
        "--pattern",
        required=False,
        help="Optional regex to filter OCR seeds. Only seeds matching "
        "the regex are eligible. Example for capital C or E: "
        "'(?<![A-Za-z])[CE]'.",
    )
    parser.add_argument(
        "--pattern-label",
        required=False,
        help="Human-readable label for the pattern, written into "
        "_README.md (e.g. 'capital C or E targeted').",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Subfolder name suffix + log basename. " "Default: <YYYYMMDD_HHMMSS>.",
    )
    parser.add_argument(
        "--no-config-log",
        action="store_true",
        help="Disable configuration logging inside the function.",
    )

    args = parser.parse_args()

    source_lines_dir = (
        Path(args.source_lines_dir)
        if args.source_lines_dir
        else project_root / "data/processed/filtered_images/20260618_160948/original/kept"
    )
    ocr_seed_dir = (
        Path(args.ocr_seed_dir)
        if args.ocr_seed_dir
        else project_root / "data/processed/transcription/ocr_kept_20260622_120413"
    )
    if args.exclude_folder:
        exclude_folders = [Path(p) for p in args.exclude_folder]
    else:
        exclude_folders = [
            project_root / "data/processed/annotated_samples/OCR/full_annotated",
            project_root / "data/processed/annotated_samples/OCR/validation",
        ]
    output_root = Path(args.output_root) if args.output_root else project_root / "tests/ocr"
    logs_dir = (
        Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "sample_annotation_batch"
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_sample_annotation_batch(
        source_lines_dir=source_lines_dir,
        ocr_seed_dir=ocr_seed_dir,
        exclude_folders=exclude_folders,
        output_root=output_root,
        n_target=args.n_target,
        seed=args.seed,
        pattern=args.pattern,
        pattern_label=args.pattern_label,
        logs_dir=logs_dir,
        run_name=run_name,
        log_config=not args.no_config_log,
        output_subfolder_prefix=args.output_subfolder_prefix,
    )


if __name__ == "__main__":
    main()
