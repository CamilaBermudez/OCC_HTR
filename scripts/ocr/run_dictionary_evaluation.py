import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.dictionary_evaluation import run_dictionary_evaluation


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Evaluate OCR transcription quality against a reference dictionary",
    )

    parser.add_argument(
        "--transcription-dir",
        required=False,
        help="Folder containing the *.txt transcription files (top-level glob)",
    )
    parser.add_argument(
        "--dictionary-path",
        required=False,
        help="Path to the JSON dictionary (lemma -> variants)",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Where to save per-run CSV/JSON artifacts (omit to skip artifact saving)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=82,
        help="Base fuzzy-match score threshold (0-100). Default: 82",
    )
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=2,
        help="Drop OCR tokens shorter than this length. Default: 2",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)
    parser.add_argument(
        "--no-config-log",
        action="store_true",
        help="Disable configuration logging inside the function",
    )

    args = parser.parse_args()

    transcription_dir = (
        Path(args.transcription_dir)
        if args.transcription_dir
        else project_root / "data/processed/transcription/ocr_kept_20260515_104644"
    )
    dictionary_path = (
        Path(args.dictionary_path)
        if args.dictionary_path
        else project_root / "data/raw/DOM_lemma_variants.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "dictionary_eval"
    run_name = args.run_name or f"dict_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dictionary_evaluation(
        transcription_dir=transcription_dir,
        dictionary_path=dictionary_path,
        output_dir=output_dir,
        fuzzy_threshold=args.fuzzy_threshold,
        min_word_length=args.min_word_length,
        logs_dir=logs_dir,
        run_name=run_name,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
