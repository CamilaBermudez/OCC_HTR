import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_augmentation.corpus_categorization import (
    categorize_corpus,
    has_roman_numeral,
    substring_pattern,
    word_pattern,
)


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Scan a corpus of *.txt files and label every line that contains "
            "any of the requested patterns. Whole-word patterns are specified "
            "as a comma-separated list; the strict Roman-numeral pattern is "
            "an opt-in toggle (default on)."
        )
    )

    parser.add_argument(
        "--corpus-dir",
        required=False,
        help="Directory of *.txt corpus files. " "Default: data/raw/COMETA_medieval_corpus",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output root. The JSON file lands at output-dir/run-name/"
        "cometa_categorized.json. Default: data/processed/synthetic_seeds",
    )
    parser.add_argument(
        "--word-patterns",
        required=False,
        default="am,ma",
        help="Comma-separated list of whole-word patterns (case-insensitive). "
        "Each becomes its own category. Default: am,ma",
    )
    parser.add_argument(
        "--substring-patterns",
        required=False,
        default="",
        help="Comma-separated list of substring patterns (case-insensitive). "
        "Each fires when the substring appears anywhere in the line, "
        "including inside larger words (e.g. 'un' matches 'unexpected'). "
        "To avoid collisions with --word-patterns, category names are "
        "prefixed with 'substring_'. Default: empty.",
    )
    parser.add_argument(
        "--include-roman-numerals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also label lines containing strict Roman numerals "
        "(category 'roman_numeral'). Default: on. Pass "
        "--no-include-roman-numerals to disable.",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
        help="Directory for the run text log. " "Default: logs/corpus_categorization",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Subdirectory/log identifier for this run. " "Default: categorize_<timestamp>.",
    )

    args = parser.parse_args()

    corpus_dir = (
        Path(args.corpus_dir)
        if args.corpus_dir
        else project_root / "data/raw/COMETA_medieval_corpus"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data/processed/synthetic_seeds"
    )
    logs_dir = (
        Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "corpus_categorization"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"categorize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build the pattern dict from CLI flags. Whole-word patterns come from
    # --word-patterns; substring patterns from --substring-patterns (category
    # name prefixed with 'substring_' to avoid collisions with word patterns);
    # the Roman-numeral pattern is the prebuilt strict matcher.
    words = [w.strip() for w in args.word_patterns.split(",") if w.strip()]
    substrings = [s.strip() for s in (args.substring_patterns or "").split(",") if s.strip()]
    patterns = {w: word_pattern(w) for w in words}
    patterns.update({f"substring_{s}": substring_pattern(s) for s in substrings})
    if args.include_roman_numerals:
        patterns["roman_numeral"] = has_roman_numeral

    assert patterns, (
        "No patterns selected. Pass --word-patterns, --substring-patterns, "
        "or --include-roman-numerals."
    )

    categorize_corpus(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        patterns=patterns,
        logs_dir=logs_dir,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
