import argparse
import datetime
import json
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
    parser.add_argument(
        "--cut-to-lines",
        action="store_true",
        help="Treat each corpus file as one stream of words and cut it into "
        "pseudo-lines whose length is drawn from --line-lengths-json. Use for "
        "paragraph-style sources (e.g. data/raw/medical_texts) that lack "
        "manuscript-style line breaks.",
    )
    parser.add_argument(
        "--line-lengths-json",
        required=False,
        help="Path to a JSON written by notebooks/ocr/ocr_line_length_stats.ipynb "
        "(contains a 'lengths' list of per-line word counts). Required with "
        "--cut-to-lines.",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Skip pattern filtering — every line becomes a sample under "
        "category 'all'. Use with paragraph corpora where every line should "
        "feed the synthetic-text generator.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for --cut-to-lines length sampling (default: 42).",
    )
    parser.add_argument(
        "--output-filename",
        required=False,
        default="cometa_categorized.json",
        help="Name of the JSON file written under output-dir/run-name. "
        "Default: cometa_categorized.json.",
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
    # the Roman-numeral pattern is the prebuilt strict matcher. Patterns are
    # ignored when --keep-all is set.
    patterns: dict | None = None
    if not args.keep_all:
        words = [w.strip() for w in args.word_patterns.split(",") if w.strip()]
        substrings = [s.strip() for s in (args.substring_patterns or "").split(",") if s.strip()]
        patterns = {w: word_pattern(w) for w in words}
        patterns.update({f"substring_{s}": substring_pattern(s) for s in substrings})
        if args.include_roman_numerals:
            patterns["roman_numeral"] = has_roman_numeral
        assert patterns, (
            "No patterns selected. Pass --word-patterns, --substring-patterns, "
            "--include-roman-numerals, or --keep-all."
        )

    length_pool: list[int] | None = None
    if args.cut_to_lines:
        assert args.line_lengths_json, "--cut-to-lines requires --line-lengths-json"
        lengths_doc = json.loads(Path(args.line_lengths_json).read_text(encoding="utf-8"))
        length_pool = lengths_doc["lengths"]
        assert length_pool, f"'lengths' field is empty in {args.line_lengths_json}"

    categorize_corpus(
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        patterns=patterns,
        logs_dir=logs_dir,
        run_name=run_name,
        output_filename=args.output_filename,
        cut_to_lines=args.cut_to_lines,
        length_pool=length_pool,
        keep_all=args.keep_all,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
