import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_augmentation.medieval_text_generation import (
    generate_medieval_text_dataset,
)


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Render synthetic medieval-text PNGs from a categorized-samples "
            "JSON. Applies probabilistic orthographic substitutions: long s "
            "(ſ) for non-final 's', rotunda r (ꝛ) for 'r' after a round "
            "letter (b/d/h/o/p/v/w/y). Outputs images + consolidated labels."
        )
    )

    parser.add_argument(
        "--input-json",
        required=False,
        help="Categorized-samples JSON (output of run_corpus_categorization.py). "
        "Default: data/processed/synthetic_seeds/cometa_categorized.json",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output root. The per-run save directory is output-dir/run-name. "
        "Default: data/processed/synthetic_seeds/medieval_text",
    )
    parser.add_argument(
        "--font-path",
        required=False,
        help="Path to the TTF/OTF font to render with. "
        "Default: fonts/merged_font_code_cmpl2.ttf",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=60,
        help="Pixel font size (default: 60).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=20,
        help="Padding around the text in pixels (default: 20).",
    )
    parser.add_argument(
        "--p-long-s-begin",
        type=float,
        default=0.95,
        help="Probability of substituting 's' at the start of a word (default: 0.95).",
    )
    parser.add_argument(
        "--p-long-s-middle",
        type=float,
        default=0.80,
        help="Probability of substituting 's' mid-word (default: 0.80).",
    )
    parser.add_argument(
        "--p-rotunda-r",
        type=float,
        default=0.70,
        help="Probability of substituting 'r' after a round letter (default: 0.70).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Per-sample seeds = base_seed + sample_index (default: 42).",
    )
    parser.add_argument(
        "--categories-filter",
        required=False,
        help="Comma-separated category names. If set, render only samples "
        "whose categories overlap. Default: render all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        required=False,
        help="Optional cap on rendered samples (useful for previewing).",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
        help="Directory for the run text log. Default: logs/medieval_text",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Subdirectory under output-dir. Default: medieval_<timestamp>.",
    )

    args = parser.parse_args()

    input_json = (
        Path(args.input_json)
        if args.input_json
        else project_root / "data/processed/synthetic_seeds/cometa_categorized.json"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data/processed/synthetic_seeds/medieval_text"
    )
    font_path = (
        Path(args.font_path)
        if args.font_path
        else project_root / "fonts/merged_font_code_cmpl2.ttf"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "medieval_text"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"medieval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    categories_filter: set[str] | None = None
    if args.categories_filter:
        categories_filter = {c.strip() for c in args.categories_filter.split(",") if c.strip()}

    generate_medieval_text_dataset(
        input_json=input_json,
        output_dir=output_dir,
        run_name=run_name,
        font_path=font_path,
        font_size=args.font_size,
        margin=args.margin,
        p_long_s_begin=args.p_long_s_begin,
        p_long_s_middle=args.p_long_s_middle,
        p_rotunda_r=args.p_rotunda_r,
        base_seed=args.base_seed,
        categories_filter=categories_filter,
        max_samples=args.max_samples,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
