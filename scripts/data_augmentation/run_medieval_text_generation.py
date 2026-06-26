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
        "Default: data/processed/synthetic_seeds",
    )
    parser.add_argument(
        "--font-path",
        required=False,
        help="Path to a single TTF/OTF font to render with. "
        "Default: fonts/merged_font_code_cmpl2.ttf. Ignored when "
        "--fonts-dir is given.",
    )
    parser.add_argument(
        "--fonts-dir",
        required=False,
        help="Directory of *.ttf / *.otf fonts. When set, the generator "
        "loads every font in the directory and picks one at random per "
        "line, so the synthetic corpus has multiple scribal-hand "
        "variations. Per-line text is rewritten (long-s -> s, rotunda-r "
        "-> r) when the chosen font lacks the glyph, and the saved "
        "label matches the rewrite. Overrides --font-path.",
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
        "--p-tironian-et",
        type=float,
        default=0.0,
        help="Probability of inserting ⁊ after each sentence terminator (.?!). "
        "Requires --et-stamp-dir; default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--et-stamp-dir",
        required=False,
        help="Directory of cropped ⁊ glyphs (PNG/JPG) from real manuscript "
        "pages. Each crop should show one ⁊ on parchment; alpha is derived "
        "from pixel darkness, so no transparency work is required.",
    )
    parser.add_argument(
        "--c-stamp-dir",
        required=False,
        help="Directory of cropped illuminated 'C' initials from real "
        "manuscript pages. When the literal word 'Capitol' appears, the "
        "C is composited from this folder and the rest of the word is "
        "rendered in rubric red. Pass an empty/non-existent path to disable.",
    )
    parser.add_argument(
        "--p-capital-e",
        type=float,
        default=0.0,
        help="Probability of swapping a word-initial E/e for an illuminated "
        "E stamp (with the rest of the word rendered in rubric red). "
        "Requires --e-stamp-dir; default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--e-stamp-dir",
        required=False,
        help="Directory of cropped illuminated 'E' initials. Used with "
        "--p-capital-e to render some sentence-starting E-words in the "
        "rubric style.",
    )
    parser.add_argument(
        "--p-abbreviation",
        type=float,
        default=0.0,
        help="Per-character probability of swapping a base letter "
        "(a/e/l/m/n/o/p/q/r) for a scribal-abbreviation stamp (ñ, q̃, "
        "õ, …). Requires --abbrev-base-dir; default: 0.0 (disabled).",
    )
    parser.add_argument(
        "--abbrev-base-dir",
        required=False,
        help="Root directory containing subfolders of abbreviation stamps "
        "(e_tilde/, n_tilde/, q_tilde/, etc.). The mapping from folder "
        "name to label is hard-coded in ABBREV_MAP.",
    )
    parser.add_argument(
        "--max-abbreviation-per-line",
        type=int,
        default=3,
        help="Hard cap on abbreviation substitutions per input line. "
        "Without this a long sentence ends up with 5-8 stamps even at "
        "modest --p-abbreviation. Default: 3.",
    )
    parser.add_argument(
        "--max-abbreviation-per-word",
        type=int,
        default=1,
        help="Hard cap on abbreviation substitutions per WORD. Two stamps "
        "landing in the same word (e.g. 'autr̃ẽiat' with both r̃ and ẽ) "
        "looks crowded and unlike the reference manuscript. Default: 1.",
    )
    parser.add_argument(
        "--enable-pattern-stamps",
        action="store_true",
        help="Enable syllable / ligature pattern stamps (am, an, au, cum, "
        "em, ma, me, mi, mu, nu, um, un, x, standalone-o). Each pattern "
        "has its own probability (~80%%, with x and standalone-o at "
        "100%%) defined in PATTERN_STAMPS_CFG. Stamps are loaded from "
        "subfolders of --abbrev-base-dir matching the folder name in "
        "the config.",
    )
    parser.add_argument(
        "--p-end-decor",
        type=float,
        default=0.0,
        help="Probability of pasting a purely-decorative end-of-line "
        "stamp from <abbrev-base-dir>/end_decor/. The label is NOT "
        "modified — the model must learn to ignore the mark. Default: "
        "0.0 (disabled).",
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
        help="Subdirectory under output-dir. Default: medieval_text_<timestamp>.",
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
        else project_root / "data/processed/synthetic_seeds"
    )
    font_path = (
        Path(args.font_path)
        if args.font_path
        else project_root / "fonts/merged_font_code_cmpl2.ttf"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "medieval_text"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"medieval_text_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    categories_filter: set[str] | None = None
    if args.categories_filter:
        categories_filter = {c.strip() for c in args.categories_filter.split(",") if c.strip()}

    generate_medieval_text_dataset(
        input_json=input_json,
        output_dir=output_dir,
        run_name=run_name,
        font_path=font_path,
        fonts_dir=Path(args.fonts_dir) if args.fonts_dir else None,
        font_size=args.font_size,
        margin=args.margin,
        p_long_s_begin=args.p_long_s_begin,
        p_long_s_middle=args.p_long_s_middle,
        p_rotunda_r=args.p_rotunda_r,
        p_tironian_et=args.p_tironian_et,
        et_stamp_dir=Path(args.et_stamp_dir) if args.et_stamp_dir else None,
        c_stamp_dir=Path(args.c_stamp_dir) if args.c_stamp_dir else None,
        p_capital_e=args.p_capital_e,
        e_stamp_dir=Path(args.e_stamp_dir) if args.e_stamp_dir else None,
        p_abbreviation=args.p_abbreviation,
        abbrev_base_dir=Path(args.abbrev_base_dir) if args.abbrev_base_dir else None,
        max_abbreviation_per_line=args.max_abbreviation_per_line,
        max_abbreviation_per_word=args.max_abbreviation_per_word,
        enable_pattern_stamps=args.enable_pattern_stamps,
        p_end_decor=args.p_end_decor,
        base_seed=args.base_seed,
        categories_filter=categories_filter,
        max_samples=args.max_samples,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
