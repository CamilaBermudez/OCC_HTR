import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_preprocessing.ink_bleed_detection import detect_ink_bleed


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Score every line image in --input-folder for ink bleed and "
            "write a JSON marking which images show signs of bleed. "
            "Uses Otsu-based background uniformity and intermediate-pixel "
            "ratio as heuristics; threshold is tunable via --bleed-threshold."
        )
    )

    parser.add_argument(
        "--input-folder",
        required=True,
        help="Directory of line images (*.png, *.jpg, *.jpeg at top level). "
        "Typically the filtered/kept lines folder.",
    )
    parser.add_argument(
        "--output-base-dir",
        required=False,
        help="Parent directory under which bleed_detection_<timestamp>/ is "
        "created. Default: data/processed/ink_bleed_detection",
    )
    parser.add_argument(
        "--bleed-threshold",
        type=float,
        default=0.35,
        help="An image is flagged as has_bleed if bleed_score >= this value "
        "(default: 0.35). Inspect bleed_score_distribution in the output "
        "JSON summary to recalibrate.",
    )
    parser.add_argument(
        "--w-bg-std",
        type=float,
        default=0.6,
        help="Weight of the background-uniformity sub-score in the "
        "composite bleed_score (default: 0.6).",
    )
    parser.add_argument(
        "--w-intermediate",
        type=float,
        default=0.4,
        help="Weight of the intermediate-pixel-ratio sub-score (default: 0.4).",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
        help="Directory for the plain-text run log. Default: logs/ink_bleed_detection",
    )

    args = parser.parse_args()

    output_base_dir = (
        Path(args.output_base_dir)
        if args.output_base_dir
        else project_root / "data/processed/ink_bleed_detection"
    )
    logs_dir = (
        Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "ink_bleed_detection"
    )

    detect_ink_bleed(
        images_dir=Path(args.input_folder),
        output_base_dir=output_base_dir,
        bleed_threshold=args.bleed_threshold,
        w_bg_std=args.w_bg_std,
        w_intermediate=args.w_intermediate,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
