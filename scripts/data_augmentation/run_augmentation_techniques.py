import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_augmentation.augmentation_techniques import get_parchment_crops


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Extract empty-parchment crops from manuscript page scans by "
            "sampling random fixed-size windows and keeping the ones with "
            "lowest Canny edge density (= least text content)."
        )
    )

    parser.add_argument(
        "--input-folder",
        required=False,
        help="Directory of manuscript page images (*.jpg, *.png at top level). "
        "Default: runs/detect/predict-8",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        help="Output root for parchment crops. The per-run save directory is "
        "output-folder/run-name. Default: data/processed/synthetic_seeds/parchment_crops",
    )
    parser.add_argument(
        "--crop-size",
        required=False,
        type=int,
        default=200,
        help="Square crop size in pixels (default: 200).",
    )
    parser.add_argument(
        "--candidates-per-page",
        required=False,
        type=int,
        default=40,
        help="Number of random windows sampled per page before scoring (default: 40).",
    )
    parser.add_argument(
        "--keep-per-page",
        required=False,
        type=int,
        default=3,
        help="Maximum crops kept per page (after sorting by edge score) (default: 3).",
    )
    parser.add_argument(
        "--edge-threshold",
        required=False,
        type=float,
        default=6.0,
        help="Reject crops with Canny mean above this. Lower = stricter (default: 6.0).",
    )
    parser.add_argument(
        "--min-brightness",
        required=False,
        type=float,
        default=100.0,
        help="Reject crops with mean grayscale value below this (0-255). "
        "Filters out solid-dark regions like page borders / book spine "
        "that would otherwise pass the Canny filter. Default: 100.0.",
    )
    parser.add_argument(
        "--max-blue-fraction",
        required=False,
        type=float,
        default=0.002,
        help="Reject crops whose share of saturated-blue pixels exceeds "
        "this (0.0-1.0). Filters out illuminated-initial frames whose blue "
        "pigment Canny doesn't see. Default: 0.002 (0.2%%).",
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=0,
        help="RNG seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--plot-parchments",
        action="store_true",
        help="Show a preview grid of the first 16 kept crops. Off by default.",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument(
        "--run-name",
        required=False,
        help="Subdirectory name under output-folder. Default: parchments_<timestamp>.",
    )

    args = parser.parse_args()

    input_folder = (
        Path(args.input_folder) if args.input_folder else project_root / "runs/detect/predict-8"
    )
    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else project_root / "data/processed/synthetic_seeds/parchment_crops"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "parchment_crops"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"parchments_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    get_parchment_crops(
        input_dir=input_folder,
        output_dir=output_folder,
        run_name=run_name,
        crop_size=args.crop_size,
        candidates_page=args.candidates_per_page,
        keep_page=args.keep_per_page,
        edge_threshold=args.edge_threshold,
        min_brightness=args.min_brightness,
        max_blue_fraction=args.max_blue_fraction,
        seed=args.seed,
        plot_=args.plot_parchments,
    )


if __name__ == "__main__":
    main()
