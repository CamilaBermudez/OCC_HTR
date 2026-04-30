from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
from datetime import datetime
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.data_preprocessing.filtering_noisy_images import run_filtering_pipeline


def main():
    load_dotenv()

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Run filtering pipeline for extracted manuscript lines"
    )

    parser.add_argument("--binarized-src", required=False)
    parser.add_argument("--extracted-src", required=False)
    parser.add_argument("--dst-base-dir", required=False)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)

    parser.add_argument("--size-thresholds", type=float,
        nargs="+", default=[0.03], help="List of size thresholds (example: --size-thresholds 0.03 0.05)",)

    parser.add_argument("--density-thresholds", type=float, nargs="+",
        default=[0.001, 0.997], help="List of density thresholds (example: --density-thresholds 0.001 0.997)",)

    args = parser.parse_args()

    binarized_src = (Path(args.binarized_src) if args.binarized_src else project_root / "data"/ "processed"/ "binarized_images"/ "20260427_233547")
    extracted_src = (Path(args.extracted_src) if args.extracted_src else project_root/ "data"/ "processed"/ "extracted_lines"/ "extraction_20260427_221639")
    dst_base_dir = (Path(args.dst_base_dir) if args.dst_base_dir else project_root / "data"/ "processed"/ "filtered_images")

    logs_dir = (Path(args.logs_dir) if args.logs_dir else project_root / "logs"/ "filtering")

    run_name = (args.run_name or f"filter_{binarized_src.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    result = run_filtering_pipeline(
        binarized_src=binarized_src,
        extracted_src=extracted_src,
        dst_base_dir=dst_base_dir,
        logs_dir=str(logs_dir),
        run_name=run_name,
        size_thresholds=args.size_thresholds,
        density_thresholds=args.density_thresholds,
    )

    print(f"\n{'=' * 50}")
    print("FILTERING SUMMARY")
    print(f"{'=' * 50}")
    print(f"Analyzed:   {result.get('total_analyzed', 0)}")
    print(f"Kept:       {result.get('kept', 0)}")
    print(f"Removed:    {result.get('removed', 0)}")
    print(f"Output:     {result.get('output_dir')}")
    print(f"Tracking:   {result.get('tracking_csv')}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()