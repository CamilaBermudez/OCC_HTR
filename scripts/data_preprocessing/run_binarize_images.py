from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
import datetime
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.data_preprocessing.binarize_images import run_binarization_pipeline 


def main():
    load_dotenv()

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    parser = argparse.ArgumentParser(description="Run image binarization pipeline")

    parser.add_argument("--input-path", required=False)
    parser.add_argument("--output-base-dir", required=False)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)

    parser.add_argument("--gaussian-filter",type=int,nargs=2,default=[3, 3], help="Gaussian filter kernel size (example: --gaussian-filter 3 3)")
    parser.add_argument("--method",type=str, default="otsu_gaussian", help="Binarization method",)
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output files",)

    args = parser.parse_args()

    input_path = (Path(args.input_path) if args.input_path else project_root / "data"/ "processed"/ "extracted_lines"/ "extraction_20260427_221639")
    output_base_dir = (Path(args.output_base_dir) if args.output_base_dir else project_root / "data" / "processed"/ "binarized_images")

    logs_dir = (Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "binarization")

    run_name = (args.run_name or f"bin_{input_path.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    result = run_binarization_pipeline(
        input_path=input_path,
        output_base_dir=output_base_dir,
        logs_dir=str(logs_dir),
        run_name=run_name,
        gaussian_filter=tuple(args.gaussian_filter),
        method=args.method,
        dry_run=args.dry_run,
    )

    print(f"\n{'=' * 50}")
    print("BINARIZATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Folders:      {result.get('total_folders', 0)}")
    print(f"Total images: {result.get('total_images', 0)}")
    print(f"Success:      {result.get('success', 0)}")
    print(f"Failed:       {result.get('failed', 0)}")

    if result.get("success_rate_percent") is not None:
        print(f"Success rate: {result['success_rate_percent']:.1f}%")

    print(f"Output:       {result.get('output_dir')}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()