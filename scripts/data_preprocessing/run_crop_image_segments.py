from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
import datetime
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.data_preprocessing.crop_image_segments import crop_all_images  


def main():
    load_dotenv()

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Crop extracted lines from segmented manuscript images"
    )

    parser.add_argument("--input-folder", required=False)
    parser.add_argument("--output-kraken-path", required=True)
    parser.add_argument("--output-folder", required=False)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)

    args = parser.parse_args()

    input_folder = (Path(args.input_folder) if args.input_folder else project_root / "data"/ "raw"/ "original_manuscript"/ "reproduction14453_100")
    output_kraken_path = Path(args.output_kraken_path)
    output_folder = (Path(args.output_folder) if args.output_folder else project_root/ "data"/ "processed"/ "extracted_lines")
    logs_dir = (Path(args.logs_dir) if args.logs_dir else project_root/ "logs"/ "cropping")

    run_name = (args.run_name or f"crop_{input_folder.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    result = crop_all_images(input_folder=input_folder,output_kraken_path=output_kraken_path,
                             output_folder=output_folder, logs_dir=str(logs_dir),run_name=run_name)

    print(
        f"\nDone: {result['success']}/{result['total']} images | "
        f"Output: {result['output']}\n")


if __name__ == "__main__":
    main()