import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_preprocessing.image_segmentation import segment_all_images


def main():
    load_dotenv()
    print("Script started")  # add this
    print(f"CWD: {os.getcwd()}")
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(description="Segment images using masks")

    parser.add_argument("--input-folder", required=False)
    parser.add_argument("--output-folder", required=False)
    parser.add_argument("--masks-folder", required=True)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)

    args = parser.parse_args()

    input_folder = (
        Path(args.input_folder)
        if args.input_folder
        else project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    )
    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else project_root / "data" / "processed" / "segmented_images"
    )
    masks_folder = Path(args.masks_folder)

    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "segmentation"
    run_name = (
        args.run_name
        or f"seg_{input_folder.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    result = segment_all_images(
        input_folder=input_folder,
        output_folder=output_folder,
        masks_folder=masks_folder,
        logs_dir=str(logs_dir),
        run_name=run_name,
    )
    print(f"Result: {result}")
    print(
        f"\nDone: {result['success']}/{result['total']} images | " f"Output: {result['output']}\n"
    )


if __name__ == "__main__":
    main()
