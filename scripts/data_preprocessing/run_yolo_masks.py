from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
import sys
import datetime
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.data_preprocessing.yolo_masks import build_mask_yolo


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate YOLO masks")

    parser.add_argument("--model-path", required=True)
    parser.add_argument("--images-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)

    args = parser.parse_args()

    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")

    model_path = args.model_path
    images_path = args.images_path
    output_path = args.output_path

    logs_dir = args.logs_dir or os.path.join(PROJECT_ROOT, "logs", "mask_generation")

    run_name = args.run_name or f"mask_{Path(images_path).name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result = build_mask_yolo(
        model_path=model_path,
        images_path=images_path,
        output_path=output_path,
        logs_dir=logs_dir,
        run_name=run_name
    )

    print(f"Masks generated: {result['summary']['images_with_mainzone']}/{result['summary']['total_images']} images")
    print(f"Masks saved to: {result['masks_dir']}")

    if result.get("config_file"):
        print(f"Config logged to: {result['config_file']}")


if __name__ == "__main__":
    main()