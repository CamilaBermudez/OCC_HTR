from pathlib import Path
from dotenv import load_dotenv
import argparse
import os
import datetime
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.data_preprocessing.resize_images import resize_all_images  


def main():
    load_dotenv()

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Resize all filtered manuscript images"
    )

    parser.add_argument("--input-folder", required=False)
    parser.add_argument("--output-folder", required=False)
    parser.add_argument("--target-size", type=int, default=224, help="Target size for resized images (default: 224)")
    parser.add_argument("--logs-dir", required=False, help="Optional: directory for log files")
    parser.add_argument("--run-name", required=False, help="Optional: custom name for this run")

    args = parser.parse_args()

    input_folder = (Path(args.input_folder) if args.input_folder else project_root/ "data" / "processed"/ "filtered_images"/ "20260430_224958"/ "original"/ "kept")
    output_folder = (Path(args.output_folder) if args.output_folder else project_root / "data" / "processed" / "resized_samples")

    logs_dir = (Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "resizing")
    run_name = (args.run_name or f"resize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")


    resize_all_images(
        input_dir=input_folder,
        output_dir=output_folder,
        target_size=args.target_size,
        logs_dir=str(logs_dir),
        run_name=run_name)

    print(
        f"\nDone: Images resized successfully | "
        f"Output: {output_folder} | "
        f"Target size: {args.target_size}\n"
    )


if __name__ == "__main__":
    main()