import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_preprocessing.plot_bounds import plot_all_images_with_bounds


def main():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    parser = argparse.ArgumentParser(description="Plot bounding boxes on images")

    parser.add_argument("--input-dir", required=False)
    parser.add_argument("--kraken-output-path", required=True)
    parser.add_argument("--output-dir", required=False)
    parser.add_argument("--font-size", type=int, default=20)
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument("--run-name", required=False)

    args = parser.parse_args()
    logger.info(f"Input dir: {args.input_dir}")
    logger.info(f"Kraken output: {args.kraken_output_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Input images: {len(list(Path(args.input_dir).glob('*.png')))} files")

    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else project_root / "data" / "raw" / "original_manuscript" / "reproduction14453_100"
    )
    kraken_output_path = Path(args.kraken_output_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data" / "processed" / "plotted_bounds"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "plotting"
    run_name = (
        args.run_name
        or f"plot_{input_dir.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    result = plot_all_images_with_bounds(
        input_dir=input_dir,
        kraken_output_path=kraken_output_path,
        output_dir=output_dir,
        font_size=args.font_size,
        logs_dir=str(logs_dir),
        run_name=run_name,
    )
    print(
        f"\nDone: {result['processed']}/{result['total']} images | " f"Output: {result['output']}\n"
    )


if __name__ == "__main__":
    main()
