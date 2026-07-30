import argparse
import datetime
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_augmentation.augmentation_techniques import batch_augment_directory


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Apply the Option-A augmentation pipeline (ink degradation + real "
            "parchment composite + page warp + scan-capture effects) to every "
            "image in --input-folder and save the augmented variants for later use."
        )
    )

    parser.add_argument(
        "--input-folder",
        required=True,
        help="Directory of source images (*.png, *.jpg at top level).",
    )
    parser.add_argument(
        "--parchment-folder",
        required=False,
        help="Directory of parchment crops produced by run_augmentation_techniques.py. "
        "Default: the most-recently-modified run subdir under "
        "data/processed/synthetic_seeds/parchment_crops/.",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        help="Output root for augmented images. The per-run save directory is "
        "output-folder/run-name. Default: data/processed/synthetic_seeds/augmented_images",
    )
    parser.add_argument(
        "--n-augmentations",
        type=int,
        default=1,
        help="Number of augmented variants generated per source image (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        help="Base RNG seed. Per-call seeds are derived as "
        "seed + image_index * n_augmentations + variant_index. "
        "If omitted, augmentations are non-deterministic.",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Subdirectory name under output-folder. Default: aug_<timestamp>.",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Preview mode: only process the first N source images "
        "(default: 5, override with --sample-size). Useful for quickly "
        "verifying the pipeline produces what you expect before running "
        "against the whole dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="When --sample is set, how many source images to process. "
        "Ignored when --sample is not set. Default: 5.",
    )
    parser.add_argument(
        "--target-line-height",
        type=int,
        default=40,
        help="Final output line height in px (aspect ratio preserved), applied "
        "AFTER augmentation so effects render at full scale then downsample like "
        "a real scan. Default 40 matches the real crops (~38-39 px, ~400 wide). "
        "Pass 0 to keep the native ~115 px render size. See spec §6.5.18.",
    )

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    assert input_folder.is_dir(), f"--input-folder is not a directory: {input_folder}"

    # Resolve the parchment-crop directory.
    if args.parchment_folder:
        parchment_dir = Path(args.parchment_folder)
    else:
        parchment_root = project_root / "data/processed/synthetic_seeds/parchment_crops"
        subdirs = [d for d in parchment_root.glob("*") if d.is_dir()]
        assert subdirs, (
            f"No parchment runs found under {parchment_root}. Run "
            "scripts/data_augmentation/run_augmentation_techniques.py first, "
            "or pass --parchment-folder explicitly."
        )
        parchment_dir = max(subdirs, key=lambda p: p.stat().st_mtime)

    parchment_files = sorted(parchment_dir.glob("parchment_*.png"))
    assert parchment_files, f"No parchment_*.png found in {parchment_dir}"
    print(f"Using {len(parchment_files)} parchment crops from {parchment_dir}")

    output_folder = (
        Path(args.output_folder)
        if args.output_folder
        else project_root / "data/processed/synthetic_seeds/augmented_images"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "augmentation"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"aug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    batch_augment_directory(
        input_dir=input_folder,
        output_dir=output_folder,
        run_name=run_name,
        parchment_files=parchment_files,
        n_augmentations=args.n_augmentations,
        seed=args.seed,
        logs_dir=str(logs_dir),
        sample_size=args.sample_size if args.sample else None,
        target_line_height=args.target_line_height if args.target_line_height else None,
    )


if __name__ == "__main__":
    main()
