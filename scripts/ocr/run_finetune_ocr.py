import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.finetune import finetune


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the Kraken OCR model on the augmented synthetic "
            "dataset. Splits source lines (not images) into train/val so "
            "augmented variants of one line never straddle the boundary. "
            "Use --smoke for a quick end-to-end sanity check before "
            "committing to a full run."
        )
    )

    parser.add_argument(
        "--augmented-folder",
        required=True,
        help="Augmented images directory (typically aug_<timestamp>/).",
    )
    parser.add_argument(
        "--labels-json",
        required=True,
        help="Corrected labels JSON (labels_<timestamp>/labels.json).",
    )
    parser.add_argument(
        "--base-model",
        required=False,
        help="Path to base .mlmodel. Default: models/ocr/catmus-medieval.mlmodel",
    )
    parser.add_argument(
        "--output-base-dir",
        required=False,
        help="Parent directory under which finetune_<timestamp>/ is created. "
        "Default: models/ocr/finetuned",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of source lines held out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the source-line shuffle and split (default: 42).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: restrict to --smoke-size source lines and "
        "run for --smoke-epochs epochs with fixed quit strategy. Use to "
        "verify the pipeline before committing to a full run.",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=50,
        help="When --smoke is set, number of source lines to use (default: 50).",
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=2,
        help="When --smoke is set, number of epochs (default: 2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=-1,
        help="Epoch budget for full runs. -1 + early stopping = train "
        "until convergence (default: -1).",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=5,
        help="Early-stopping patience in epochs without improvement (default: 5).",
    )
    parser.add_argument(
        "--lrate",
        type=float,
        default=1e-5,
        help="Learning rate. Fine-tune default 1e-5 keeps the base "
        "catmus weights anchored; the original ketos default (1e-3) "
        "overwrites real-manuscript features after a few epochs "
        "(default: 1e-5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="ketos -B batch size (default: 1, CPU-friendly).",
    )
    parser.add_argument(
        "--resize",
        choices=["add", "union", "both", "new", "fail"],
        default="union",
        help="ketos --resize codec strategy (default: union — safely add "
        "any characters absent from the base codec).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="ketos -d device (default: cpu). Use 'cuda:0' if a GPU is available.",
    )
    parser.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        help="By default, after training the script keeps only the best "
        "checkpoint as model_best.mlmodel and deletes the per-epoch "
        "model_*.mlmodel files (per-epoch metrics are preserved in "
        "epoch_stats.json/.md). Pass this flag to keep every checkpoint.",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
        help="Directory for the plain-text run log. Default: logs/finetune_ocr",
    )

    args = parser.parse_args()

    base_model = (
        Path(args.base_model)
        if args.base_model
        else project_root / "models/ocr/catmus-medieval.mlmodel"
    )
    output_base_dir = (
        Path(args.output_base_dir)
        if args.output_base_dir
        else project_root / "models/ocr/finetuned"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "finetune_ocr"

    finetune(
        augmented_folder=Path(args.augmented_folder),
        labels_json=Path(args.labels_json),
        base_model=base_model,
        output_base_dir=output_base_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        smoke=args.smoke,
        smoke_size=args.smoke_size,
        smoke_epochs=args.smoke_epochs,
        epochs=args.epochs,
        lag=args.lag,
        lrate=args.lrate,
        batch_size=args.batch_size,
        resize=args.resize,
        device=args.device,
        keep_all_checkpoints=args.keep_all_checkpoints,
        logs_dir=logs_dir,
    )


if __name__ == "__main__":
    main()
