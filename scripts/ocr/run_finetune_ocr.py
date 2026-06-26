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
        required=False,
        help="Augmented images directory (typically aug_<timestamp>/). "
        "Required unless --no-synth-train is set.",
    )
    parser.add_argument(
        "--labels-json",
        required=False,
        help="Corrected labels JSON (labels_<timestamp>/labels.json). "
        "Required unless --no-synth-train is set.",
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
        help="Smoke-test mode: restrict to --smoke-size source lines. "
        "Uses the same early-stopping quit strategy as the full run, so "
        "--smoke-epochs is a MAX-epoch cap rather than a fixed count; "
        "training terminates as soon as val_accuracy plateaus for --lag "
        "epochs. Use to verify the pipeline or to sweep ideas quickly "
        "before committing to a full-corpus run.",
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
        help="When --smoke is set, max epochs before training is force-"
        "stopped (default: 2). With early stopping active this is just "
        "the upper bound; runs typically finish sooner on plateau.",
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
    parser.add_argument(
        "--real-folder",
        required=False,
        help="Directory of real-manuscript line crops as <stem>.png + "
        "<stem>.gt.txt pairs (kraken convention). When given together "
        "with --n-real-train/--n-real-val, those samples are mixed into "
        "the staged train and val splits so the model sees real data "
        "alongside the synthetic augmentations.",
    )
    parser.add_argument(
        "--real-train-frac",
        type=float,
        default=0.0,
        help="Fraction of the real folder used as TRAIN mix-in (default "
        "0.0 — no mix-in). Floor-rounded against the live folder count "
        "so the split auto-scales as the corrected pool grows.",
    )
    parser.add_argument(
        "--real-val-frac",
        type=float,
        default=0.0,
        help="Fraction of the real folder used as VAL (default 0.0). "
        "With --real-replaces-synth-val=true (the default) the synthetic "
        "val list is rewritten to contain ONLY these real samples, so "
        "val_accuracy genuinely measures real-manuscript performance. "
        "real-train-frac + real-val-frac must be <= 1.0.",
    )
    parser.add_argument(
        "--real-replaces-synth-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether real val samples replace (default) or append to "
        "the synthetic val split. Use --no-real-replaces-synth-val to "
        "keep the synthetic samples in val.",
    )
    parser.add_argument(
        "--no-synth-train",
        action="store_true",
        help="Skip the synthetic stage entirely — train + val come "
        "from --real-folder only. Use this for Phase 1 (catmus + small "
        "verified real set) when there is no benefit from re-teaching "
        "the model the generic medieval distribution it already knows.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --augment to ketos (random per-batch transforms) "
        "instead of the default --no-augment. Recommended when the "
        "training pool is small (e.g. real-only) so the model sees "
        "more visual variation without needing more annotated lines.",
    )

    args = parser.parse_args()
    if not args.no_synth_train:
        assert args.augmented_folder and args.labels_json, (
            "--augmented-folder and --labels-json are required unless "
            "--no-synth-train is passed."
        )

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
        augmented_folder=Path(args.augmented_folder) if args.augmented_folder else None,
        labels_json=Path(args.labels_json) if args.labels_json else None,
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
        real_folder=Path(args.real_folder) if args.real_folder else None,
        real_train_frac=args.real_train_frac,
        real_val_frac=args.real_val_frac,
        real_replaces_synth_val=args.real_replaces_synth_val,
        no_synth_train=args.no_synth_train,
        ketos_augment=args.augment,
    )


if __name__ == "__main__":
    main()
