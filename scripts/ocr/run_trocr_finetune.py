import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.trocr_finetune import DEFAULT_DECODER_ID, DEFAULT_ENCODER_ID, finetune_trocr


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a Swin (image encoder) + BERT (text decoder) "
            "VisionEncoderDecoderModel on the hand-annotated manuscript "
            "line pool. Uses HuggingFace Seq2SeqTrainer with generation-"
            "time CER/WER on a held-out val split. Best checkpoint (by "
            "lowest val CER) is copied to <run_dir>/best_model/."
        ),
    )

    parser.add_argument(
        "--real-folder",
        required=False,
        help="Directory of <stem>.png + <stem>.gt.txt pairs. Default: "
        "data/processed/annotated_samples/OCR/full_annotated",
    )
    parser.add_argument(
        "--output-base-dir",
        required=False,
        help="Parent directory under which trocr_<timestamp>/ is created. "
        "Default: models/ocr/finetuned",
    )
    parser.add_argument(
        "--encoder-id",
        default=DEFAULT_ENCODER_ID,
        help=f"HuggingFace image encoder id. Default: {DEFAULT_ENCODER_ID}.",
    )
    parser.add_argument(
        "--decoder-id",
        default=DEFAULT_DECODER_ID,
        help=f"HuggingFace text decoder id. Default: {DEFAULT_DECODER_ID}.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of pairs held out for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the train/val split + trainer. Default: 42.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Max training epochs (early stopping may cut it short). Default: 20.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate. Default: 5e-5.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device training batch size. Default: 8. " "Halve this if you hit OOM on MPS.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Per-device eval batch size. Default: 8.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Tokenizer truncation length. Default: 128 (fits CATMuS lines).",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Beam-search width used at eval + inference time. Default: 4.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Block repeated n-grams during generation. Default: 3.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Beam-search length penalty. Default: 1.0.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Epochs without CER improvement before training halts. Default: 5.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0 (avoids MPS fork-safety issues).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | mps | cuda | cpu. Default auto picks mps > cuda > cpu.",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument(
        "--no-config-log",
        action="store_true",
        help="Disable configuration logging inside the function.",
    )

    args = parser.parse_args()

    real_folder = (
        Path(args.real_folder)
        if args.real_folder
        else project_root / "data/processed/annotated_samples/OCR/full_annotated"
    )
    output_base_dir = (
        Path(args.output_base_dir)
        if args.output_base_dir
        else project_root / "models/ocr/finetuned"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "trocr_finetune"

    finetune_trocr(
        real_folder=real_folder,
        output_base_dir=output_base_dir,
        encoder_id=args.encoder_id,
        decoder_id=args.decoder_id,
        val_fraction=args.val_fraction,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_target_length=args.max_target_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        early_stopping_patience=args.early_stopping_patience,
        dataloader_num_workers=args.dataloader_num_workers,
        device=args.device,
        logs_dir=logs_dir,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
