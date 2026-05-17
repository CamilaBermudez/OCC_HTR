import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.tokenizer.BPE_tokenizer import train_occitan_htr_tokenizer


def main():
    load_dotenv()

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(description="Train BPE tokenizer for HTR")

    parser.add_argument(
        "--input_path",
        type=str,
        default=str(project_root / "data" / "processed" / "tokenizer_corpora"),
        help="Directory containing .txt corpus files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(project_root / "data" / "processed" / "tokenizer"),
        help="Directory to save tokenizer outputs",
    )
    parser.add_argument(
        "--type", type=str, default="byte", help="Tokenizer type byte or char, default byte"
    )
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size. Default 100")
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=str(project_root / "logs" / "tokenizer"),
        help="Directory to save logs",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    type = Path(args.type)
    vocab_size = args.vocab_size
    logs_dir = Path(args.logs_dir)

    run_name = args.run_name or f"tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train_occitan_htr_tokenizer(
        input_path=input_path,
        output_path=output_path,
        type=type,
        vocab_size=vocab_size,
        logs_dir=logs_dir,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
