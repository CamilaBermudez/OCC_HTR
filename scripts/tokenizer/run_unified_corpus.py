import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
import sys
sys.path.insert(0, str(Path(os.environ.get("PROJECT_ROOT", "."))))
from src.tokenizer.unified_corpus import unify_corpus, setup_logger  # adjust import


def main():
        load_dotenv()

        project_root = Path(os.environ.get("PROJECT_ROOT", "."))
        parser = argparse.ArgumentParser()

        parser.add_argument("--input_dir", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)

        parser.add_argument("--logs_dir", type=str, default=None)
        parser.add_argument("--run_name", type=str, default=None)

        args = parser.parse_args()
        logs_dir = (Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "tokenizer_corpora")
        run_name = (args.run_name or f"tokenizer_corpora_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        texts = unify_corpus(input_dir= input_dir,
                            output_dir= output_dir,
                            logs_dir=logs_dir,
                            run_name=run_name)
        #Example:python src/tokenizer/unified_corpus.py --input_dir data/raw/COMETA_medieval_corpus --output_dir data/processed/tokenizer_corpora



if __name__ == "__main__":
    main()