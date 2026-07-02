import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.ocr.clean_medusa_output import run_clean_medusa_output


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description="Clean chat-template artefacts from Medusa .txt outputs.",
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder of Medusa .txt files (walked recursively).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="If set, write cleaned files here mirroring the input layout. "
        "If omitted, rewrite the input files in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching any files.",
    )
    parser.add_argument("--logs-dir", required=False)
    parser.add_argument(
        "--run-name",
        required=False,
        help="Log basename + summary run identifier. " "Default: clean_medusa_<timestamp>.",
    )
    parser.add_argument(
        "--no-config-log",
        action="store_true",
        help="Disable configuration logging inside the function.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    logs_dir = (
        Path(args.logs_dir) if args.logs_dir else project_root / "logs" / "clean_medusa_output"
    )
    run_name = args.run_name or f"clean_medusa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_clean_medusa_output(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        dry_run=args.dry_run,
        logs_dir=logs_dir,
        run_name=run_name,
        log_config=not args.no_config_log,
    )


if __name__ == "__main__":
    main()
