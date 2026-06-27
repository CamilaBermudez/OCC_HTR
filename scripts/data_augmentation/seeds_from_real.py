"""Turn a folder of <stem>.gt.txt verified-label files into a categorized-
seed JSON that ``medieval_text_generation`` can read.

The verified .gt.txt files already contain the verbatim manuscript text
(long-s, rotunda-r, tironian-et, etc.). Treating them as the seed
corpus for the synthetic pipeline means the synthetic images carry the
exact texts the model needs to learn — no more domain drift introduced
by the COMETA / medical_texts corpora that don't match the target
manuscript's vocabulary.

Output JSON shape matches ``corpus_categorization`` (one row per
sample, ``samples[sample_id] = {"categories": [...], "text": ...}``)
so the downstream make targets can use it without changes.

Usage:
    PROJECT_ROOT=. uv run python scripts/data_augmentation/seeds_from_real.py \\
        --real-folder ./tests/ocr/real_corrected_20260625 \\
        --output-dir  ./data/processed/synthetic_seeds
"""

import argparse
import datetime
import json
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Build a categorized-seed JSON from the verified .gt.txt files in a "
            "real-corrected folder, so the synthetic pipeline can render images "
            "of the exact verbatim texts the model is trained to predict."
        )
    )
    parser.add_argument(
        "--real-folder",
        required=True,
        help="Folder of <stem>.gt.txt files (e.g. tests/ocr/real_corrected_20260625).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output root. JSON lands at output-dir/<run_name>/seeds_from_real.json. "
        "Default: data/processed/synthetic_seeds",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Run subdirectory name. Default: from_real_<timestamp>.",
    )
    parser.add_argument(
        "--category",
        default="real_seed",
        help="Single category label applied to every sample. Default: real_seed.",
    )
    args = parser.parse_args()

    real_folder = Path(args.real_folder)
    assert real_folder.is_dir(), f"Real folder not found: {real_folder}"

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "data/processed/synthetic_seeds"
    )
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"from_real_{stamp}"
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "seeds_from_real.json"

    # Collect (stem, text) pairs. Skip empties and warn.
    samples: dict[str, dict] = {}
    skipped_empty: list[str] = []
    for gt in sorted(real_folder.glob("*.gt.txt")):
        text = gt.read_text(encoding="utf-8").strip()
        if not text:
            skipped_empty.append(gt.stem)
            continue
        sample_id = f"{gt.name}:1"
        samples[sample_id] = {"categories": [args.category], "text": text}

    assert samples, f"No non-empty .gt.txt files in {real_folder}"

    doc = {
        "summary": {
            "source_folder": str(real_folder),
            "run": run_name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_samples": len(samples),
            "n_skipped_empty": len(skipped_empty),
            "categories": [args.category],
            "lines_per_category": {args.category: len(samples)},
        },
        "samples": samples,
    }
    output_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"wrote {output_path}")
    print(f"  {len(samples)} samples  |  {len(skipped_empty)} skipped (empty)")
    if skipped_empty:
        print(f"  skipped stems: {skipped_empty[:5]}{'...' if len(skipped_empty) > 5 else ''}")


if __name__ == "__main__":
    main()
