"""Merge ink-bleed detection JSON into the validation manifest.

Reads the JSON produced by ``src/data_preprocessing/ink_bleed_detection.py``
(fields ``images.<rel_key>.bleed_score`` and ``images.<rel_key>.has_bleed``)
and joins it to an existing manifest CSV on ``stem``. Emits a new manifest
with extra columns:

  - ``bleed_score`` : float in [0, 1], the composite score from the
    project's Otsu-based metric (min-max normalised within the run).
  - ``has_bleed_pNN``: True iff the row's ``bleed_score`` is at or above the
    NN-th percentile of the score distribution. One column per value passed
    via ``--percentiles`` (default: 75 90 95 99).

The single ``has_bleed`` flag written by ``detect_ink_bleed`` (derived from
the single ``--bleed-percentile`` used at detection time) is preserved as
``has_bleed_run`` so nothing is lost.

Usage:
    python3 scripts/ocr/merge_ink_bleed_to_manifest.py \\
        --bleed-json tests/ocr/evaluations/ink_bleed_val300_20260718/ink_bleed_20260718_180817.json \\
        --input-manifest tests/ocr/validation_300_manifest_.csv \\
        --output-manifest tests/ocr/validation_300_manifest__with_bleed.csv \\
        --percentiles 75 90 95 99
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bleed-json",
        required=True,
        help="Path to ink_bleed_<timestamp>.json produced by run_ink_bleed_detection.py.",
    )
    parser.add_argument(
        "--input-manifest",
        required=False,
        help="Existing manifest CSV to enrich. Default: tests/ocr/validation_300_manifest_.csv",
    )
    parser.add_argument(
        "--output-manifest",
        required=False,
        help="Output CSV path. Default: input path with '_with_bleed' appended before '.csv'.",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=[75, 90, 95, 99],
        help="One or more percentile thresholds to encode as has_bleed_pNN columns.",
    )
    args = parser.parse_args()

    bleed_json = (
        Path(args.bleed_json)
        if Path(args.bleed_json).is_absolute()
        else project_root / args.bleed_json
    )
    input_manifest = (
        Path(args.input_manifest)
        if args.input_manifest
        else project_root / "tests/ocr/validation_300_manifest_.csv"
    )
    if args.output_manifest:
        output_manifest = Path(args.output_manifest)
    else:
        output_manifest = input_manifest.with_name(
            input_manifest.stem + "_with_bleed" + input_manifest.suffix
        )

    print(f"bleed JSON:       {bleed_json}")
    print(f"input manifest:   {input_manifest}")
    print(f"output manifest:  {output_manifest}")
    print(f"percentiles:      {args.percentiles}\n")

    data = json.loads(bleed_json.read_text(encoding="utf-8"))
    images = data["images"]

    # rel_key looks like "05_garde_001_line_47.png"; strip extension for stem match.
    score_by_stem: dict[str, float] = {}
    has_bleed_run_by_stem: dict[str, bool] = {}
    for rel_key, entry in images.items():
        stem = Path(rel_key).stem
        score_by_stem[stem] = float(entry["bleed_score"])
        has_bleed_run_by_stem[stem] = bool(entry.get("has_bleed", False))

    scores_arr = np.array(list(score_by_stem.values()))
    thresholds: dict[int, float] = {
        p: float(np.percentile(scores_arr, p)) for p in args.percentiles
    }
    print("Bleed-score distribution:")
    for p in [5, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:>2} = {np.percentile(scores_arr, p):.4f}")
    print(
        f"  mean = {scores_arr.mean():.4f}, min = {scores_arr.min():.4f}, max = {scores_arr.max():.4f}\n"
    )
    print("Configured percentile cutoffs (strict >=, matches source semantics):")
    for p, t in thresholds.items():
        n_above = int((scores_arr >= t).sum())
        print(f"  has_bleed_p{p}: threshold={t:.4f}, {n_above} of {len(scores_arr)} lines flagged")
    print()

    with input_manifest.open() as f:
        reader = csv.DictReader(f)
        base_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if "stem" not in base_fields:
        raise SystemExit("Input manifest must have a 'stem' column")

    extra_fields = ["bleed_score", "has_bleed_run"] + [f"has_bleed_p{p}" for p in args.percentiles]
    out_fields = base_fields + extra_fields

    missing_in_json = 0
    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            new_row = dict(row)
            stem = row["stem"]
            if stem in score_by_stem:
                score = score_by_stem[stem]
                new_row["bleed_score"] = f"{score:.6f}"
                new_row["has_bleed_run"] = "True" if has_bleed_run_by_stem[stem] else "False"
                for p, t in thresholds.items():
                    new_row[f"has_bleed_p{p}"] = "True" if score >= t else "False"
            else:
                missing_in_json += 1
                new_row["bleed_score"] = ""
                new_row["has_bleed_run"] = ""
                for p in args.percentiles:
                    new_row[f"has_bleed_p{p}"] = ""
            writer.writerow(new_row)

    if missing_in_json:
        print(f"WARN: {missing_in_json} manifest rows had no matching image in the JSON")
    print(f"Wrote enriched manifest: {output_manifest}")


if __name__ == "__main__":
    main()
