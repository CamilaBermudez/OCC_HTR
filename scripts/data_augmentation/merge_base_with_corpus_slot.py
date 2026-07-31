"""Merge a fresh base aug pool with the corpus-only slot of an existing pool.

Given a fresh base aug pool (real-derived renders, e.g. 3000 pairs from the
corrected 600 real ``.gt.txt`` files) and an existing hybrid pool that contains
the same shape of base + a corpus-derived slot (e.g. 3000 real + 1000 COMETA
or medical), emits a new hybrid pool = fresh base + the corpus-derived slot
copied verbatim.

A file is classified as **real-derived** iff its name starts with any of the
600 stems in ``--real-folder`` followed by ``.gt_``. Everything else in the
old pool is treated as **corpus-derived** and carried over as-is.

The corpus-derived slot doesn't depend on the 600 real annotations, so
carrying it over is safe when re-basing after annotation corrections.

Usage:
    python3 scripts/data_augmentation/merge_base_with_corpus_slot.py \\
        --base-aug ./data/.../aug_20260721_121550 \\
        --base-labels ./data/.../labels_20260721_121550/labels.json \\
        --old-pool ./data/.../aug_20260712_v2_medical \\
        --old-labels ./data/.../labels_20260712_v2_medical/labels.json \\
        --real-folder ./data/processed/annotated_samples/OCR/full_annotated \\
        --output-aug ./data/.../aug_20260721_v2_medical \\
        --output-labels ./data/.../labels_20260721_v2_medical/labels.json
"""

import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv


def load_real_stems(real_folder: Path) -> set[str]:
    stems = set()
    for gt in real_folder.glob("*.gt.txt"):
        stems.add(gt.stem.replace(".gt", ""))
    if not stems:
        raise SystemExit(f"No .gt.txt files found under {real_folder}")
    return stems


def is_real_derived(filename: str, real_stems: set[str]) -> bool:
    """Real-derived files match '<real_stem>.gt_...' pattern."""
    return any(filename.startswith(s + ".gt_") for s in real_stems)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-aug", required=True, help="Fresh base aug pool dir (real-derived only)"
    )
    parser.add_argument("--base-labels", required=True, help="Fresh base labels.json")
    parser.add_argument(
        "--old-pool", required=True, help="Existing hybrid pool dir (base + corpus slot)"
    )
    parser.add_argument("--old-labels", required=True, help="Existing hybrid pool labels.json")
    parser.add_argument(
        "--real-folder", required=True, help="Folder of 600 real .gt.txt (for classification)"
    )
    parser.add_argument("--output-aug", required=True, help="New hybrid pool dir to create")
    parser.add_argument(
        "--output-labels", required=True, help="New hybrid pool labels.json to write"
    )
    args = parser.parse_args()

    base_aug = Path(args.base_aug)
    base_labels_path = Path(args.base_labels)
    old_pool = Path(args.old_pool)
    old_labels_path = Path(args.old_labels)
    real_folder = Path(args.real_folder)
    output_aug = Path(args.output_aug)
    output_labels_path = Path(args.output_labels)

    for p, name in [
        (base_aug, "--base-aug"),
        (base_labels_path, "--base-labels"),
        (old_pool, "--old-pool"),
        (old_labels_path, "--old-labels"),
        (real_folder, "--real-folder"),
    ]:
        if not p.exists():
            raise SystemExit(f"{name} does not exist: {p}")

    # Load classification set
    real_stems = load_real_stems(real_folder)
    print(f"Loaded {len(real_stems)} real stems for classification")

    # Load labels
    base_labels: dict[str, str] = json.loads(base_labels_path.read_text(encoding="utf-8"))
    old_labels: dict[str, str] = json.loads(old_labels_path.read_text(encoding="utf-8"))
    print(f"Base labels: {len(base_labels)} entries")
    print(f"Old pool labels: {len(old_labels)} entries")

    # Extract corpus-derived subset from old pool (everything NOT real-derived)
    corpus_labels = {
        fn: txt for fn, txt in old_labels.items() if not is_real_derived(fn, real_stems)
    }
    old_real_derived_count = len(old_labels) - len(corpus_labels)
    print(
        f"Old pool decomposition: {old_real_derived_count} real-derived + {len(corpus_labels)} corpus-derived"
    )

    if not corpus_labels:
        raise SystemExit(
            f"Old pool {old_pool} has zero corpus-derived entries — nothing to merge in."
        )

    # Compose new labels: fresh base + old corpus slot
    new_labels = {**base_labels, **corpus_labels}
    if len(new_labels) != len(base_labels) + len(corpus_labels):
        overlap = set(base_labels) & set(corpus_labels)
        raise SystemExit(
            f"Base and corpus labels overlap on {len(overlap)} keys — indicates classification bug. "
            f"Example overlaps: {list(overlap)[:5]}"
        )

    # Create output aug dir + copy PNGs
    output_aug.mkdir(parents=True, exist_ok=True)

    # Base PNGs
    n_copied_base = 0
    for png_name in base_labels:
        src = base_aug / png_name
        dst = output_aug / png_name
        if not src.is_file():
            raise SystemExit(f"Base PNG missing: {src}")
        shutil.copy2(src, dst)
        n_copied_base += 1
    print(f"Copied {n_copied_base} base PNGs into {output_aug}")

    # Corpus PNGs from old pool
    n_copied_corpus = 0
    for png_name in corpus_labels:
        src = old_pool / png_name
        dst = output_aug / png_name
        if not src.is_file():
            raise SystemExit(f"Corpus PNG missing in old pool: {src}")
        shutil.copy2(src, dst)
        n_copied_corpus += 1
    print(f"Copied {n_copied_corpus} corpus PNGs from {old_pool}")

    # Write new labels.json
    output_labels_path.parent.mkdir(parents=True, exist_ok=True)
    output_labels_path.write_text(
        json.dumps(new_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(new_labels)} labels to {output_labels_path}")

    # Final sanity check
    on_disk = sorted(p.name for p in output_aug.glob("*.png"))
    label_keys = sorted(new_labels)
    if on_disk != label_keys:
        missing_on_disk = set(label_keys) - set(on_disk)
        missing_in_labels = set(on_disk) - set(label_keys)
        raise SystemExit(
            f"Sanity check failed. "
            f"On disk but no label: {len(missing_in_labels)}. "
            f"Labels but no PNG: {len(missing_on_disk)}."
        )
    print(f"Sanity check OK: {len(on_disk)} PNGs match {len(new_labels)} labels one-to-one")


if __name__ == "__main__":
    main()
