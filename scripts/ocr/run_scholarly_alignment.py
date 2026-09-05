"""Align the scholarly edition to the manuscript's line structure (CLI).

Wraps :mod:`src.ocr.scholarly_alignment` (the two-pass anchored-DP aligner that
was developed in ``notebooks/ocr/text_alignment.ipynb``): the scholarly edition
is a continuous text with the editor's lineation, and an auxiliary per-page OCR
transcription (e.g. the catmus full-corpus run) supplies the manuscript's line
breaks. Output = the reference text, verbatim and lossless, re-broken at the
manuscript's line boundaries — the ``AlbucE_aligned_<stamp>.txt`` every
downstream align/diff/compare step consumes.

The lossless invariant (concatenating all aligned lines reproduces the
reference word-for-word) is verified after the run; a violation exits non-zero.

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/run_scholarly_alignment.py \
        --reference-txt data/raw/AlbucE.txt \
        --ocr-dir data/processed/transcription/ocr_kept_20260622_120413 \
        --output-root tests/ocr
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.ocr.scholarly_alignment import (
    align_ocr_to_reference,
    enforce_page_boundaries,
    save_aligned,
    save_side_by_side,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("scholarly_alignment")


def verify_lossless(ref_text: str, aligned_doc) -> bool:
    """Concatenated aligned lines must equal the reference word list exactly."""
    ref_words = ref_text.split()
    aligned_words = [w for _, lines in aligned_doc for line in lines for w in line.split()]
    if ref_words == aligned_words:
        return True
    for i, (a, b) in enumerate(zip(ref_words, aligned_words)):
        if a != b:
            log.error("first mismatch at word %d: ref=%r aligned=%r", i, a, b)
            break
    log.error(
        "lossless check FAILED: %d ref words vs %d aligned", len(ref_words), len(aligned_words)
    )
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reference-txt", type=Path, required=True, help="scholarly edition (plain text)"
    )
    ap.add_argument(
        "--ocr-dir",
        type=Path,
        required=True,
        help="auxiliary OCR transcription: flat dir of per-page *.txt files",
    )
    ap.add_argument("--output-root", type=Path, default=Path("tests/ocr"))
    ap.add_argument("--prefix", default="AlbucE", help="output filename prefix")
    ap.add_argument(
        "--boundary-mode",
        choices=("trim", "warn"),
        default="trim",
        help="trim = force each ref word onto exactly one page; warn = report overlaps only",
    )
    ap.add_argument(
        "--no-side-by-side", action="store_true", help="skip the two-column review file"
    )
    args = ap.parse_args()

    ref_text = args.reference_txt.read_text(encoding="utf-8")
    aligned_doc, page_info = align_ocr_to_reference(ref_text, args.ocr_dir)
    aligned_doc = enforce_page_boundaries(
        aligned_doc, page_info, ref_text, mode=args.boundary_mode, verbose=True
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = args.output_root / f"{args.prefix}_aligned_{stamp}.txt"
    save_aligned(aligned_doc, out)
    if not args.no_side_by_side:
        save_side_by_side(
            args.ocr_dir,
            aligned_doc,
            args.output_root / f"{args.prefix}_aligned_sidebyside_{stamp}.txt",
        )

    ok = verify_lossless(ref_text, aligned_doc)
    n_lines = sum(len(lines) for _, lines in aligned_doc)
    log.info("aligned %d pages / %d lines -> %s | lossless=%s", len(aligned_doc), n_lines, out, ok)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
