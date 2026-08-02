"""Serious error assessment of OCR-vs-scholarly differences on a random line sample.

Samples N aligned (model_text, scholarly_text) pairs from a ``line_alignment.json``
(produced by ``align_transcriptions.py``), then for each pair:

  * computes the raw character CER (Levenshtein / len(scholarly)),
  * computes a *folded* CER (after u/v, i/j, long-s folding + mark/space strip) —
    the residual once purely-orthographic noise is removed,
  * runs the **shipped** page-level diff classifier (``diff_page``) on the single
    pair and records the category of every difference it emits.

Prints aggregate statistics (category totals, CER distribution, share of clean
lines, alignment-quality bins) and writes a full TSV of all N sampled pairs
(``ocr`` | ``scholarly`` | ``score`` | ``cer`` | ``folded_cer`` | ``diffs``) for
manual audit.

Reusable across models: point ``--alignment-json`` at any model's alignment file.

    PROJECT_ROOT=. uv run python scripts/ocr/assess_line_errors.py \
        --alignment-json data/processed/transcription/<model>/line_alignment.json \
        --n 200 --seed 42 --out <report.tsv>
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from src.ocr.line_diff import _fold, diff_page

_FOLD = str.maketrans({"v": "u", "j": "i", "ſ": "s", "ꝛ": "r"})


def raw_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def folded_cer(ref: str, hyp: str) -> float:
    """CER after folding both sides to comparable ASCII letters (drops orthographic noise)."""
    fr, fh = _fold(ref), _fold(hyp)
    if not fr:
        return 0.0 if not fh else 1.0
    return Levenshtein.distance(fr, fh) / len(fr)


def has_abbrev_mark(text: str) -> bool:
    return any(ord(c) > 127 and unicodedata.category(c)[0] in ("L", "M") for c in text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alignment-json", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(args.alignment_json.read_text(encoding="utf-8"))

    aligned: list[dict] = []  # both texts present
    ocr_only = 0  # OCR line with no scholarly (addition / over-segmentation)
    sch_only = 0  # scholarly line with no OCR (deletion / merged block)
    for page_key, page in data.items():
        for p in page["pairs"]:
            mt, st = p.get("model_text"), p.get("scholarly_text")
            if mt is not None and st is not None:
                aligned.append(
                    {"page": page_key, "ocr": mt, "sch": st, "score": p.get("score", 0.0)}
                )
            elif mt is not None:
                ocr_only += 1
            elif st is not None:
                sch_only += 1

    rng = random.Random(args.seed)
    sample = rng.sample(aligned, min(args.n, len(aligned)))

    cat_totals = Counter()
    lines_with_cat = Counter()  # how many lines contain >=1 of a category
    n_clean = 0  # 0 diffs emitted (identical modulo suppressed orthographic/spacing)
    n_exact = 0  # raw strings identical
    n_fold_exact = 0  # folded strings identical (pure orthographic apart)
    raw_cers, fold_cers = [], []
    score_bins = Counter()
    rows: list[dict] = []

    for s in sample:
        ocr, sch, score = s["ocr"], s["sch"], s["score"]
        rc, fc = raw_cer(sch, ocr), folded_cer(sch, ocr)
        raw_cers.append(rc)
        fold_cers.append(fc)
        diffs = diff_page([sch], [(0, ocr)])
        cats = [d.type for d in diffs]
        cat_totals.update(cats)
        for c in set(cats):
            lines_with_cat[c] += 1
        if not diffs:
            n_clean += 1
        if ocr == sch:
            n_exact += 1
        if _fold(ocr) == _fold(sch):
            n_fold_exact += 1
        score_bins[
            "1.0"
            if score >= 0.999
            else "0.8-1.0"
            if score >= 0.8
            else "0.5-0.8"
            if score >= 0.5
            else "<0.5"
        ] += 1
        rows.append(
            {
                "page": s["page"],
                "score": f"{score:.2f}",
                "cer": f"{rc:.3f}",
                "folded_cer": f"{fc:.3f}",
                "ocr": ocr,
                "sch": sch,
                "diffs": " | ".join(f"{d.type}:{d.ocr_text!r}->{d.base_text!r}" for d in diffs),
            }
        )

    n = len(sample)

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def med(xs: list[float]) -> float:
        return sorted(xs)[len(xs) // 2] if xs else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("page\tscore\tcer\tfolded_cer\tocr\tscholarly\tdiffs\n")
        for r in rows:
            f.write(
                f"{r['page']}\t{r['score']}\t{r['cer']}\t{r['folded_cer']}\t"
                f"{r['ocr']}\t{r['sch']}\t{r['diffs']}\n"
            )

    print(f"=== corpus: {args.alignment_json} ===")
    print(
        f"aligned pairs (both present): {len(aligned)}  |  OCR-only: {ocr_only}  |  scholarly-only: {sch_only}"
    )
    print(f"sampled: {n} (seed {args.seed})\n")
    print("--- per-line divergence magnitude (OCR vs scholarly) ---")
    print(f"raw CER    mean {mean(raw_cers):.3f}  median {med(raw_cers):.3f}")
    print(
        f"folded CER mean {mean(fold_cers):.3f}  median {med(fold_cers):.3f}   (after u/v i/j long-s fold)"
    )
    print(f"exact match (raw):    {n_exact}/{n} ({100*n_exact/n:.0f}%)")
    print(f"exact match (folded): {n_fold_exact}/{n} ({100*n_fold_exact/n:.0f}%)")
    print(f"clean lines (no diff emitted): {n_clean}/{n} ({100*n_clean/n:.0f}%)\n")
    print("--- classified differences (shipped diff_page, orthographic+spacing suppressed) ---")
    print(f"{'category':<14} {'total spans':>11} {'lines w/ >=1':>12}")
    for c, t in cat_totals.most_common():
        print(f"{c:<14} {t:>11} {lines_with_cat[c]:>12}")
    print("\n--- alignment-score bins (sample) ---")
    for b in ("1.0", "0.8-1.0", "0.5-0.8", "<0.5"):
        print(f"  {b:<8} {score_bins[b]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
