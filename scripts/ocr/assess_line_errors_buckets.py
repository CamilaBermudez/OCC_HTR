"""Root-cause bucketing of the diff spans from the 200-line error assessment.

Re-runs the same seeded sample as ``assess_line_errors.py`` but classifies every
emitted diff span into *why* it exists, separating what the model is actually
responsible for from editorial normalization and segmentation artifacts:

  editorial_punct   punctuation the editor adds/changes (, . ; : ¶) — not an error
  abbrev_expansion  brevigraph the editor expands (⁊, tildes, del=de lo) — we
                    predict the diplomatic form, so this is expected, not an error
  article_split     de la / de lo / de lu written joined by the scribe, spaced by
                    the editor — editorial word-segmentation (shows as a deletion)
  boundary_spill    add/del of material at the very start/end of the line — the
                    manuscript lineation != scholarly lineation, so edge words
                    spill in/out. An alignment artifact, not an OCR error.
  content_add/del   a whole word genuinely over-generated / dropped mid-line
  substitution      a genuine misread or variant (the real OCR-error signal)
  MISALIGNED_PAIR   whole pair with alignment score < 0.7 (bad line match)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.ocr.line_diff import _fold, diff_page

_ARTICLES = {"lo", "la", "los", "las", "lu", "l", "le", "els"}


def bucket_span(dtype: str, o: str, b: str, ocr_line: str, sch_line: str) -> str:
    if dtype == "punctuation":
        return "editorial_punct"
    if dtype == "abbreviation":
        return "abbrev_expansion"
    if dtype == "deletion":
        if _fold(b) in _ARTICLES:
            return "article_split"
        bs = b.strip()
        if bs and (sch_line.strip().startswith(bs) or sch_line.strip().endswith(bs)):
            return "boundary_spill"
        return "content_del"
    if dtype == "addition":
        os_ = o.strip()
        if os_ and (ocr_line.strip().startswith(os_) or ocr_line.strip().endswith(os_)):
            return "boundary_spill"
        if _fold(o) in _ARTICLES:
            return "article_split"
        return "content_add"
    return "substitution"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alignment-json", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--misalign-threshold", type=float, default=0.7)
    args = ap.parse_args()

    data = json.loads(args.alignment_json.read_text(encoding="utf-8"))
    aligned = [
        {"ocr": p["model_text"], "sch": p["scholarly_text"], "score": p.get("score", 0.0)}
        for page in data.values()
        for p in page["pairs"]
        if p.get("model_text") is not None and p.get("scholarly_text") is not None
    ]
    sample = random.Random(args.seed).sample(aligned, min(args.n, len(aligned)))

    buckets = Counter()
    n_misaligned = 0
    misaligned_spans = 0
    subs_examples: list[str] = []
    content_del_examples: list[str] = []
    for s in sample:
        ocr, sch, score = s["ocr"], s["sch"], s["score"]
        diffs = diff_page([sch], [(0, ocr)])
        if score < args.misalign_threshold:
            n_misaligned += 1
            misaligned_spans += len(diffs)
            buckets["MISALIGNED_PAIR(spans)"] += len(diffs)
            continue
        for d in diffs:
            bkt = bucket_span(d.type, d.ocr_text, d.base_text, ocr, sch)
            buckets[bkt] += 1
            if bkt == "substitution" and len(subs_examples) < 40:
                subs_examples.append(f"{d.ocr_text!r}->{d.base_text!r}")
            if bkt == "content_del" and len(content_del_examples) < 25:
                content_del_examples.append(f"{d.base_text!r}")

    total = sum(buckets.values())
    print(
        f"sampled {len(sample)} pairs; {n_misaligned} misaligned (score<{args.misalign_threshold}) "
        f"contributing {misaligned_spans} garbage spans\n"
    )
    print(f"{'bucket':<24} {'spans':>6} {'share':>7}   root cause")
    order = [
        ("editorial_punct", "EDITORIAL — editor punctuation"),
        ("abbrev_expansion", "EDITORIAL — brevigraph expansion (we predict diplomatic)"),
        ("article_split", "EDITORIAL — de+lo/la spacing"),
        ("boundary_spill", "ARTIFACT  — lineation mismatch, edge word-spill"),
        ("MISALIGNED_PAIR(spans)", "ARTIFACT  — whole pair mis-aligned (score<0.7)"),
        ("content_del", "OCR ERROR — dropped word"),
        ("content_add", "OCR ERROR — over-generated word"),
        ("substitution", "OCR ERROR — misread / variant"),
    ]
    for k, desc in order:
        v = buckets.get(k, 0)
        print(f"{k:<24} {v:>6} {100*v/total:>6.1f}%   {desc}")
    print(f"{'TOTAL':<24} {total:>6}")

    editorial = sum(
        buckets.get(k, 0) for k in ("editorial_punct", "abbrev_expansion", "article_split")
    )
    artifact = buckets.get("boundary_spill", 0) + buckets.get("MISALIGNED_PAIR(spans)", 0)
    ocr_err = sum(buckets.get(k, 0) for k in ("content_del", "content_add", "substitution"))
    print(
        f"\nROLL-UP:  editorial {editorial} ({100*editorial/total:.0f}%)  |  "
        f"artifact {artifact} ({100*artifact/total:.0f}%)  |  "
        f"genuine OCR error {ocr_err} ({100*ocr_err/total:.0f}%)"
    )
    print("\nsample of genuine substitutions:\n  " + "\n  ".join(subs_examples[:30]))


if __name__ == "__main__":
    main()
