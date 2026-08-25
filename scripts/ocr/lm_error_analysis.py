"""Char-level error breakdown + substitution confusion tables for two outputs.

Purpose: quantify what the char-LM reranker changes vs the raw CTC 1-best
(spec §6.8 / §6.13). For each model we char-align every line to the GT
(Levenshtein editops, src=GT dest=pred) and tally substitution / insertion /
deletion counts and the substitution confusion pairs (GT char → predicted char).

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/lm_error_analysis.py \
        --gt-dir data/processed/annotated_samples/OCR/validation \
        --pred "kraken_0.9710=data/processed/transcription/kraken_600real_8020_val300" \
        --pred "kraken_0.9743=data/processed/transcription/krakenLM_val300" --top 12
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def _load(folder: Path, suffix: str) -> dict[str, str]:
    return {
        f.name[: -len(suffix)]: f.read_text(encoding="utf-8").strip()
        for f in sorted(folder.glob(f"*{suffix}"))
    }


def analyse(gt: dict[str, str], pred: dict[str, str]):
    sub = ins = dele = ref_chars = 0
    w_edits = w_ref = 0
    subs: Counter[tuple[str, str]] = Counter()
    for stem, g in gt.items():
        p = pred.get(stem, "")
        ref_chars += len(g)
        for tag, i, j in Levenshtein.editops(g, p):  # transform GT -> pred
            if tag == "replace":
                sub += 1
                subs[(g[i], p[j])] += 1
            elif tag == "insert":
                ins += 1  # a char the model added (not in GT)
            else:
                dele += 1  # a GT char the model dropped
        w_edits += Levenshtein.distance(g.split(), p.split())
        w_ref += len(g.split())
    cer = (sub + ins + dele) / ref_chars
    return {
        "cer": cer,
        "wer": w_edits / w_ref,
        "sub": sub,
        "ins": ins,
        "del": dele,
        "ref_chars": ref_chars,
        "subs": subs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--pred", action="append", required=True, help="name=dir (repeatable)")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    gt = _load(args.gt_dir, ".gt.txt")
    results = {}
    for spec in args.pred:
        name, d = spec.split("=", 1)
        results[name] = analyse(gt, _load(Path(d), ".txt"))

    names = list(results)
    print(f"GT lines={len(gt)}  ref_chars={next(iter(results.values()))['ref_chars']}\n")
    print(f"{'model':>16}{'char-acc':>10}{'CER':>8}{'WER':>8}{'subs':>7}{'ins':>6}{'del':>6}")
    for n in names:
        r = results[n]
        print(
            f"{n:>16}{1 - r['cer']:>10.4f}{r['cer']:>8.4f}{r['wer']:>8.4f}"
            f"{r['sub']:>7}{r['ins']:>6}{r['del']:>6}"
        )

    for n in names:
        r = results[n]
        print(f"\n=== top {args.top} substitution confusions — {n}  (GT→pred, count) ===")
        for (a, b), c in r["subs"].most_common(args.top):
            print(f"  {a!r:>6} → {b!r:<6} {c}")

    if len(names) == 2:
        a, b = names
        ca, cb = results[a]["subs"], results[b]["subs"]
        keys = set(ca) | set(cb)
        print(f"\n=== biggest confusion changes: {b} minus {a} (LM impact) ===")
        deltas = sorted(((cb[k] - ca[k], k) for k in keys), key=lambda x: x[0])
        print("  most REDUCED by the LM:")
        for d, (x, y) in deltas[:8]:
            if d < 0:
                print(f"    {x!r}→{y!r}: {ca[(x, y)]} → {cb[(x, y)]}  ({d:+d})")
        print("  most INTRODUCED by the LM:")
        for d, (x, y) in reversed(deltas[-8:]):
            if d > 0:
                print(f"    {x!r}→{y!r}: {ca[(x, y)]} → {cb[(x, y)]}  ({d:+d})")


if __name__ == "__main__":
    main()
