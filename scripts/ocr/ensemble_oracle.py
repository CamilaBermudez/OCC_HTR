"""Oracle-headroom check for a per-line kraken/TrOCR ensemble (spec §6.13).

The hard-case analysis found the two leaders fail on largely disjoint lines. Before building a
real confidence-gated ensemble, this asks the cheap ceiling question: if a PERFECT per-line
router always picked the better model for each line, how much would the corpus metric improve
over the best single model? If the oracle headroom is tiny, the ensemble is not worth building.

Reads the two per_line_*.csv dumps (stem, cer, mean_conf, min_conf, gt, pred) and computes
corpus CER/WER for: each model alone, the per-line ORACLE (min edit distance), and two
*achievable* naive routers (pick the model with the higher mean- / min-confidence per line).
NB the kraken side here is the RAW 0.9710 CTC (no LM rescore); the deployed 0.9743 adds the
post-hoc char-LM on top, which an ensemble could also inherit.

    PROJECT_ROOT=. uv run python scripts/ocr/ensemble_oracle.py \
        --dir tests/ocr/evaluations/longtail_confidence
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def load(p):
    return {r["stem"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def corpus(rows):
    """rows = list of (pred, gt). Returns (char_acc, word_acc)."""
    cd = cn = wd = wn = 0
    for pred, gt in rows:
        nc, nw = max(1, len(gt)), max(1, len(gt.split()))
        # clip per line: over-production can't push a line past 100% wrong (CER/WER<=1, acc>=0)
        cd += min(Levenshtein.distance(pred, gt), nc)
        cn += nc
        wd += min(Levenshtein.distance(pred.split(), gt.split()), nw)
        wn += nw
    return 1 - cd / cn, 1 - wd / wn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence"))
    a = ap.parse_args()
    K = load(a.dir / "per_line_kraken.csv")
    T = load(a.dir / "per_line_trocr.csv")
    stems = [s for s in K if s in T]
    print(f"{len(stems)} shared lines\n")

    def dist(src, s):
        return Levenshtein.distance(src[s]["pred"], src[s]["gt"])

    # per-line source selection -> list of (pred, gt)
    def pick(chooser):
        out = []
        for s in stems:
            src = chooser(s)
            out.append((src[s]["pred"], src[s]["gt"]))
        return out

    kraken = pick(lambda s: K)
    trocr = pick(lambda s: T)
    oracle = pick(lambda s: K if dist(K, s) <= dist(T, s) else T)
    r_mean = pick(lambda s: K if float(K[s]["mean_conf"]) >= float(T[s]["mean_conf"]) else T)
    r_min = pick(lambda s: K if float(K[s]["min_conf"]) >= float(T[s]["min_conf"]) else T)

    rows = [
        ("kraken 0.9710 (CTC, raw)", kraken),
        ("TrOCR 0.9617", trocr),
        ("ORACLE (per-line best)", oracle),
        ("router: max mean-conf", r_mean),
        ("router: max min-conf", r_min),
    ]
    print(f"{'source':<26} | {'char acc':>8} | {'word acc':>8}")
    base_c = corpus(kraken)[0]
    for name, rws in rows:
        ca, wa = corpus(rws)
        tag = (
            f"  (Δchar {100 * (ca - base_c):+.2f}pp)"
            if name.startswith(("ORACLE", "router"))
            else ""
        )
        print(f"{name:<26} | {ca:>8.4f} | {wa:>8.4f}{tag}")

    # line-win breakdown
    kw = sum(1 for s in stems if dist(K, s) < dist(T, s))
    tw = sum(1 for s in stems if dist(T, s) < dist(K, s))
    tie = len(stems) - kw - tw
    print(f"\nline wins: kraken {kw}  |  TrOCR {tw}  |  tie {tie}")
    # how often the naive routers agree with the oracle choice
    agree_mean = sum(
        1
        for s in stems
        if (float(K[s]["mean_conf"]) >= float(T[s]["mean_conf"])) == (dist(K, s) <= dist(T, s))
    )
    print(
        f"max-mean-conf router agrees with oracle pick on {agree_mean}/{len(stems)} lines "
        f"({100 * agree_mean / len(stems):.0f}%)"
    )


if __name__ == "__main__":
    main()
