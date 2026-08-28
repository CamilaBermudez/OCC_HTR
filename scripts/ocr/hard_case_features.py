"""What makes a line hard — and for WHICH model? Feature analysis of the hard-case funnel.

The long-tail zoom (spec §6.13) found kraken and TrOCR fail on *largely disjoint* lines (only
11 of a 49-line union are hard for both). This characterises the four groups (hard-for-BOTH,
kraken-only, TrOCR-only, neither) by line features — image size, text length, ink-bleed score
(§6.5.8), minim + abbreviation density — to see what drives each model's errors and whether a
per-line router could exploit the difference.

Inputs: per-line tail flags from longtail_confidence (per_line_{kraken,trocr}.csv) + the
ink-bleed scores (ink_bleed_val300 JSON) + the validation crops/GT.

    PROJECT_ROOT=. uv run python scripts/ocr/hard_case_features.py \
        --longtail-dir tests/ocr/evaluations/longtail_confidence \
        --bleed tests/ocr/evaluations/ink_bleed_val300_20260718/ink_bleed_20260718_180817.json \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/longtail_confidence
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import unicodedata
from pathlib import Path

import matplotlib
from PIL import Image
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MINIM = [
    "in",
    "ni",
    "im",
    "mi",
    "iu",
    "ui",
    "un",
    "nu",
    "mn",
    "nm",
    "nn",
    "mm",
    "uu",
    "iii",
    "am",
    "ma",
]
_ROM = re.compile(r"\b[ivxlc]{2,}\b")
_ABBR = set("ñõãẽĩũ⁊")
FEATS = ["w", "h", "chars", "words", "bleed", "minim", "abbr"]
OURS, GOLD, GREY = "#9C2A24", "#8A6A26", "#8A8072"


def features(stem, gt, val_dir, bleed):
    w, h = Image.open(f"{val_dir}/{stem}.png").size
    low = gt.lower()
    mc = sum(low.count(s) for s in _MINIM) + len(_ROM.findall(low))
    ab = sum(1 for ch in unicodedata.normalize("NFD", gt) if unicodedata.combining(ch)) + sum(
        gt.count(c) for c in _ABBR
    )
    return {
        "w": w,
        "h": h,
        "chars": len(gt),
        "words": len(gt.split()),
        "bleed": bleed.get(stem + ".png", {}).get("bleed_score", float("nan")),
        "minim": mc,
        "abbr": ab,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--longtail-dir", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence")
    )
    ap.add_argument("--bleed", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence"))
    a = ap.parse_args()

    def load(p):
        return {r["stem"]: r for r in csv.DictReader(open(p))}

    K = load(a.longtail_dir / "per_line_kraken.csv")
    T = load(a.longtail_dir / "per_line_trocr.csv")
    bleed = json.loads(a.bleed.read_text())["images"]
    sh = [s for s in K if s in T]
    kh = {s for s in sh if K[s]["tail"] == "1"}
    th = {s for s in sh if T[s]["tail"] == "1"}
    groups = {
        "BOTH": kh & th,
        "kraken-only": kh - th,
        "trocr-only": th - kh,
        "neither": set(sh) - kh - th,
    }
    F = {s: features(s, K[s]["gt"], a.val_dir, bleed) for s in sh}

    def m(S, k):
        v = [F[s][k] for s in S if F[s][k] == F[s][k]]
        return st.mean(v) if v else float("nan")

    print(f"{'group':>12} | " + " | ".join(f"{k:>6}" for k in FEATS) + " |   n")
    for g, S in groups.items():
        print(f"{g:>12} | " + " | ".join(f"{m(S,k):>6.2f}" for k in FEATS) + f" | {len(S):>3}")

    print("\nSpearman(feature, per-line CER) — what makes lines hard for each model:")
    print(f"{'feature':>8} | {'kraken':>7} | {'trocr':>7}")
    for k in FEATS:
        idx = [s for s in sh if F[s][k] == F[s][k]]
        x = [F[s][k] for s in idx]
        rk = spearmanr(x, [float(K[s]["cer"]) for s in idx]).correlation
        rt = spearmanr(x, [float(T[s]["cer"]) for s in idx]).correlation
        print(f"{k:>8} | {rk:>7.2f} | {rt:>7.2f}")

    # dump per-line features + group
    def grp(s):
        return next(g for g, S in groups.items() if s in S)

    with open(a.out / "hard_case_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "group", "kraken_cer", "trocr_cer", *FEATS])
        for s in sh:
            w.writerow([s, grp(s), K[s]["cer"], T[s]["cer"], *[F[s][k] for k in FEATS]])

    # figure: bleed (physical) vs minim (content), coloured by group -> complementary axes
    col = {"BOTH": "#3B3B3B", "kraken-only": OURS, "trocr-only": GOLD, "neither": GREY}
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for g, S in groups.items():
        xs = [F[s]["bleed"] for s in S]
        ys = [F[s]["minim"] + F[s]["abbr"] for s in S]
        ax.scatter(
            xs,
            ys,
            s=(70 if g != "neither" else 18),
            c=col[g],
            alpha=(0.85 if g != "neither" else 0.35),
            label=f"{g} (n={len(S)})",
            edgecolors="white",
            linewidths=0.6,
        )
    ax.set(
        xlabel="ink-bleed score (physical degradation)",
        ylabel="minim + abbreviation count (content ambiguity)",
        title="Hard-case features — kraken fails on bleed, TrOCR on content",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out / "hard_case_features.png", dpi=140, bbox_inches="tight")
    print(f"\nsaved {a.out}/hard_case_features.png + hard_case_features.csv")


if __name__ == "__main__":
    main()
