"""What makes a line hard — and for WHICH model? Feature analysis of the hard-case funnel.

The long-tail zoom (spec §6.13) found kraken and TrOCR fail on *largely disjoint* lines (only
11 of a 49-line union are hard for both). This characterises the four groups (hard-for-BOTH,
kraken-only, TrOCR-only, neither) by line features — image size, text length, ink-bleed score
(§6.5.8), minim run-length + special-character density — to see what drives each model's errors, and whether a
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
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# maximal runs of adjacent minim letters (i, m, n, u); {2,} = 2+ in a row. Adjacency is what
# creates the ambiguity — a lone minim letter is readable, "iiii"/"mmuni" is not.
_RUN = re.compile(r"[imnu]{2,}")
FEATS = ["w", "h", "chars", "words", "bleed", "runlen", "special"]
OURS, GOLD, GREY = "#9C2A24", "#8A6A26", "#8A8072"


def features(stem, gt, val_dir, bleed):
    w, h = Image.open(f"{val_dir}/{stem}.png").size
    low = gt.lower()
    # minim ambiguity = total chars inside maximal runs of >=2 adjacent minim letters. Isolated
    # minim letters aren't confusable, so only runs count ('sic'->0, 'dixit'->0, 'minim'->5).
    mc = sum(len(r) for r in _RUN.findall(low))
    # special-glyph count: diacritic letters (ñõãẽĩũ, incl. loose combining marks) + medieval
    # abbreviation symbols (⁊ Tironian et, ꝑ p-with-stroke). NFC composes precomposed forms so each
    # glyph counts ONCE; the ¶ paragraph marker is excluded (structural, not a hard letterform).
    # We count special CHARACTERS, not "abbreviations" (which need a semantic call we can't make).
    sp = sum(1 for ch in unicodedata.normalize("NFC", gt) if ord(ch) > 127 and ch != "¶")
    return {
        "w": w,
        "h": h,
        "chars": len(gt),
        "words": len(gt.split()),
        "bleed": bleed.get(stem + ".png", {}).get("bleed_score", float("nan")),
        "runlen": mc,
        "special": sp,
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

    # length-normalised densities: raw counts don't control for line length, so also report
    # count / line-length (per 100 chars), averaged per line so long lines don't inflate.
    print("\nlength-normalised (per 100 chars):")
    print(f"{'group':>12} | {'avg chars':>9} | {'runlen/100ch':>12} | {'special/100ch':>13}")
    for g, S in groups.items():
        ch = st.mean([F[s]["chars"] for s in S])
        rd = 100 * st.mean([F[s]["runlen"] / max(1, F[s]["chars"]) for s in S])
        sd = 100 * st.mean([F[s]["special"] / max(1, F[s]["chars"]) for s in S])
        print(f"{g:>12} | {ch:>9.1f} | {rd:>12.2f} | {sd:>13.2f}")

    print("\nSpearman(feature, per-line CER) — what makes lines hard for each model:")
    print(f"{'feature':>8} | {'kraken':>7} | {'trocr':>7}")
    for k in FEATS:
        idx = [s for s in sh if F[s][k] == F[s][k]]
        x = [F[s][k] for s in idx]
        rk = spearmanr(x, [float(K[s]["cer"]) for s in idx]).correlation
        rt = spearmanr(x, [float(T[s]["cer"]) for s in idx]).correlation
        print(f"{k:>8} | {rk:>7.2f} | {rt:>7.2f}")

    # ink-bleed: where do the hard groups sit in the OVERALL bleed distribution? (means hide this)
    allb = np.array([F[s]["bleed"] for s in sh if F[s]["bleed"] == F[s]["bleed"]])
    p75, p90, p99 = (float(np.percentile(allb, q)) for q in (75, 90, 99))
    print(
        f"\nink-bleed percentiles (overall n={len(allb)}): p75 {p75:.3f}  p90 {p90:.3f}  p99 {p99:.3f}"
    )
    print(
        f"{'group':>12} | {'median':>6} | {'%>p75':>5} | {'%>p90':>5} | {'%>p99':>5} | {'mean %ile-rank':>13}"
    )
    ranks = {
        s: 100 * r / (len(allb) - 1) for r, s in enumerate(sorted(sh, key=lambda s: F[s]["bleed"]))
    }
    for g, S in groups.items():
        v = np.array([F[s]["bleed"] for s in S if F[s]["bleed"] == F[s]["bleed"]])
        mr = np.mean([ranks[s] for s in S])
        print(
            f"{g:>12} | {np.median(v):>6.3f} | {100 * np.mean(v > p75):>4.0f}% | "
            f"{100 * np.mean(v > p90):>4.0f}% | {100 * np.mean(v > p99):>4.0f}% | {mr:>12.0f}%"
        )

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
        ys = [F[s]["runlen"] + F[s]["special"] for s in S]
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
        ylabel="minim run-length + special-char count (content ambiguity)",
        title="Hard-case features — kraken fails on bleed, TrOCR on content",
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out / "hard_case_features.png", dpi=140, bbox_inches="tight")
    print(f"\nsaved {a.out}/hard_case_features.png + hard_case_features.csv")


if __name__ == "__main__":
    main()
