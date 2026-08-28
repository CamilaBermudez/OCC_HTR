"""Reproduce the AlbucE 6-model results — the char-vs-word CI scatter + the corpus /
per-line-median tables (see ``docs/model_results.{md,html}``).

The figures embedded here are the final 300-val results: corpus + median CER/WER,
char/word accuracy, and 95% paired-bootstrap CIs (``scripts/ocr/bootstrap_ocr_ci.py``,
10 000×, seed 42). Catmus and Medusa are the full-page eval (their fair protocol,
``tests/ocr/evaluations/seven_way_vs_validation_300/``); the fine-tuned kraken and
TrOCR models are the native line-crop eval.

    PROJECT_ROOT=. uv run python scripts/ocr/plot_model_results.py --out-dir docs/figures
    # writes model_results_scatter.{pdf,png}, corpus.csv, median.csv and prints the tables
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# name, group, arch, CER, char[lo,hi], WER, word[lo,hi], median CER/char/WER/word, size,
# and a label offset (dx, dy in axis %, horizontal alignment) so the 6 labels don't collide.
MODELS = [
    {
        "name": "kraken · CTC + LM",
        "group": "ours",
        "arch": "CTC + char-LM",
        "cer": 0.0256,
        "char": 0.9744,
        "cLo": 0.9707,
        "cHi": 0.9780,
        "wer": 0.1627,
        "word": 0.8373,
        "wLo": 0.8143,
        "wHi": 0.8597,
        "mcer": 0.0238,
        "mchar": 0.9762,
        "mwer": 0.1250,
        "mword": 0.8750,
        "size": "4.08 M / 16 MB",
        "off": (0.12, 0.9, "left"),
    },
    {
        "name": "kraken · CTC",
        "group": "ours",
        "arch": "CTC (VGSL CRNN)",
        "cer": 0.0290,
        "char": 0.9710,
        "cLo": 0.9670,
        "cHi": 0.9749,
        "wer": 0.1798,
        "word": 0.8202,
        "wLo": 0.7962,
        "wHi": 0.8441,
        "mcer": 0.0250,
        "mchar": 0.9750,
        "mwer": 0.1250,
        "mword": 0.8750,
        "size": "4.08 M / 16 MB",
        "off": (0.12, -1.4, "left"),
    },
    {
        "name": "TrOCR · light-aug",
        "group": "ours",
        "arch": "ViT + RoBERTa",
        "cer": 0.0383,
        "char": 0.9617,
        "cLo": 0.9573,
        "cHi": 0.9660,
        "wer": 0.2174,
        "word": 0.7826,
        "wLo": 0.7590,
        "wHi": 0.8059,
        "mcer": 0.0278,
        "mchar": 0.9722,
        "mwer": 0.1818,
        "mword": 0.8182,
        "size": "282.6 M / 1.1 GB",
        "off": (0.12, 0.6, "left"),
    },
    {
        "name": "Catmus",
        "group": "base",
        "arch": "Kraken CTC, frozen",
        "cer": 0.0387,
        "char": 0.9613,
        "cLo": 0.9562,
        "cHi": 0.9663,
        "wer": 0.1434,
        "word": 0.8566,
        "wLo": 0.8389,
        "wHi": 0.8738,
        "mcer": 0.0278,
        "mchar": 0.9722,
        "mwer": 0.1250,
        "mword": 0.8750,
        "size": "4.08 M / 16 MB",
        "off": (-0.12, 0.9, "right"),
    },
    {
        "name": "TrOCR · med4k",
        "group": "ours",
        "arch": "ViT + RoBERTa",
        "cer": 0.0452,
        "char": 0.9548,
        "cLo": 0.9496,
        "cHi": 0.9596,
        "wer": 0.2293,
        "word": 0.7707,
        "wLo": 0.7462,
        "wHi": 0.7938,
        "mcer": 0.0286,
        "mchar": 0.9714,
        "mwer": 0.1667,
        "mword": 0.8333,
        "size": "282.6 M / 1.1 GB",
        "off": (0.12, 0.6, "left"),
    },
    {
        "name": "Medusa",
        "group": "base",
        "arch": "Qwen-VL, 9 B",
        "cer": 0.0490,
        "char": 0.9510,
        "cLo": 0.9459,
        "cHi": 0.9558,
        "wer": 0.3106,
        "word": 0.6894,
        "wLo": 0.6593,
        "wHi": 0.7191,
        "mcer": 0.0435,
        "mchar": 0.9565,
        "mwer": 0.2857,
        "mword": 0.7143,
        "size": "9 B / ~18 GB",
        "off": (0.12, 0.6, "left"),
    },
]

OURS = "#9C2A24"  # madder rubric
BASE = "#8A6A26"  # illuminated ochre
INK = "#2A251F"
GRID = "#D9D1C2"


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def plot(out_dir: Path, formats: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "Georgia", "DejaVu Serif"],
            "font.size": 11,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for d in MODELS:
        c = OURS if d["group"] == "ours" else BASE
        x, y = d["char"] * 100, d["word"] * 100
        xerr = [[(d["char"] - d["cLo"]) * 100], [(d["cHi"] - d["char"]) * 100]]
        yerr = [[(d["word"] - d["wLo"]) * 100], [(d["wHi"] - d["word"]) * 100]]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=c,
            elinewidth=1.3,
            alpha=0.45,
            capsize=2.5,
            zorder=2,
        )
        ax.plot(x, y, "o", ms=7.5, color=c, mec="white", mew=1.2, zorder=3)
        dx, dy, ha = d["off"]
        ax.annotate(
            d["name"],
            (x + dx, y + dy),
            ha=ha,
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            zorder=4,
        )
        ax.annotate(
            f"{_pct(d['char'])} · {_pct(d['word'])}",
            (x + dx, y + dy - 0.9),
            ha=ha,
            va="top",
            fontsize=8.5,
            color="#8A8072",
            family="DejaVu Sans",
            zorder=4,
        )

    ax.set_xlim(94.5, 97.7)
    ax.set_ylim(66.5, 87.5)
    ax.set_xlabel("Character accuracy (%)")
    ax.set_ylabel("Word accuracy (%)")
    ax.set_title(
        "AlbucE OCR — accuracy trade-off (300-line validation, 95% CI)", fontsize=12.5, pad=12
    )
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=OURS, mec="white", ms=8, label="This thesis"),
            Line2D([], [], marker="o", ls="", color=BASE, mec="white", ms=8, label="Baseline"),
            Line2D([], [], color="#8A8072", lw=1.3, alpha=0.6, label="95% CI (char × word)"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = out_dir / f"model_results_scatter.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def _md_tables() -> str:
    best_lo = lambda k: min(d[k] for d in MODELS)  # noqa: E731 (best CER/WER = lowest)
    best_hi = lambda k: max(d[k] for d in MODELS)  # noqa: E731 (best acc = highest)
    b = {
        "cer": best_lo("cer"),
        "char": best_hi("char"),
        "wer": best_lo("wer"),
        "word": best_hi("word"),
        "mcer": best_lo("mcer"),
        "mchar": best_hi("mchar"),
        "mwer": best_lo("mwer"),
        "mword": best_hi("mword"),
    }
    em = lambda v, k, s: f"**{s}**" if abs(v - b[k]) < 1e-9 else s  # noqa: E731

    out = [
        "## Corpus-level\n",
        "| Model | Arch | CER | char-acc [95% CI] | WER | word-acc [95% CI] | Size |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in MODELS:
        tag = "ours" if d["group"] == "ours" else "baseline"
        cer = em(d["cer"], "cer", f"{d['cer']:.4f}")
        char = em(d["char"], "char", _pct(d["char"])) + f" [{_pct(d['cLo'])}, {_pct(d['cHi'])}]"
        wer = em(d["wer"], "wer", f"{d['wer']:.4f}")
        word = em(d["word"], "word", _pct(d["word"])) + f" [{_pct(d['wLo'])}, {_pct(d['wHi'])}]"
        out.append(
            f"| {d['name']} ({tag}) | {d['arch']} | {cer} | {char} | {wer} | {word} | {d['size']} |"
        )
    out += [
        "\n## Per-line median\n",
        "| Model | median CER | median char-acc | median WER | median word-acc |",
        "|---|---|---|---|---|",
    ]
    for d in MODELS:
        mcer = em(d["mcer"], "mcer", f"{d['mcer']:.4f}")
        mchar = em(d["mchar"], "mchar", _pct(d["mchar"]))
        mwer = em(d["mwer"], "mwer", f"{d['mwer']:.4f}")
        mword = em(d["mword"], "mword", _pct(d["mword"]))
        out.append(f"| {d['name']} | {mcer} | {mchar} | {mwer} | {mword} |")
    return "\n".join(out)


def _write_csvs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "name",
        "group",
        "arch",
        "cer",
        "char",
        "cLo",
        "cHi",
        "wer",
        "word",
        "wLo",
        "wHi",
        "size",
    ]
    with open(out_dir / "corpus.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(MODELS)
    mcols = ["name", "mcer", "mchar", "mwer", "mword"]
    with open(out_dir / "median.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mcols, extrasaction="ignore")
        w.writeheader()
        w.writerows(MODELS)
    print(f"wrote {out_dir / 'corpus.csv'}, {out_dir / 'median.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--no-plot", action="store_true", help="tables + CSVs only")
    args = ap.parse_args()

    print(_md_tables(), "\n")
    _write_csvs(args.out_dir)
    if not args.no_plot:
        plot(args.out_dir, args.formats)


if __name__ == "__main__":
    main()
