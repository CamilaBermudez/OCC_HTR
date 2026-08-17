"""Where in the manuscript do errors concentrate? Per-line CER + per-page aggregation
over the annotated GT lines, transcribed by a CTC model (default frozen catmus — it
never trained on any of our lines, so its error map is UNBIASED across the whole 600+300).

Outputs: a per-page table (n lines, mean CER, % perfectly-predicted), a JSON, and a figure
(per-page CER across the manuscript, per-line CER histogram, per-page %-perfect).

    PROJECT_ROOT=. uv run python scripts/ocr/error_distribution.py \
        --model models/ocr/catmus-medieval.mlmodel \
        --gt-dirs data/processed/annotated_samples/OCR/full_annotated \
                  data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/error_distribution_catmus
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from kraken import rpred
from kraken.lib import models
from rapidfuzz.distance import Levenshtein

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.ocr.transcribe_line_crops import _synthesised_seg  # noqa: E402

_PAGE = re.compile(r"_line_\d+$")


def _page(stem: str) -> str:
    return _PAGE.sub("", stem)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/ocr/catmus-medieval.mlmodel"))
    ap.add_argument("--gt-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    crops = []
    for d in a.gt_dirs:
        crops += sorted(d.glob("*.png"))
    net = models.load_any(str(a.model), device=a.device)

    per_line = []  # (page, stem, cer, n_ref, perfect)
    for cr in crops:
        g = cr.with_name(cr.stem + ".gt.txt")
        if not g.is_file():
            continue
        gt = g.read_text(encoding="utf-8").strip()
        if not gt:
            continue
        try:
            im, seg = _synthesised_seg(cr)
            preds = list(rpred.rpred(net, im, seg))
            pred = (getattr(preds[0], "prediction", "") or "") if preds else ""
        except Exception:  # noqa: BLE001
            pred = ""
        d = Levenshtein.distance(pred, gt)
        per_line.append((_page(cr.stem), cr.stem, d / max(1, len(gt)), len(gt), int(d == 0)))

    # per-page aggregation, in manuscript (sorted-page) order
    by_page: dict[str, list] = defaultdict(list)
    for pg, _stem, cer, nref, perfect in per_line:
        by_page[pg].append((cer, nref, perfect))
    pages = sorted(by_page)
    rows = []
    for pg in pages:
        vals = by_page[pg]
        cers = np.array([v[0] for v in vals])
        nref = np.array([v[1] for v in vals])
        perf = np.array([v[2] for v in vals])
        # corpus CER for the page = sum edits / sum ref chars
        corpus_cer = float((cers * nref).sum() / nref.sum()) if nref.sum() else 0.0
        rows.append(
            {
                "page": pg,
                "n_lines": len(vals),
                "corpus_cer": round(corpus_cer, 4),
                "mean_line_cer": round(float(cers.mean()), 4),
                "pct_perfect": round(100 * float(perf.mean()), 1),
            }
        )

    all_cer = np.array([v[2] for v in per_line])  # per-line cer
    all_perf = np.array([v[4] for v in per_line])
    overall = {
        "n_lines": len(per_line),
        "n_pages": len(pages),
        "overall_mean_line_cer": round(float(all_cer.mean()), 4),
        "overall_pct_perfect": round(100 * float(all_perf.mean()), 1),
        "model": str(a.model),
    }
    (a.out / "error_distribution.json").write_text(
        json.dumps({"overall": overall, "by_page": rows}, indent=2), encoding="utf-8"
    )

    print(f"{'page':<22} | {'n':>4} | {'corpusCER':>9} | {'%perfect':>8}")
    for r in rows:
        print(
            f"{r['page']:<22} | {r['n_lines']:>4} | {r['corpus_cer']:>9.4f} | {r['pct_perfect']:>8.1f}"
        )
    print(
        f"\nOVERALL: {overall['n_lines']} lines / {overall['n_pages']} pages | "
        f"mean-line CER {overall['overall_mean_line_cer']} | "
        f"{overall['overall_pct_perfect']}% perfect"
    )

    # figure
    fig, ax = plt.subplots(3, 1, figsize=(13, 11))
    x = np.arange(len(pages))
    ax[0].bar(x, [r["corpus_cer"] for r in rows], color="#e57373")
    ax[0].axhline(
        overall["overall_mean_line_cer"], ls="--", c="k", lw=1, label="overall mean-line CER"
    )
    ax[0].set(
        xticks=x,
        ylabel="corpus CER",
        title=f"Per-page error across the manuscript ({a.model.name})",
    )
    ax[0].set_xticklabels([r["page"] for r in rows], rotation=90, fontsize=5)
    ax[0].legend()
    ax[1].hist(all_cer, bins=40, color="#64b5f6")
    ax[1].set(xlabel="per-line CER", ylabel="lines", title="Per-line CER distribution")
    ax[2].bar(x, [r["pct_perfect"] for r in rows], color="#81c784")
    ax[2].set(xticks=x, ylabel="% lines perfect", title="Per-page perfectly-predicted rate")
    ax[2].set_xticklabels([r["page"] for r in rows], rotation=90, fontsize=5)
    fig.tight_layout()
    p = a.out / "error_distribution.png"
    fig.savefig(p, dpi=130)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
