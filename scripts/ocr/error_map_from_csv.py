"""Per-page error map on the 300-val for the DEPLOYED leaders, from the per-line eval CSVs
(run_evaluate_ocr.py output: stem, model, n_chars_ref, edit_chars, cer, ...). Unlike
error_distribution.py this re-uses existing predictions — no re-transcription — and the
300-val is held out for both deployed models, so the map is unbiased.

    PROJECT_ROOT=. uv run python scripts/ocr/error_map_from_csv.py \
        --csv "kraken+LM=tests/ocr/evaluations/krakenLM_val300/krakenLM_val300.csv" \
        --csv "TrOCR=tests/ocr/evaluations/mixedmed4k_val300/mixedmed4k_val300.csv" \
        --out tests/ocr/evaluations/error_map_val300_leaders
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PAGE = re.compile(r"_line_\d+$")


def _page(stem: str) -> str:
    return _PAGE.sub("", stem)


def load(csv_path: Path) -> dict[str, tuple[int, int]]:
    """{stem: (edit_chars, n_chars_ref)} from a per-line eval CSV."""
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[row["stem"]] = (int(row["edit_chars"]), int(row["n_chars_ref"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", required=True, help="label=path.csv (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    models = {}
    for spec in a.csv:
        label, path = spec.split("=", 1)
        models[label] = load(Path(path))

    pages = sorted({_page(s) for m in models.values() for s in m})
    report = {"overall": {}, "by_page": []}
    for label, per in models.items():
        tot_e = sum(e for e, _ in per.values())
        tot_n = sum(n for _, n in per.values())
        perfect = np.mean([e == 0 for e, _ in per.values()]) * 100
        report["overall"][label] = {
            "n_lines": len(per),
            "corpus_cer": round(tot_e / max(1, tot_n), 4),
            "pct_perfect": round(float(perfect), 1),
        }

    for pg in pages:
        row = {"page": pg}
        for label, per in models.items():
            items = [(e, n) for s, (e, n) in per.items() if _page(s) == pg]
            if items:
                e = sum(i[0] for i in items)
                n = sum(i[1] for i in items)
                row[f"{label}_cer"] = round(e / max(1, n), 4)
                row[f"{label}_n"] = len(items)
                row[f"{label}_perfect"] = round(100 * np.mean([i[0] == 0 for i in items]), 1)
        report["by_page"].append(row)
    (a.out / "error_map.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("OVERALL (300-val):", json.dumps(report["overall"], indent=2))
    labels = list(models)
    x = np.arange(len(pages))
    w = 0.8 / len(labels)
    colors = ["#e57373", "#64b5f6", "#81c784", "#ba68c8"]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    for i, label in enumerate(labels):
        cers = [
            next(
                (
                    r[f"{label}_cer"]
                    for r in report["by_page"]
                    if r["page"] == pg and f"{label}_cer" in r
                ),
                0.0,
            )
            for pg in pages
        ]
        ax[0].bar(
            x + i * w,
            cers,
            w,
            label=f"{label} (corpus {report['overall'][label]['corpus_cer']})",
            color=colors[i % len(colors)],
        )
        perf = [
            next(
                (r.get(f"{label}_perfect", 0.0) for r in report["by_page"] if r["page"] == pg), 0.0
            )
            for pg in pages
        ]
        ax[1].bar(x + i * w, perf, w, label=label, color=colors[i % len(colors)])
    ax[0].set(xticks=x, ylabel="corpus CER", title="Per-page CER on 300-val — deployed leaders")
    ax[0].set_xticklabels(pages, rotation=90, fontsize=5)
    ax[0].legend()
    ax[1].set(
        xticks=x, ylabel="% lines perfect", title="Per-page perfectly-predicted rate (300-val)"
    )
    ax[1].set_xticklabels(pages, rotation=90, fontsize=5)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(a.out / "error_map_val300.png", dpi=130)
    print(f"pages={len(pages)}  saved {a.out}/error_map_val300.png")


if __name__ == "__main__":
    main()
