"""Does per-line error depend on line LENGTH? CER/WER vs GT length (n_chars_ref) for the
deployed leaders on the held-out 300-val, straight from the per-line eval CSVs
(run_evaluate_ocr.py output) — no re-transcription, same reuse pattern as
error_map_from_csv.py. All models are scored on the same GT lines, so length bins
(quantile edges over the shared GT lengths) are identical across models.

Per length bin: n lines, corpus CER (sum edits / sum ref chars), mean/median line CER,
% perfect lines, corpus WER. Plus Pearson + Spearman correlation of length vs line CER.
Note: %-perfect falls with length half-mechanically (more chars = more chances for >=1
edit); corpus CER per bin is the honest "are long lines proportionally worse" number.

    PROJECT_ROOT=. uv run python scripts/ocr/error_vs_length.py \
        --csv "kraken+LM=tests/ocr/evaluations/krakenLM_val300/krakenLM_val300.csv" \
        --csv "TrOCR=tests/ocr/evaluations/mixedmed4k_val300/mixedmed4k_val300.csv" \
        --out tests/ocr/evaluations/error_vs_length_val300_leaders
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#e57373", "#64b5f6", "#81c784", "#ba68c8"]


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "analysis", run_name: str | None = None
):
    """File + console logger, same shape as the src.ocr modules."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logs_dir) / f"{run_name}.log"
    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("=== %s Run Started | Run: %s ===", task_name.upper(), run_name)
    logger.info("Log file: %s", log_file)
    return logger, str(log_file)


def load(csv_path: Path) -> dict[str, dict]:
    """{stem: row-with-int/float-fields} from a per-line eval CSV."""
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[row["stem"]] = {
                "n_chars_ref": int(row["n_chars_ref"]),
                "n_words_ref": int(row["n_words_ref"]),
                "edit_chars": int(row["edit_chars"]),
                "edit_words": int(row["edit_words"]),
                "cer": float(row["cer"]),
                "wer": float(row["wer"]),
            }
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), enough for Spearman without scipy."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", action="append", required=True, help="label=path.csv (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=4, help="number of quantile length bins")
    ap.add_argument("--run-name", default=None, help="log file name under logs/analysis/")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    run_name = a.run_name or f"error_vs_length_{datetime.datetime.now():%Y%m%d}"
    logger, _ = setup_simple_logging("logs/analysis", "error_vs_length", run_name)
    logger.info("Config: %s", json.dumps({"csv": a.csv, "out": str(a.out), "bins": a.bins}))

    models: dict[str, dict[str, dict]] = {}
    for spec in a.csv:
        label, path = spec.split("=", 1)
        models[label] = load(Path(path))
        logger.info("%s: %d lines from %s", label, len(models[label]), path)

    # shared GT lengths (same stems/GT for every model); quantile bin edges over them
    lengths = {s: r["n_chars_ref"] for per in models.values() for s, r in per.items()}
    lens = np.array(sorted(lengths.values()))
    edges = np.unique(np.quantile(lens, np.linspace(0, 1, a.bins + 1)).round().astype(int))
    logger.info(
        "GT length: min %d / median %d / max %d | bin edges %s",
        lens.min(),
        int(np.median(lens)),
        lens.max(),
        edges.tolist(),
    )

    def bin_of(n: int) -> int:
        return int(np.clip(np.searchsorted(edges, n, side="right") - 1, 0, len(edges) - 2))

    bin_labels = [
        f"{edges[i]}–{edges[i + 1]}" + (" (incl.)" if i == len(edges) - 2 else "")
        for i in range(len(edges) - 1)
    ]

    report: dict = {"bin_edges": edges.tolist(), "bin_labels": bin_labels, "models": {}}
    for label, per in models.items():
        n = np.array([r["n_chars_ref"] for r in per.values()])
        cer = np.array([r["cer"] for r in per.values()])
        rows = []
        for b in range(len(edges) - 1):
            sel = [r for r in per.values() if bin_of(r["n_chars_ref"]) == b]
            ec = sum(r["edit_chars"] for r in sel)
            nc = sum(r["n_chars_ref"] for r in sel)
            ew = sum(r["edit_words"] for r in sel)
            nw = sum(r["n_words_ref"] for r in sel)
            lcer = np.array([r["cer"] for r in sel])
            rows.append(
                {
                    "bin": bin_labels[b],
                    "n_lines": len(sel),
                    "corpus_cer": round(ec / max(1, nc), 4),
                    "mean_line_cer": round(float(lcer.mean()), 4) if len(sel) else None,
                    "median_line_cer": round(float(np.median(lcer)), 4) if len(sel) else None,
                    "pct_perfect": round(100 * float(np.mean(lcer == 0)), 1) if len(sel) else None,
                    "corpus_wer": round(ew / max(1, nw), 4),
                }
            )
        report["models"][label] = {
            "n_lines": len(per),
            "pearson_len_cer": round(float(np.corrcoef(n, cer)[0, 1]), 4),
            "spearman_len_cer": round(spearman(n, cer), 4),
            "by_bin": rows,
        }
        logger.info(
            "%s | Pearson(len, CER) %.4f | Spearman(len, CER) %.4f",
            label,
            report["models"][label]["pearson_len_cer"],
            report["models"][label]["spearman_len_cer"],
        )
        logger.info(
            "%s | %-14s | %4s | %9s | %8s | %10s | %8s | %8s",
            label,
            "len bin",
            "n",
            "corpusCER",
            "meanCER",
            "medianCER",
            "%perfect",
            "corpusWER",
        )
        for r in rows:
            logger.info(
                "%s | %-14s | %4d | %9.4f | %8.4f | %10.4f | %8.1f | %8.4f",
                label,
                r["bin"],
                r["n_lines"],
                r["corpus_cer"],
                r["mean_line_cer"],
                r["median_line_cer"],
                r["pct_perfect"],
                r["corpus_wer"],
            )

    (a.out / "error_vs_length.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = list(models)
    x = np.arange(len(bin_labels))
    w = 0.8 / len(labels)
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    for i, label in enumerate(labels):
        c = COLORS[i % len(COLORS)]
        per = models[label]
        rows = report["models"][label]["by_bin"]
        ax[0].scatter(
            [r["n_chars_ref"] for r in per.values()],
            [r["cer"] for r in per.values()],
            s=9,
            alpha=0.45,
            color=c,
            label=f"{label} (Spearman {report['models'][label]['spearman_len_cer']:+.3f})",
        )
        ax[1].bar(x + i * w, [r["corpus_cer"] for r in rows], w, label=label, color=c)
        ax[2].bar(x + i * w, [r["pct_perfect"] for r in rows], w, label=label, color=c)
    ax[0].set(
        xlabel="GT line length (chars)",
        ylabel="line CER",
        title="Per-line CER vs GT length (300-val, deployed leaders)",
    )
    for e in edges[1:-1]:
        ax[0].axvline(e, ls=":", c="gray", lw=0.8)
    ax[1].set(
        xticks=x + w * (len(labels) - 1) / 2,
        ylabel="corpus CER",
        title="Corpus CER by length bin (quantile bins over shared GT lengths)",
    )
    ax[1].set_xticklabels(bin_labels)
    ax[2].set(
        xticks=x + w * (len(labels) - 1) / 2,
        xlabel="GT length bin (chars)",
        ylabel="% lines perfect",
        title="Perfectly-predicted rate by length bin",
    )
    ax[2].set_xticklabels(bin_labels)
    for axis in ax:
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out / "error_vs_length.png", dpi=130)
    logger.info("Artefacts: %s/{error_vs_length.json,error_vs_length.png}", a.out)


if __name__ == "__main__":
    main()
