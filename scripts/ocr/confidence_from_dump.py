"""Local calibration ('does the model know when it's wrong?', spec §6.13) for catmus (CTC,
run locally) + Medusa (VLM, from the cluster-produced JSON) over the 300-val. Medusa JSON
is {stem: {text, confs}} from medusa_confidence_dump.py; catmus is transcribed here with
per-char peak-frame confidence. Same metrics as confidence_analysis.py (AUROC, ECE,
conf✓/conf✗, line-AUROC, ρ) + a combined reliability/histogram plot.

    PROJECT_ROOT=. uv run python scripts/ocr/confidence_from_dump.py \
        --catmus models/ocr/catmus-medieval.mlmodel \
        --medusa-json preds/medusa_conf_val300.json \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/confidence_catmus_medusa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def label_errors(pred: str, gt: str) -> list[bool]:
    err = [False] * len(pred)
    for op in Levenshtein.editops(pred, gt):
        if op.tag in ("replace", "delete") and op.src_pos < len(pred):
            err[op.src_pos] = True
    return err


def ece(confs, correct, n_bins=10):
    confs, correct = np.asarray(confs), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, xs, accs, cms = 0.0, [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if not m.any():
            continue
        e += m.mean() * abs(correct[m].mean() - confs[m].mean())
        xs.append((lo + hi) / 2)
        accs.append(correct[m].mean())
        cms.append(confs[m].mean())
    return e, np.array(cms), np.array(accs)


def analyse(name: str, preds: dict[str, tuple[str, list[float]]], gts: dict[str, str]) -> dict:
    cc, cf, lc, lcer = [], [], [], []
    for stem, gt in gts.items():
        pred, confs = preds.get(stem, ("", []))
        if not pred or len(confs) != len(pred):
            continue
        for is_err, cnf in zip(label_errors(pred, gt), confs, strict=True):
            cc.append(0 if is_err else 1)
            cf.append(cnf)
        lc.append(float(np.mean(confs)))
        lcer.append(Levenshtein.distance(pred, gt) / max(1, len(gt)))
    cc, cf = np.array(cc), np.array(cf)
    auroc = roc_auc_score(1 - cc, 1 - cf) if cc.min() != cc.max() else float("nan")
    e, conf_bins, acc_bins = ece(cf, cc)
    line_err = (np.array(lcer) > 0).astype(int)
    line_auroc = (
        roc_auc_score(line_err, -np.array(lc)) if line_err.min() != line_err.max() else float("nan")
    )
    rho, _ = spearmanr(lc, lcer)
    return {
        "name": name,
        "n_char": len(cc),
        "char_acc": float(cc.mean()),
        "conf_correct": float(cf[cc == 1].mean()),
        "conf_error": float(cf[cc == 0].mean()),
        "auroc": auroc,
        "ece": e,
        "line_auroc": line_auroc,
        "rho": rho,
        "conf_bins": conf_bins,
        "acc_bins": acc_bins,
        "cf": cf,
        "cc": cc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catmus", type=Path, required=True)
    ap.add_argument("--medusa-json", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    crops = sorted(a.val_dir.glob("*.png"))
    gts = {
        c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip() for c in crops
    }
    gts = {k: v for k, v in gts.items() if v}

    from kraken.lib import models  # noqa: PLC0415
    from kraken.lib.dataset import ImageInputTransforms  # noqa: PLC0415
    from kraken.lib.segmentation import extract_polygons  # noqa: PLC0415

    from src.ocr.transcribe_line_crops import _synthesised_seg  # noqa: PLC0415

    net = models.load_any(str(a.catmus), device="cpu")
    b, ch_, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, ch_, (16, 0), valid_norm=False)
    catmus_preds: dict[str, tuple[str, list[float]]] = {}
    for c in crops:
        if c.stem not in gts:
            continue
        try:
            im, seg = _synthesised_seg(c)
            box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
            preds = net.predict(ts(box).unsqueeze(0))[0]  # [(char, start, end, conf)]
            catmus_preds[c.stem] = (
                "".join(p[0] for p in preds),
                [float(p[3]) for p in preds],
            )
        except Exception:  # noqa: BLE001
            catmus_preds[c.stem] = ("", [])

    med = json.loads(a.medusa_json.read_text(encoding="utf-8"))
    medusa_preds = {k: (v["text"], v["confs"]) for k, v in med.items()}

    results = [
        analyse("catmus (CTC)", catmus_preds, gts),
        analyse("Medusa (VLM)", medusa_preds, gts),
    ]

    print(
        f"\n{'model':<16} | {'char_acc':>8} | {'conf✓':>6} | {'conf✗':>6} | {'AUROC':>6} | "
        f"{'ECE':>6} | {'AUROC_ln':>8} | {'ρ(cf,cer)':>9}"
    )
    for r in results:
        print(
            f"{r['name']:<16} | {r['char_acc']:>8.4f} | {r['conf_correct']:>6.3f} | "
            f"{r['conf_error']:>6.3f} | {r['auroc']:>6.3f} | {r['ece']:>6.3f} | "
            f"{r['line_auroc']:>8.3f} | {r['rho']:>9.3f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for r in results:
        axes[0, 0].plot(r["conf_bins"], r["acc_bins"], "o-", label=r["name"])
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    axes[0, 0].set(xlabel="confidence", ylabel="empirical accuracy", title="Reliability (char)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    for ax, r in zip((axes[0, 1], axes[1, 0]), results, strict=False):
        ax.hist(r["cf"][r["cc"] == 1], bins=30, alpha=0.6, density=True, label="correct")
        ax.hist(r["cf"][r["cc"] == 0], bins=30, alpha=0.6, density=True, label="error")
        ax.set(xlabel="confidence", ylabel="density", title=f"{r['name']}: conf | correctness")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.95,
        "\n".join(
            f"{r['name']}\n  char_acc {r['char_acc']:.3f}  AUROC {r['auroc']:.3f}\n"
            f"  conf✓ {r['conf_correct']:.3f}  conf✗ {r['conf_error']:.3f}\n"
            f"  ECE {r['ece']:.3f}  line-AUROC {r['line_auroc']:.3f}  ρ {r['rho']:.3f}\n"
            for r in results
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(a.out / "confidence_catmus_medusa.png", dpi=130)
    (a.out / "metrics.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("conf_bins", "acc_bins", "cf", "cc")}
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {a.out}/confidence_catmus_medusa.png")


if __name__ == "__main__":
    main()
