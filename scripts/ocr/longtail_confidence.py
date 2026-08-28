"""Long-tail hard-case CONFIDENCE zoom — does the model know when it's wrong?

Prior analysis (spec §6.2) showed the per-line error is long-tailed: ~25% of lines are
perfect (p25 CER = 0) while a small tail of hard lines (CER up to ~0.23) drives the corpus
metric. This zooms into that tail and asks whether the recogniser's own CONFIDENCE flags the
hard cases (a usable triage signal — send low-confidence lines to a human) or is blindsided
(confident-but-wrong, the dangerous failure mode).

Confidence: kraken = peak-frame CTC posterior (native); TrOCR = per-token softmax prob.
Reuses the per-model collection in scripts/ocr/confidence_analysis.py.

    PROJECT_ROOT=. uv run python scripts/ocr/longtail_confidence.py \
        --model-kind kraken \
        --model models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/longtail_confidence
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reuse the per-model confidence collection from the sibling script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from confidence_analysis import kraken_preds, label_errors, trocr_preds  # noqa: E402

HIGH = 0.90  # "the model was sure" threshold, for over-confident errors
OURS = "#9C2A24"
BODY = "#8A8072"


def collect(preds, gts):
    lines = []
    for stem, gt in gts.items():
        pred, confs = preds.get(stem, ("", []))
        cer = Levenshtein.distance(pred, gt) / max(1, len(gt))
        ok = pred != "" and len(confs) == len(pred)
        err = label_errors(pred, gt) if ok else []
        lines.append(
            {
                "stem": stem,
                "gt": gt,
                "pred": pred,
                "cer": cer,
                "mean_conf": float(np.mean(confs)) if ok else float("nan"),
                "min_conf": float(np.min(confs)) if ok else float("nan"),
                "n_err": int(sum(err)) if ok else Levenshtein.distance(pred, gt),
                "overconf_err": int(
                    sum(1 for e, c in zip(err, confs, strict=True) if e and c > HIGH)
                )
                if ok
                else 0,
                "confs": list(confs) if ok else [],
                "err": err,
            }
        )
    return lines


def _mean(rows, k):
    v = [r[k] for r in rows if not np.isnan(r[k])] if rows else []
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-kind", choices=["kraken", "trocr"], default="kraken")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence"))
    ap.add_argument("--n-worst", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    label = a.label or ("kraken (CTC)" if a.model_kind == "kraken" else "TrOCR (ViT+RoBERTa)")

    crops = sorted(a.val_dir.glob("*.png"))
    gts = {
        c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip() for c in crops
    }
    gts = {k: v for k, v in gts.items() if v}
    crops = [c for c in crops if c.stem in gts]

    print(f"running {label} ({a.model_kind}) on {len(crops)} lines ...")
    preds = (
        kraken_preds(a.model, crops, a.device)
        if a.model_kind == "kraken"
        else trocr_preds(a.model, crops, a.device)
    )
    lines = collect(preds, gts)

    # ---- define the long tail: worst decile by per-line CER ----
    cers = np.array([L["cer"] for L in lines])
    thr = float(np.percentile(cers, 90))
    for L in lines:
        L["tail"] = L["cer"] >= thr and L["cer"] > 0
    valid = [L for L in lines if not np.isnan(L["mean_conf"])]
    tail = [L for L in valid if L["tail"]]
    body = [L for L in valid if not L["tail"]]

    # ---- does confidence flag the tail? ----
    y = np.array([1 if L["tail"] else 0 for L in valid])
    auc_mean = (
        roc_auc_score(y, -np.array([L["mean_conf"] for L in valid]))
        if y.min() != y.max()
        else float("nan")
    )
    auc_min = (
        roc_auc_score(y, -np.array([L["min_conf"] for L in valid]))
        if y.min() != y.max()
        else float("nan")
    )

    # ---- over-confident errors (wrong AND sure) ----
    all_err_conf = [c for L in valid for e, c in zip(L["err"], L["confs"], strict=True) if e]
    tail_err_conf = [c for L in tail for e, c in zip(L["err"], L["confs"], strict=True) if e]
    oc_all = float(np.mean([c > HIGH for c in all_err_conf])) if all_err_conf else float("nan")
    oc_tail = float(np.mean([c > HIGH for c in tail_err_conf])) if tail_err_conf else float("nan")

    # ---- char-level calibration (conf as an error detector) ----
    cc = [0 if e else 1 for L in valid for e in L["err"]]
    cf = [c for L in valid for c in L["confs"]]
    cc, cf = np.array(cc), np.array(cf)
    auc_char = roc_auc_score(1 - cc, 1 - cf) if cc.min() != cc.max() else float("nan")

    # ---- report ----
    print(f"\ntail = worst decile, CER ≥ {thr:.4f}  ({len(tail)} of {len(valid)} lines)")
    print(
        f"{'group':>6} | {'n':>3} | {'mean CER':>8} | {'mean conf':>9} | {'min conf':>8} | {'over-conf err/line':>18}"
    )
    for nm, rows in (("body", body), ("tail", tail)):
        print(
            f"{nm:>6} | {len(rows):>3} | {_mean(rows,'cer'):>8.4f} | {_mean(rows,'mean_conf'):>9.3f} | "
            f"{_mean(rows,'min_conf'):>8.3f} | {_mean(rows,'overconf_err'):>18.2f}"
        )
    print(
        f"\nconfidence as a hard-line detector (AUROC):  mean-conf {auc_mean:.3f}   min-conf {auc_min:.3f}"
    )
    print(
        f"char-level: AUROC(low-conf ⇒ error) {auc_char:.3f}   |  conf on correct {cf[cc==1].mean():.3f}  on error {cf[cc==0].mean():.3f}"
    )
    print(
        f"over-confident errors (conf>{HIGH}): all lines {100*oc_all:.1f}%   tail lines {100*oc_tail:.1f}%"
    )

    # ---- worst-N zoom table (markdown) ----
    worst = sorted(lines, key=lambda L: -L["cer"])[: a.n_worst]
    md = ["| stem | CER | errs | mean conf | min conf | GT → pred |", "|---|---|---|---|---|---|"]
    for L in worst:
        mc = "—" if np.isnan(L["mean_conf"]) else f"{L['mean_conf']:.2f}"
        mn = "—" if np.isnan(L["min_conf"]) else f"{L['min_conf']:.2f}"
        md.append(
            f"| {L['stem']} | {L['cer']:.3f} | {L['n_err']} | {mc} | {mn} | "
            f"`{L['gt'][:42]}` → `{L['pred'][:42]}` |"
        )
    (a.out / f"worst_lines_{a.model_kind}.md").write_text("\n".join(md), encoding="utf-8")

    # ---- per-line CSV dump ----
    with open(a.out / f"per_line_{a.model_kind}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["stem", "cer", "mean_conf", "min_conf", "n_err", "overconf_err", "tail", "gt", "pred"]
        )
        for L in lines:
            w.writerow(
                [
                    L["stem"],
                    f"{L['cer']:.4f}",
                    f"{L['mean_conf']:.4f}",
                    f"{L['min_conf']:.4f}",
                    L["n_err"],
                    L["overconf_err"],
                    int(L["tail"]),
                    L["gt"],
                    L["pred"],
                ]
            )

    # ---- figure ----
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Georgia", "DejaVu Serif"],
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    # (1) confidence vs per-line CER
    ax[0].scatter(
        [L["cer"] for L in body],
        [L["mean_conf"] for L in body],
        s=22,
        c=BODY,
        alpha=0.6,
        label="body (easy)",
    )
    ax[0].scatter(
        [L["cer"] for L in tail],
        [L["mean_conf"] for L in tail],
        s=34,
        c=OURS,
        alpha=0.85,
        label="tail (hard)",
    )
    ax[0].axvline(thr, ls="--", c="#555", lw=1)
    ax[0].set(
        xlabel="per-line CER",
        ylabel="mean confidence",
        title=f"Confidence vs error  (ρ={np.corrcoef([L['cer'] for L in valid],[L['mean_conf'] for L in valid])[0,1]:.2f})",
    )
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].grid(alpha=0.25)
    # (2) mean-conf distribution, body vs tail
    ax[1].hist(
        [L["mean_conf"] for L in body], bins=24, density=True, alpha=0.6, color=BODY, label="body"
    )
    ax[1].hist(
        [L["mean_conf"] for L in tail], bins=16, density=True, alpha=0.7, color=OURS, label="tail"
    )
    ax[1].set(
        xlabel="mean confidence",
        ylabel="density",
        title=f"Confidence: tail vs body (detector AUROC {auc_mean:.2f})",
    )
    ax[1].legend(frameon=False, fontsize=9)
    ax[1].grid(alpha=0.25)
    # (3) per-char confidence of the single worst line
    wl = next((L for L in worst if L["confs"]), None)
    if wl:
        xs = range(len(wl["confs"]))
        cols = [OURS if e else "#3E7A46" for e in wl["err"]]
        ax[2].bar(xs, wl["confs"], color=cols, width=0.9)
        ax[2].axhline(HIGH, ls=":", c="#555", lw=1)
        ax[2].set(
            ylim=(0, 1.02),
            xlabel="predicted character",
            ylabel="confidence",
            title=f"Worst line — {wl['stem'][:20]} (CER {wl['cer']:.2f})",
        )
        ax[2].set_xticks(list(xs))
        ax[2].set_xticklabels(list(wl["pred"]), fontsize=7)
        ax[2].margins(x=0.01)
    fig.suptitle(f"Long-tail confidence zoom — {label} (300-val)", y=1.02, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(a.out / f"longtail_confidence_{a.model_kind}.png", dpi=140, bbox_inches="tight")
    print(
        f"\nsaved {a.out}/longtail_confidence_{a.model_kind}.png  (+ worst_lines / per_line dumps)"
    )


if __name__ == "__main__":
    main()
