"""Supervised feature router — can a cheap image-derivable classifier beat the 0.9743 leader?

The confidence router (argmax of calibrated per-model confidence) lost to deploying kraken+LM
everywhere (0.9732 < 0.9743). This is the stronger, direct test the user asked for: instead of
argmax, TRAIN a classifier on features we already have (both models' per-line confidences, their
gap, ink-bleed §6.5.8, line geometry) to predict *which model wins each line*, then route by it.
Fixes the two weaknesses of argmax routing — it optimises the comparison directly and can learn the
"default to the stronger model" bias. No vision model, no raw pixels → no overfit-from-pixels risk.

Protocol: train on the 600 dev, tune the decision threshold on dev, report on the full 300-val
(matches the LM λ-tuning). Labels + kraken output use the deployed kraken+LM text (0.9743) so the
comparison is fair. The KEY diagnostic is the val AUC for predicting the winner: if features can't
rank who-wins (AUC≈0.5), the idea is dead regardless of the router's corpus number.

    PROJECT_ROOT=. uv run python scripts/ocr/feature_router.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

FEATS = [
    "conf_k_mean",
    "conf_k_min",
    "conf_t_mean",
    "conf_t_min",
    "gap",
    "bleed_z",
    "w",
    "h",
    "aspect",
]


def load(p):
    return {r["stem"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def build(per_k, per_t, resc, bleed, crops):
    """Return stems, X (features), ktext, ttext, gt, label (1=TrOCR strictly wins)."""
    pk, pt = load(per_k), load(per_t)
    rk = load(resc)
    bl = json.loads(Path(bleed).read_text())["images"]
    stems = [s for s in pk if s in pt and s in rk]
    raw_bleed = np.array([bl.get(s + ".png", {}).get("bleed_score", np.nan) for s in stems])
    bz = (raw_bleed - np.nanmean(raw_bleed)) / (np.nanstd(raw_bleed) + 1e-9)  # per-split z-score
    X, ktext, ttext, gt, y = [], {}, {}, {}, []
    for i, s in enumerate(stems):
        w, h = Image.open(crops / f"{s}.png").size
        ck, ct = float(pk[s]["mean_conf"]), float(pt[s]["mean_conf"])
        X.append(
            [
                ck,
                float(pk[s]["min_conf"]),
                ct,
                float(pt[s]["min_conf"]),
                ck - ct,
                0.0 if not np.isfinite(bz[i]) else bz[i],
                w,
                h,
                w / max(1, h),
            ]
        )
        ktext[s] = rk[s]["pred_rescored"]
        ttext[s] = pt[s]["pred"]
        gt[s] = pk[s]["gt"]
        y.append(int(Levenshtein.distance(ttext[s], gt[s]) < Levenshtein.distance(ktext[s], gt[s])))
    return stems, np.array(X, float), ktext, ttext, gt, np.array(y)


def corpus(pairs):
    cd = cn = wd = wn = 0
    for p, g in pairs:
        nc, nw = max(1, len(g)), max(1, len(g.split()))
        # clip per line: over-production can't push a line past 100% wrong (CER/WER<=1, acc>=0)
        cd += min(Levenshtein.distance(p, g), nc)
        cn += nc
        wd += min(Levenshtein.distance(p.split(), g.split()), nw)
        wn += nw
    return 1 - cd / cn, 1 - wd / wn


def routed(stems, proba, thr, ktext, ttext, gt):
    return [(ttext[s], gt[s]) if proba[i] > thr else (ktext[s], gt[s]) for i, s in enumerate(stems)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-dir", type=Path, default=Path("tests/ocr/evaluations/router_dev600"))
    ap.add_argument(
        "--val-dir", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence")
    )
    ap.add_argument(
        "--dev-bleed",
        type=Path,
        default=Path(
            "tests/ocr/evaluations/router_dev600/bleed_dev/ink_bleed_20260829_095842.json"
        ),
    )
    ap.add_argument(
        "--val-bleed",
        type=Path,
        default=Path(
            "tests/ocr/evaluations/ink_bleed_val300_20260718/ink_bleed_20260718_180817.json"
        ),
    )
    ap.add_argument(
        "--dev-crops",
        type=Path,
        default=Path("data/processed/annotated_samples/OCR/full_annotated"),
    )
    ap.add_argument(
        "--val-crops", type=Path, default=Path("data/processed/annotated_samples/OCR/validation")
    )
    a = ap.parse_args()

    ds, dX, dk, dt, dg, dy = build(
        a.dev_dir / "per_line_kraken.csv",
        a.dev_dir / "per_line_trocr.csv",
        a.dev_dir / "kraken_rescored_dev.csv",
        a.dev_bleed,
        a.dev_crops,
    )
    vs, vX, vk, vt, vg, vy = build(
        a.val_dir / "per_line_kraken.csv",
        a.val_dir / "per_line_trocr.csv",
        a.val_dir / "kraken_rescored_val.csv",
        a.val_bleed,
        a.val_crops,
    )
    print(
        f"dev {len(ds)} lines ({dy.sum()} TrOCR-win, {100*dy.mean():.0f}%)  |  "
        f"val {len(vs)} lines ({vy.sum()} TrOCR-win, {100*vy.mean():.0f}%)\n"
    )

    scaler = StandardScaler().fit(dX)
    dXs, vXs = scaler.transform(dX), scaler.transform(vX)

    dep = corpus([(vk[s], vg[s]) for s in vs])
    oracle = corpus(
        [
            (
                vk[s]
                if Levenshtein.distance(vk[s], vg[s]) <= Levenshtein.distance(vt[s], vg[s])
                else vt[s],
                vg[s],
            )
            for s in vs
        ]
    )

    def eval_model(name, clf, Xtr, Xte):
        clf.fit(Xtr, dy)
        # dev-tuned threshold: maximise dev routed char-acc
        pdev = clf.predict_proba(Xtr)[:, 1]
        best_thr, best_acc = 0.5, -1
        for thr in np.linspace(0.1, 0.95, 40):
            ca = corpus(routed(ds, pdev, thr, dk, dt, dg))[0]
            if ca > best_acc:
                best_acc, best_thr = ca, thr
        pval = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(vy, pval) if vy.min() != vy.max() else float("nan")
        rows = routed(vs, pval, best_thr, vk, vt, vg)
        ca, wa = corpus(rows)
        nt = int(sum(pval > best_thr))
        cv = cross_val_score(clf, Xtr, dy, cv=5, scoring="roc_auc").mean()
        return name, auc, cv, best_thr, ca, wa, nt

    logit = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    gbm = HistGradientBoostingClassifier(
        max_depth=2, max_iter=150, learning_rate=0.05, l2_regularization=1.0, min_samples_leaf=20
    )

    print(
        f"{'':<22} | {'val AUC':>7} | {'devCV AUC':>9} | {'thr':>5} | {'char':>7} | {'word':>7} | {'→T':>4} | {'Δchar':>7}"
    )
    print(
        f"{'all kraken+LM (dep)':<22} | {'':>7} | {'':>9} | {'':>5} | {dep[0]:>7.4f} | {dep[1]:>7.4f} | {'':>4} |    —"
    )
    print(
        f"{'ORACLE':<22} | {'':>7} | {'':>9} | {'':>5} | {oracle[0]:>7.4f} | {oracle[1]:>7.4f} | {vy.sum():>4} | {100*(oracle[0]-dep[0]):>+6.2f}"
    )
    results = []
    for nm, clf, Xtr, Xte in [("logistic", logit, dXs, vXs), ("HistGBM", gbm, dX, vX)]:
        r = eval_model(nm, clf, Xtr, Xte)
        results.append(r)
        print(
            f"{'router: '+r[0]:<22} | {r[1]:>7.3f} | {r[2]:>9.3f} | {r[3]:>5.2f} | {r[4]:>7.4f} | "
            f"{r[5]:>7.4f} | {r[6]:>4} | {100*(r[4]-dep[0]):>+6.2f}"
        )

    # bootstrap Δchar vs deployed for the logistic router (2000 resamples)
    logit.fit(dXs, dy)
    pval = logit.predict_proba(vXs)[:, 1]
    thr = results[0][3]
    rrows = routed(vs, pval, thr, vk, vt, vg)
    dep_d = np.array([[Levenshtein.distance(vk[s], vg[s]), max(1, len(vg[s]))] for s in vs])
    rt_d = np.array([[Levenshtein.distance(p, g), max(1, len(g))] for p, g in rrows])
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(vs), size=(2000, len(vs)))
    d = [
        (1 - rt_d[i][:, 0].sum() / rt_d[i][:, 1].sum())
        - (1 - dep_d[i][:, 0].sum() / dep_d[i][:, 1].sum())
        for i in idx
    ]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(
        f"\nlogistic router paired bootstrap Δchar vs deployed: {100*np.mean(d):+.3f}pp "
        f"[{100*lo:+.3f}, {100*hi:+.3f}]  "
        f"{'significant' if (lo > 0 or hi < 0) else 'NOT significant (CI spans 0)'}"
    )
    # feature importances (logistic coefs on standardised features)
    print("logistic coefs:", {f: round(c, 2) for f, c in zip(FEATS, logit.coef_[0], strict=True)})


if __name__ == "__main__":
    main()
