"""Temperature scaling for the kraken (CTC) confidence — is the over-confidence fixable?

The long-tail zoom (spec §6.13, corrected) found kraken's genuine peak-frame posterior is
already a GOOD error detector (char AUROC ~0.96) but MILDLY over-confident. Temperature scaling
(Guo et al. 2017) softens the softmax by a scalar T fit on a dev split — it CANNOT change the
argmax (accuracy is untouched), only the confidence magnitudes. Two separable questions:
  (a) ECE / over-confidence — does softening make confidence match empirical accuracy? (expected:
      yes, a clear improvement, since kraken is mildly over-confident.)
  (b) discrimination (AUROC) — does it make confidence a better error detector? (near-monotonic,
      so expected to change little — and the ranking is already good, so this is a polish.)

Uses each emitted char's peak-frame posterior col (softmax over the alphabet); softened
conf_T = max(col**(1/T)) / sum(col**(1/T)). Fits T on a 100-line dev split (seed 42), reports
before/after on the held-out rest.

    PROJECT_ROOT=. uv run python scripts/ocr/temperature_scale_kraken.py \
        --model models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/longtail_confidence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confidence_analysis import ece, label_errors  # noqa: E402


def kraken_cols(model_path, crops, device="cpu"):
    """Per line: (pred string, [peak-frame posterior vector per char])."""
    from kraken.lib import models
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.lib.segmentation import extract_polygons

    from src.ocr.transcribe_line_crops import _synthesised_seg

    net = models.load_any(str(model_path), device=device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    out = {}
    for crop in crops:
        try:
            im, seg = _synthesised_seg(crop)
            box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
            preds = net.predict(ts(box).unsqueeze(0))[0]  # [(char, s, e, conf)]
            om = np.asarray(net.outputs)
            om = om[0] if om.ndim == 3 else om  # [labels, frames]
            chars, cols = [], []
            for ch, s, e, _ in preds:
                e = max(e, s)
                peak = s + int(om[:, s : e + 1].max(axis=0).argmax())
                col = om[:, peak].astype(np.float64)
                col = col / max(col.sum(), 1e-9)  # normalise to a distribution
                chars.append(ch)
                cols.append(col)
            out[crop.stem] = ("".join(chars), cols)
        except Exception:  # noqa: BLE001
            out[crop.stem] = ("", [])
    return out


def conf_T(cols, T):
    """Softened peak-char confidence for a [N, C] posterior array at temperature T."""
    pT = np.power(cols, 1.0 / T)
    return pT.max(axis=1) / pT.sum(axis=1)


def report(name, conf, y):
    """y = 1 if the char is CORRECT. Returns (ece, auroc, conf_correct, conf_error, overconf)."""
    correct = np.asarray(y)
    e = ece(conf, correct)[0]
    auroc = roc_auc_score(1 - correct, 1 - conf) if correct.min() != correct.max() else float("nan")
    oc = float(np.mean(conf[correct == 0] > 0.9))
    return {
        "name": name,
        "ece": e,
        "auroc": auroc,
        "cc": float(conf[correct == 1].mean()),
        "ce": float(conf[correct == 0].mean()),
        "overconf": oc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument(
        "--dev-dir",
        type=Path,
        default=Path("data/processed/annotated_samples/OCR/full_annotated"),
        help="set to FIT T on (the 600 annotated — held out from the 300-val), "
        "so the 300-val stays a pure held-out report, consistent with the LM λ-tuning",
    )
    ap.add_argument("--val-dir", type=Path, required=True, help="the 300-val — REPORTED on in FULL")
    ap.add_argument("--out", type=Path, default=Path("tests/ocr/evaluations/longtail_confidence"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    def collect(dir_):
        crops = sorted(dir_.glob("*.png"))
        gts = {
            c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip()
            for c in crops
        }
        gts = {k: v for k, v in gts.items() if v}
        crops = [c for c in crops if c.stem in gts]
        preds = kraken_cols(a.model, crops, a.device)
        cols, y = [], []
        for stem, (pred, cc) in preds.items():
            if not pred or len(cc) != len(pred):
                continue
            err = label_errors(pred, gts[stem])
            for col, is_err in zip(cc, err, strict=True):
                cols.append(col)
                y.append(0 if is_err else 1)
        return np.array(cols), np.array(y), len(crops)

    print(f"fitting T on dev = {a.dev_dir} (the 600 annotated) ...")
    dc, dy, ndev = collect(a.dev_dir)
    print(f"reporting on val = {a.val_dir} (the full 300-val) ...")
    tc, ty, nval = collect(a.val_dir)
    print(
        f"dev {ndev} lines / {len(dy)} chars (acc {dy.mean():.3f})  |  "
        f"val {nval} lines / {len(ty)} chars (acc {ty.mean():.3f})"
    )

    # ---- fit T on dev: minimise binary NLL of correctness vs softened confidence ----
    def nll(T):
        c = np.clip(conf_T(dc, T), 1e-6, 1 - 1e-6)
        return -np.mean(dy * np.log(c) + (1 - dy) * np.log(1 - c))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    Tstar = float(res.x)
    print(f"\nfitted T* = {Tstar:.3f}  (T>1 softens; T=1 = no change)\n")

    before = report("T=1 (raw)", conf_T(tc, 1.0), ty)
    after = report(f"T*={Tstar:.2f}", conf_T(tc, Tstar), ty)
    print(
        f"{'':>12} | {'ECE':>7} | {'AUROC':>7} | {'conf✓':>6} | {'conf✗':>6} | {'over-conf err':>13}"
    )
    for r in (before, after):
        print(
            f"{r['name']:>12} | {r['ece']:>7.4f} | {r['auroc']:>7.3f} | {r['cc']:>6.3f} | "
            f"{r['ce']:>6.3f} | {100*r['overconf']:>12.1f}%"
        )
    print(
        f"\nΔ ECE {after['ece']-before['ece']:+.4f}   Δ AUROC {after['auroc']-before['auroc']:+.3f}   "
        f"Δ over-conf-err {100*(after['overconf']-before['overconf']):+.1f}pp"
    )
    print(
        "=> ECE gain = over-confidence fixed; AUROC ~unchanged = discrimination is fundamental, "
        "not a saturation artefact."
        if abs(after["auroc"] - before["auroc"]) < 0.03
        else "=> AUROC also moved — some discrimination was hidden by saturation."
    )

    # ---- reliability curves before/after (test) ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for i, (r, T) in enumerate([(before, 1.0), (after, Tstar)]):
        _, xs, accs, confm, _ = ece(conf_T(tc, T), ty)
        ax[0].plot(confm, accs, "o-", label=r["name"])
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax[0].set(xlabel="confidence", ylabel="empirical accuracy", title="Reliability (300-val)")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.3)
    ax[1].hist(conf_T(tc, 1.0), bins=30, alpha=0.5, density=True, label="T=1")
    ax[1].hist(conf_T(tc, Tstar), bins=30, alpha=0.5, density=True, label=f"T*={Tstar:.2f}")
    ax[1].set(xlabel="confidence", ylabel="density", title="Confidence distribution")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.3)
    fig.suptitle(
        f"kraken CTC temperature scaling (T*={Tstar:.2f}) — fit on 600, report on 300-val", y=1.02
    )
    fig.tight_layout()
    fig.savefig(a.out / "temperature_scaling_kraken.png", dpi=140, bbox_inches="tight")
    print(f"\nsaved {a.out}/temperature_scaling_kraken.png")


if __name__ == "__main__":
    main()
