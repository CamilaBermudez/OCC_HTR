"""Does the model 'know when it doesn't know'? Confidence-vs-error calibration for the
two leaders (spec §6.13), kraken (CTC) and TrOCR (ViT+RoBERTa), on the 300-val.

For each model we collect per-character (predicted_char, confidence, is_error) by aligning
the prediction to GT (Levenshtein editops: replace/delete on the pred side = error), then:
  - AUROC of confidence as an error detector (does low conf flag wrong chars?)
  - ECE + reliability curve (is confidence calibrated to accuracy?)
  - mean confidence on correct vs error chars (separation)
  - line level: Spearman(mean-conf, CER) + AUROC for 'line has >=1 error'.

kraken confidence = peak-frame posterior (native). TrOCR confidence = per-token softmax
prob (greedy decode), expanded to characters.

    PROJECT_ROOT=. uv run python scripts/ocr/confidence_analysis.py \
        --kraken models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --trocr  models/ocr/finetuned/mixed_med4k_fixed \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/confidence_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def label_errors(pred: str, gt: str):
    """Per pred-char index -> is_error (True if replace/delete on the pred side)."""
    err = [False] * len(pred)
    for op in Levenshtein.editops(pred, gt):
        if op.tag in ("replace", "delete") and op.src_pos < len(pred):
            err[op.src_pos] = True
    return err


def ece(confs, correct, n_bins=10):
    confs, correct = np.asarray(confs), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, xs, accs, confm, wts = 0.0, [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if not m.any():
            continue
        a, c, w = correct[m].mean(), confs[m].mean(), m.mean()
        e += w * abs(a - c)
        xs.append((lo + hi) / 2)
        accs.append(a)
        confm.append(c)
        wts.append(w)
    return e, np.array(xs), np.array(accs), np.array(confm), np.array(wts)


def kraken_preds(model_path, crops, device="cpu"):
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
            # kraken's reported per-char conf (p[3]) is near-saturated and barely
            # discriminative (AUROC ~0.59); the genuine confidence is the peak-frame
            # softmax posterior (AUROC ~0.97). Use that.
            om = np.asarray(net.outputs)
            om = om[0] if om.ndim == 3 else om  # [labels, frames]
            chars, confs = [], []
            for ch, s, e, _ in preds:
                e = max(e, s)
                peak = s + int(om[:, s : e + 1].max(axis=0).argmax())
                chars.append(ch)
                confs.append(float(om[:, peak].max()))
            out[crop.stem] = ("".join(chars), confs)
        except Exception:  # noqa: BLE001
            out[crop.stem] = ("", [])
    return out


def trocr_preds(model_dir, crops, device="mps"):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    from src.ocr.image_prep import prepare_image

    resize = "pad"
    if (model_dir / "resize_mode.txt").is_file():
        resize = (model_dir / "resize_mode.txt").read_text().strip() or "pad"
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    out = {}
    for crop in crops:
        img = prepare_image(Image.open(crop).convert("RGB"), proc, resize)
        pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            g = model.generate(
                pixel_values=pv,
                num_beams=1,
                max_length=128,
                output_scores=True,
                return_dict_in_generate=True,
            )
        trans = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
        gen_ids = g.sequences[0, -trans.shape[1] :]
        probs = trans[0].exp().tolist()
        chars, confs = [], []
        for tid, pr in zip(gen_ids.tolist(), probs, strict=True):
            piece = tok.decode([tid], skip_special_tokens=True)
            for ch in piece:
                chars.append(ch)
                confs.append(float(pr))
        out[crop.stem] = ("".join(chars), confs)
    return out


def analyse(name, preds, gts):
    cc, cf = [], []  # per-char is_correct, confidence
    lc, lcer = [], []  # per-line mean-conf, CER
    for stem, gt in gts.items():
        pred, confs = preds.get(stem, ("", []))
        if not pred or len(confs) != len(pred):
            continue
        err = label_errors(pred, gt)
        for is_err, cnf in zip(err, confs, strict=True):
            cc.append(0 if is_err else 1)
            cf.append(cnf)
        lc.append(float(np.mean(confs)))
        lcer.append(Levenshtein.distance(pred, gt) / max(1, len(gt)))
    cc, cf = np.array(cc), np.array(cf)
    err_lab = 1 - cc  # 1 = error
    auroc = roc_auc_score(err_lab, 1 - cf) if err_lab.min() != err_lab.max() else float("nan")
    e, xs, accs, confm, wts = ece(cf, cc)
    line_has_err = (np.array(lcer) > 0).astype(int)
    line_auroc = (
        roc_auc_score(line_has_err, -np.array(lc))
        if line_has_err.min() != line_has_err.max()
        else float("nan")
    )
    rho, _ = spearmanr(lc, lcer)
    res = {
        "name": name,
        "n_char": len(cc),
        "char_acc": cc.mean(),
        "conf_correct": cf[cc == 1].mean(),
        "conf_error": cf[cc == 0].mean(),
        "auroc_char": auroc,
        "ece": e,
        "line_auroc": line_auroc,
        "spearman_conf_cer": rho,
        "rel_x": xs,
        "rel_acc": accs,
        "rel_conf": confm,
        "cf": cf,
        "cc": cc,
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kraken", type=Path, required=True, help="a CTC (.mlmodel) model")
    ap.add_argument("--kraken-label", default="kraken (CTC)")
    ap.add_argument("--trocr", type=Path, default=None, help="optional TrOCR checkpoint dir")
    ap.add_argument("--trocr-label", default="TrOCR (ViT+RoBERTa)")
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tests/ocr/evaluations/confidence_analysis"))
    ap.add_argument("--device-trocr", default="mps")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    crops = sorted(a.val_dir.glob("*.png"))
    gts = {
        c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip() for c in crops
    }
    gts = {k: v for k, v in gts.items() if v}
    crops = [c for c in crops if c.stem in gts]

    print(f"running {a.kraken_label} (CTC) ...")
    results = [analyse(a.kraken_label, kraken_preds(a.kraken, crops), gts)]
    if a.trocr is not None:
        print(f"running {a.trocr_label} ...")
        results.append(analyse(a.trocr_label, trocr_preds(a.trocr, crops, a.device_trocr), gts))

    print(
        f"\n{'model':<22} | {'char_acc':>8} | {'conf✓':>6} | {'conf✗':>6} | "
        f"{'AUROC_ch':>8} | {'ECE':>6} | {'AUROC_ln':>8} | {'ρ(cf,cer)':>9}"
    )
    for r in results:
        print(
            f"{r['name']:<22} | {r['char_acc']:>8.4f} | {r['conf_correct']:>6.3f} | "
            f"{r['conf_error']:>6.3f} | {r['auroc_char']:>8.3f} | {r['ece']:>6.3f} | "
            f"{r['line_auroc']:>8.3f} | {r['spearman_conf_cer']:>9.3f}"
        )

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for r in results:
        axes[0, 0].plot(r["rel_conf"], r["rel_acc"], "o-", label=r["name"])
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
    txt = "\n".join(
        f"{r['name']}\n  char_acc {r['char_acc']:.3f}  AUROC_char {r['auroc_char']:.3f}\n"
        f"  conf(correct) {r['conf_correct']:.3f}  conf(error) {r['conf_error']:.3f}\n"
        f"  ECE {r['ece']:.3f}  line-AUROC {r['line_auroc']:.3f}  rho {r['spearman_conf_cer']:.3f}\n"
        for r in results
    )
    axes[1, 1].text(0.0, 0.95, txt, va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    p = a.out / "confidence_calibration.png"
    fig.savefig(p, dpi=130)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
