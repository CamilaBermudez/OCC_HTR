"""Ensemble diagnostic for the two leaders (spec §6.13): kraken 0.9710 + P1 char-LM
rescore vs TrOCR (ViT+RoBERTa) 1-best, on the 300-val.

Reports, corpus-level: each model alone, the ORACLE best-of-both (per line pick the
prediction closest to GT — the ceiling any ensemble could reach), and a REALIZABLE
LM-arbitrated pick (per line choose the hypothesis the char-LM likes better — no GT
peeking). If oracle >> kraken but LM-arbitrated <= kraken, a practical ensemble can't
beat the strong model.

    PROJECT_ROOT=. uv run python scripts/ocr/ensemble_oracle.py \
        --kraken models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --trocr  models/ocr/finetuned/mixed_med4k_fixed \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --lm-dir  data/processed/annotated_samples/OCR/full_annotated \
        --lam 0.2 --device-trocr mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from src.ocr.char_lm import CharNGramLM


def corpus_acc(preds, gts):
    cd = cn = wd = wn = 0
    for p, g in zip(preds, gts, strict=True):
        cd += Levenshtein.distance(p, g)
        cn += len(g)
        wd += Levenshtein.distance(p.split(), g.split())
        wn += max(1, len(g.split()))
    return 1 - cd / cn, 1 - wd / wn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kraken", type=Path, required=True)
    ap.add_argument("--trocr", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--lm-dir", type=Path, required=True)
    ap.add_argument("--lam", type=float, default=0.2)  # kraken P1 LM weight
    ap.add_argument("--device-trocr", default="mps")
    a = ap.parse_args()

    crops = sorted(a.val_dir.glob("*.png"))
    gts = {
        c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip() for c in crops
    }
    crops = [c for c in crops if gts[c.stem]]

    def texts_of(cc):
        return [t for c in cc if (t := gts_lm.get(c.stem, "").strip())]

    lm_crops = sorted(a.lm_dir.glob("*.png"))
    gts_lm = {
        c.stem: c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        for c in lm_crops
    }
    lm = CharNGramLM(order=6).train([t for t in gts_lm.values() if t])

    # ---- kraken P1 predictions ----
    from kraken.lib import models
    from kraken.lib.dataset import ImageInputTransforms

    from src.ocr.kraken_lm import label_to_char, line_candidates, rescore
    from src.ocr.transcribe_line_crops import _synthesised_seg

    net = models.load_any(str(a.kraken), device="cpu")
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    im0, seg0 = _synthesised_seg(crops[0])
    _ = line_candidates(net, ts, [""] * 400, im0, seg0, 5)  # warm net.outputs shape
    om0 = np.asarray(net.outputs)
    l2c = label_to_char(net.codec, int(om0.shape[1] if om0.ndim == 3 else om0.shape[0]))
    krak = {}
    for crop in crops:
        try:
            cands = line_candidates(net, ts, l2c, *_synthesised_seg(crop), 5)
            krak[crop.stem] = rescore(cands, lm, a.lam, 8, 5)
        except Exception:  # noqa: BLE001
            krak[crop.stem] = ""
    print(f"kraken P1 done ({len(krak)} lines)")

    # ---- TrOCR 1-best predictions ----
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    from src.ocr.image_prep import prepare_image

    resize = (
        (a.trocr / "resize_mode.txt").read_text().strip()
        if (a.trocr / "resize_mode.txt").is_file()
        else "pad"
    )
    model = VisionEncoderDecoderModel.from_pretrained(a.trocr).to(a.device_trocr).eval()
    proc = AutoImageProcessor.from_pretrained(a.trocr)
    tok = AutoTokenizer.from_pretrained(a.trocr)
    tro = {}
    for crop in crops:
        img = prepare_image(Image.open(crop).convert("RGB"), proc, resize)
        pv = proc(images=img, return_tensors="pt").pixel_values.to(a.device_trocr)
        with torch.no_grad():
            out = model.generate(pixel_values=pv, num_beams=4, max_length=128)
        tro[crop.stem] = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    print(f"TrOCR 1-best done ({len(tro)} lines)")

    # ---- aggregate ----
    stems = [c.stem for c in crops]
    G = [gts[s] for s in stems]
    K = [krak[s] for s in stems]
    T = [tro[s] for s in stems]
    # oracle: per line pick the pred with smaller char edit distance to GT
    ORC = [
        k if Levenshtein.distance(k, g) <= Levenshtein.distance(t, g) else t
        for k, t, g in zip(K, T, G, strict=True)
    ]
    # realizable: pick the hypothesis the char-LM scores higher (per-char), no GT
    A = [
        k if lm.logscore(k, per_char=True) >= lm.logscore(t, per_char=True) else t
        for k, t in zip(K, T, strict=True)
    ]
    agree = sum(1 for k, t in zip(K, T, strict=True) if k == t)
    print(f"\nlines={len(stems)}  kraken==trocr on {agree} ({agree/len(stems):.0%})")
    print(f"{'system':<26} | {'char':>7} | {'word':>7}")
    for name, P in [
        (f"kraken P1 (λ={a.lam:.1f})", K),
        ("TrOCR 1-best", T),
        ("ORACLE best-of-both", ORC),
        ("LM-arbitrated (realizable)", A),
    ]:
        ca, wa = corpus_acc(P, G)
        print(f"{name:<26} | {ca:>7.4f} | {wa:>7.4f}")


if __name__ == "__main__":
    main()
