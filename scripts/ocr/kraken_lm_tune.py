"""Honest lambda-tuning for the kraken LM rescorer (spec §6.13): pick lambda on a
held-out dev split (LM never trained on it), then report the 300-val at that fixed
lambda. Avoids the tuned-on-test bias of the first sweep.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from kraken.lib import models
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import extract_polygons
from rapidfuzz.distance import Levenshtein

from src.ocr.char_lm import CharNGramLM
from src.ocr.kraken_lm import label_to_char, line_candidates, rescore
from src.ocr.transcribe_line_crops import _synthesised_seg


def acc(net, ts, l2c, crops, lm, lam, beam, topk):
    cd = cn = wd = wn = 0
    for crop in crops:
        gt = crop.with_name(crop.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        if not gt:
            continue
        im, seg = _synthesised_seg(crop)
        try:
            cands = line_candidates(net, ts, l2c, im, seg, topk)
        except Exception:
            continue
        pred = rescore(cands, lm, lam, beam, topk)
        nc, nw = max(1, len(gt)), max(1, len(gt.split()))
        # clip per line: over-production can't push CER/WER above 1 (accuracy below 0)
        cd += min(Levenshtein.distance(pred, gt), nc)
        cn += nc
        wd += min(Levenshtein.distance(pred.split(), gt.split()), nw)
        wn += nw
    return 1 - cd / cn, 1 - wd / wn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--annotated-dir", type=Path, required=True)  # the 600 (train+dev source)
    ap.add_argument("--lambdas", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    lambdas = [float(x) for x in a.lambdas.split(",")]
    # split the 600 crops -> LM-train / dev
    crops600 = sorted(a.annotated_dir.glob("*.png"))
    rng = random.Random(42)
    rng.shuffle(crops600)
    dev = crops600[: a.n_dev]
    train = crops600[a.n_dev :]

    def texts(cc):
        return [
            c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip()
            for c in cc
            if c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        ]

    lm_train = CharNGramLM(order=6).train(texts(train))  # trained on 500, has NOT seen dev
    lm_full = CharNGramLM(order=6).train(texts(crops600))  # deployable, all 600
    net = models.load_any(str(a.model), device=a.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    im0, seg0 = _synthesised_seg(dev[0])
    box0, _ = next(extract_polygons(im0, seg0, legacy=net.nn.use_legacy_polygons))
    net.predict(ts(box0).unsqueeze(0))
    om0 = np.asarray(net.outputs)
    l2c = label_to_char(net.codec, int(om0.shape[1] if om0.ndim == 3 else om0.shape[0]))
    print(
        f"dev={len(dev)} lines (LM-train={len(train)}), test={len(sorted(a.val_dir.glob('*.png')))}"
    )
    print(f"{'lambda':>7} | {'DEV char':>8} | {'DEV word':>8}")
    best = (None, -1)
    for lam in lambdas:
        ca, wa = acc(net, ts, l2c, dev, lm_train, lam, a.beam, a.topk)
        print(f"{lam:>7} | {ca:>8.4f} | {wa:>8.4f}")
        if wa > best[1]:
            best = (lam, wa)
    lam_star = best[0]
    print(f"\n>>> lambda* = {lam_star} (best DEV word_acc)")
    valc = sorted(a.val_dir.glob("*.png"))
    c0, w0 = acc(net, ts, l2c, valc, lm_full, 0.0, a.beam, a.topk)
    cs, ws = acc(net, ts, l2c, valc, lm_full, lam_star, a.beam, a.topk)
    print(f"\n300-VAL (LM=600, fixed lambda*={lam_star}):")
    print(f"  baseline (lambda=0): char {c0:.4f}  word {w0:.4f}")
    print(
        f"  rescored (lambda={lam_star}): char {cs:.4f}  word {ws:.4f}   (dchar {cs-c0:+.4f}, dword {ws-w0:+.4f})"
    )


if __name__ == "__main__":
    main()
