"""Kraken CTC prefix-beam-search + char-LM shallow fusion (spec §6.13, approach B5).

The per-position rescorer (`kraken_lm_tune.py`) only re-ranks the top-k *substitutions*
at each greedy-decoded character — it cannot add or drop a character, so it never fixes
insertion/deletion errors. This driver instead runs a CTC prefix beam search over the raw
per-FRAME posteriors, fusing the char n-gram LM as each new character is appended. Because
it decodes frames (not fixed greedy positions), a hypothesis can emit a glyph where greedy
read blank (fixes deletions) or blank where greedy read a glyph (fixes insertions).

Honest protocol (same as the N-best rescorer): tune the LM weight alpha on a ViT/CTC-unseen
dev split, then report the 300-val at that fixed alpha with the LM retrained on all real-
non-val. alpha=0 recovers plain CTC beam search (no LM) as the baseline.

    PROJECT_ROOT=. uv run python scripts/ocr/kraken_ctc_lm.py \
        --model models/ocr/kraken/<catmus>.mlmodel \
        --val-dir  data/processed/annotated_samples/OCR/validation \
        --lm-dir   data/processed/annotated_samples/OCR/full_annotated \
        --dev-dir  data/processed/annotated_samples/OCR/dev100_20260815 \
        --alphas 0,0.1,0.2,0.3,0.5,0.8 --beam 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from kraken.lib import models
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import extract_polygons
from rapidfuzz.distance import Levenshtein

from src.ocr.char_lm import CharNGramLM
from src.ocr.kraken_lm import ctc_beam_search, label_to_char
from src.ocr.transcribe_line_crops import _synthesised_seg


def frame_logp(net, ts, im, seg):
    """[T, V] log-probs for one line crop (kraken outputs are probabilities)."""
    box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
    net.predict(ts(box).unsqueeze(0))
    om = np.asarray(net.outputs)
    om = om[0] if om.ndim == 3 else om  # [labels, frames]
    return np.log(np.clip(om, 1e-9, None)).T  # [frames, labels]


def build_cache(net, ts, crops):
    """Precompute (gt, per-frame logp) once per line — kraken predict is the costly part."""
    cache = []
    for crop in crops:
        gt = crop.with_name(crop.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        if not gt:
            continue
        try:
            logp = frame_logp(net, ts, *_synthesised_seg(crop))
        except Exception:  # noqa: BLE001
            continue
        cache.append((gt, logp))
    return cache


def acc(cache, l2c, blank_set, lm, alpha, beam):
    cd = cn = wd = wn = 0
    for gt, logp in cache:
        pred = ctc_beam_search(logp, l2c, blank_set, lm, alpha, beam)
        nc, nw = max(1, len(gt)), max(1, len(gt.split()))
        # clip per line: over-production can't push CER/WER above 1 (accuracy below 0)
        cd += min(Levenshtein.distance(pred, gt), nc)
        cn += nc
        wd += min(Levenshtein.distance(pred.split(), gt.split()), nw)
        wn += nw
    return 1 - cd / cn, 1 - wd / wn


def texts_of(cc):
    out = []
    for c in cc:
        t = c.with_name(c.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        if t:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument(
        "--lm-dir", type=Path, required=True, help="real lines for LM-train (ViT-unseen split)"
    )
    ap.add_argument("--dev-dir", type=Path, required=True, help="held-out lines to tune alpha")
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.5,0.8")
    ap.add_argument("--beam", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    dev = sorted(a.dev_dir.glob("*.png"))
    val = sorted(a.val_dir.glob("*.png"))
    lm_train = CharNGramLM(order=6).train(texts_of(sorted(a.lm_dir.glob("*.png"))))
    lm_full = CharNGramLM(order=6).train(texts_of(sorted(a.lm_dir.glob("*.png"))) + texts_of(dev))

    net = models.load_any(str(a.model), device=a.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    logp0 = frame_logp(net, ts, *_synthesised_seg(dev[0]))
    l2c = label_to_char(net.codec, logp0.shape[1])
    blank_set = {i for i, ch in enumerate(l2c) if not ch}
    print(
        f"labels={len(l2c)} blank_labels={len(blank_set)} dev={len(dev)} test={len(val)} beam={a.beam}"
    )

    print("decoding per-frame posteriors (dev+val, once) ...")
    dev_cache = build_cache(net, ts, dev)
    val_cache = build_cache(net, ts, val)

    print(f"{'alpha':>6} | {'DEV char':>8} | {'DEV word':>8}")
    best = (0.0, -1.0)
    for al in alphas:
        ca, wa = acc(dev_cache, l2c, blank_set, lm_train, al, a.beam)
        print(f"{al:>6} | {ca:>8.4f} | {wa:>8.4f}")
        if wa > best[1]:
            best = (al, wa)
    al_star = best[0]
    print(f"\n>>> alpha* = {al_star} (best DEV word_acc)  [dev is model-unseen — honest]")

    c0, w0 = acc(val_cache, l2c, blank_set, lm_full, 0.0, a.beam)
    cs, ws = acc(val_cache, l2c, blank_set, lm_full, al_star, a.beam)
    print(f"\n300-VAL (HONEST, fixed alpha*={al_star}):")
    print(f"  CTC beam (alpha=0):      char {c0:.4f}  word {w0:.4f}")
    print(
        f"  + char-LM (alpha={al_star}):  char {cs:.4f}  word {ws:.4f}   (dchar {cs-c0:+.4f}, dword {ws-w0:+.4f})"
    )


if __name__ == "__main__":
    main()
