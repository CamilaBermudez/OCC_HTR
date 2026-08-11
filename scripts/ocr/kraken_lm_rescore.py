"""Rescore a kraken/CTC recogniser's per-character candidates with a char n-gram LM
(spec §6.13). Tests whether adding a language prior over the recogniser's *own*
alternatives recovers the top-k headroom (§6.12) — especially kraken's word-acc gap
(0.8201 < catmus 0.8512) — WITHOUT a blind lexicon swap (§6.10, settled-negative).

Method (approach-3-lite over predicted positions):
  * For each predicted character, read the top-k alphabet symbols at its peak output
    frame (net.outputs) with their log-probabilities — the recogniser's per-position
    candidate lattice.
  * Left-to-right beam search: expand each beam by the top-k candidates, scoring
    ``visual_logprob + lambda * LM_logcond(char | prefix)``; keep the best B beams.
  * lambda=0 reproduces the greedy 1-best (sanity). Sweep lambda; higher trusts the LM.

Scope caveat: this rescopes only SUBSTITUTIONS (the per-frame-alignable errors, ~46% of
kraken's errors, §6.8). Insertions/deletions have no per-frame candidate here, so they
are untouched — a real CTC prefix-beam-search (KenLM+pyctcdecode) is the upgrade if this
shows signal.

    PROJECT_ROOT=. uv run python scripts/ocr/kraken_lm_rescore.py \
        --model models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --lm data/processed/lm_corpora/lm_600.pkl --lambdas 0,0.2,0.5,1,2,4
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
from src.ocr.transcribe_line_crops import _synthesised_seg


def label_to_char(codec, n_labels: int) -> list[str]:
    out = []
    for i in range(n_labels):
        try:
            dec = codec.decode([(i, 0, 0, 1.0)])
            out.append(dec[0][0] if dec else "")
        except Exception:  # noqa: BLE001
            out.append("")
    return out


def line_candidates(net, ts, l2c, im, seg, topk: int):
    """Return per-predicted-char candidate lists [[(char, logprob), ...], ...]."""
    box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
    preds = net.predict(ts(box).unsqueeze(0))[0]  # [(char, start, end, conf)]
    om = np.asarray(net.outputs)
    om = om[0] if om.ndim == 3 else om  # [labels, frames]
    cands = []
    for _ch, s, e, _ in preds:
        e = max(e, s)
        span = om[:, s : e + 1]
        peak = s + int(span.max(axis=0).argmax())
        col = om[:, peak]
        idxs = col.argsort()[::-1][:topk]
        opts = [(l2c[int(i)], float(np.log(max(col[int(i)], 1e-9)))) for i in idxs if l2c[int(i)]]
        if opts:
            cands.append(opts)
    return cands


def rescore(cands, lm: CharNGramLM | None, lam: float, beam: int, topk: int) -> str:
    beams = [("", 0.0)]
    for pos in cands:
        nxt = []
        for text, sc in beams:
            for ch, vlp in pos[:topk]:
                lm_s = lam * lm.logcond(text, ch) if (lm and lam) else 0.0
                nxt.append((text + ch, sc + vlp + lm_s))
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:beam]
    return beams[0][0] if beams else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--lm", type=Path, required=True)
    ap.add_argument("--lambdas", default="0,0.2,0.5,1,2,4")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lm = CharNGramLM.load(args.lm)
    lambdas = [float(x) for x in args.lambdas.split(",")]

    net = models.load_any(str(args.model), device=args.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    crops = sorted(args.val_dir.glob("*.png"))
    if args.limit:
        crops = crops[: args.limit]

    # warm-up to size the label->char table
    im0, seg0 = _synthesised_seg(crops[0])
    box0, _ = next(extract_polygons(im0, seg0, legacy=net.nn.use_legacy_polygons))
    net.predict(ts(box0).unsqueeze(0))
    om0 = np.asarray(net.outputs)
    n_labels = int(om0.shape[1] if om0.ndim == 3 else om0.shape[0])
    l2c = label_to_char(net.codec, n_labels)

    # cache candidates + GT once
    data = []
    for crop in crops:
        gt = crop.with_name(crop.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        if not gt:
            continue
        im, seg = _synthesised_seg(crop)
        try:
            cands = line_candidates(net, ts, l2c, im, seg, args.topk)
        except Exception:  # noqa: BLE001
            continue
        data.append((gt, cands))

    def score(lam):
        cd = wd = cn = wn = 0
        for gt, cands in data:
            pred = rescore(cands, lm, lam, args.beam, args.topk)
            cd += Levenshtein.distance(pred, gt)
            cn += len(gt)
            wd += Levenshtein.distance(pred.split(), gt.split())
            wn += max(1, len(gt.split()))
        return 1 - cd / cn, 1 - wd / wn

    print(f"LM={args.lm.name}  lines={len(data)}  topk={args.topk}  beam={args.beam}")
    print(f"{'lambda':>7} | {'char_acc':>9} | {'word_acc':>9}")
    for lam in lambdas:
        ca, wa = score(lam)
        print(f"{lam:>7} | {ca:>9.4f} | {wa:>9.4f}")


if __name__ == "__main__":
    main()
