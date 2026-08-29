"""Does an external corpus help the kraken char-LM reranker? (spec §6.13 follow-up)

Honest sweep: train the order-6 char-LM on {annotated, +medical, +COMETA, +Pansier,
COMETA-alone, Pansier-alone}, tune lambda on a 100-line dev split (LM never trained
on it), evaluate on the 300-val at that fixed lambda. Per-position CTC candidates
are computed ONCE from the 0.9710 model and reused across all configs.

    PROJECT_ROOT=. uv run python scripts/ocr/kraken_lm_corpus_sweep.py \
        --model models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --annotated-dir data/processed/annotated_samples/OCR/full_annotated
"""

from __future__ import annotations

import argparse
import json
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

_MED = "data/processed/synthetic_seeds/categorize_20260625_143327/medical_texts_categorized.json"
_COM = "data/processed/synthetic_seeds/categorize_20260625_142702/COMETA_medieval_corpus_categorized.json"
_PAN = "data/processed/synthetic_seeds/pansier_20260820/pansier_categorized.json"


def corpus_texts(path: str) -> list[str]:
    d = json.load(open(path, encoding="utf-8"))
    s = d.get("samples", d)
    vals = s.values() if isinstance(s, dict) else s
    out = []
    for v in vals:
        t = (v.get("text") if isinstance(v, dict) else v) or ""
        if t.strip():
            out.append(t.strip())
    return out


def gt_of(crop: Path) -> str:
    return crop.with_name(crop.stem + ".gt.txt").read_text(encoding="utf-8").strip()


def acc(cands_by_crop, gts, lm, lam, beam, topk):
    cd = cn = wd = wn = 0
    for crop, cands in cands_by_crop.items():
        gt = gts[crop]
        pred = rescore(cands, lm, lam, beam, topk)
        nc, nw = max(1, len(gt)), max(1, len(gt.split()))
        # clip per line: over-production can't push CER/WER above 1 (accuracy below 0)
        cd += min(Levenshtein.distance(pred, gt), nc)
        cn += nc
        wd += min(Levenshtein.distance(pred.split(), gt.split()), nw)
        wn += nw
    return 1 - cd / cn, 1 - wd / wn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--annotated-dir", type=Path, required=True)
    ap.add_argument("--lambdas", default="0,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    lambdas = [float(x) for x in a.lambdas.split(",")]

    crops600 = sorted(a.annotated_dir.glob("*.png"))
    rng = random.Random(42)
    rng.shuffle(crops600)
    dev, train = crops600[: a.n_dev], crops600[a.n_dev :]
    val = sorted(a.val_dir.glob("*.png"))

    ann_train = [gt_of(c) for c in train if gt_of(c)]
    ann_full = [gt_of(c) for c in crops600 if gt_of(c)]
    med, com, pan = corpus_texts(_MED), corpus_texts(_COM), corpus_texts(_PAN)
    print(
        f"corpora: annotated(train={len(ann_train)}, full={len(ann_full)})  "
        f"medical={len(med)} cometa={len(com)} pansier={len(pan)}"
    )

    # candidates ONCE
    net = models.load_any(str(a.model), device=a.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    im0, seg0 = _synthesised_seg(dev[0])
    box0, _ = next(extract_polygons(im0, seg0, legacy=net.nn.use_legacy_polygons))
    net.predict(ts(box0).unsqueeze(0))
    om0 = np.asarray(net.outputs)
    l2c = label_to_char(net.codec, int(om0.shape[1] if om0.ndim == 3 else om0.shape[0]))

    def cache(crops):
        out, gts = {}, {}
        for cr in crops:
            if not gt_of(cr):
                continue
            im, seg = _synthesised_seg(cr)
            try:
                out[cr] = line_candidates(net, ts, l2c, im, seg, a.topk)
                gts[cr] = gt_of(cr)
            except Exception:
                continue
        return out, gts

    print("caching candidates (dev + val) ...")
    dev_c, dev_gt = cache(dev)
    val_c, val_gt = cache(val)
    print(f"cached dev={len(dev_c)} val={len(val_c)}")

    configs = [
        ("annotated", ann_train, ann_full),
        ("+medical", ann_train + med, ann_full + med),
        ("+cometa", ann_train + com, ann_full + com),
        ("+pansier", ann_train + pan, ann_full + pan),
        ("cometa-alone", com, com),
        ("pansier-alone", pan, pan),
    ]

    print(
        f"\n{'config':>14}{'lam*':>6}{'devW':>8}{'valChar':>9}{'valWord':>9}"
        f"{'dChar':>8}{'dWord':>8}"
    )
    base = None
    for name, tune_txt, full_txt in configs:
        lm_tune = CharNGramLM(order=6).train(tune_txt)
        best = (0.0, -1.0)
        for lam in lambdas:
            _, wa = acc(dev_c, dev_gt, lm_tune, lam, a.beam, a.topk)
            if wa > best[1]:
                best = (lam, wa)
        lam_star, devw = best
        lm_full = CharNGramLM(order=6).train(full_txt)
        vc, vw = acc(val_c, val_gt, lm_full, lam_star, a.beam, a.topk)
        if base is None:
            base = (vc, vw)
        print(
            f"{name:>14}{lam_star:>6.1f}{devw:>8.4f}{vc:>9.4f}{vw:>9.4f}"
            f"{100*(vc-base[0]):>+7.2f}{100*(vw-base[1]):>+7.2f}"
        )


if __name__ == "__main__":
    main()
