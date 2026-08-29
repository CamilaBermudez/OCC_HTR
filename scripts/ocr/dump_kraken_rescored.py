"""Dump per-line kraken predictions with and without the char-LM rescore (spec §6.13).

The confidence router's kraken branch should emit the *deployed* text (CTC + char-LM rescore,
0.9743), not raw CTC (0.9710) — otherwise a router is unfairly handicapped against the 0.9743
leader. This reuses the rescoring lattice from `kraken_lm_rescore.py` to write, per line, the raw
1-best and the λ-rescored 1-best so the router can swap in the rescored text on kraken-picked lines.

    PROJECT_ROOT=. uv run python scripts/ocr/dump_kraken_rescored.py \
        --model models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --lm data/processed/lm_corpora/lm_600.pkl --lam 0.2 \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --out tests/ocr/evaluations/longtail_confidence/kraken_rescored_val.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from kraken.lib import models
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import extract_polygons
from rapidfuzz.distance import Levenshtein

from src.ocr.char_lm import CharNGramLM
from src.ocr.transcribe_line_crops import _synthesised_seg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kraken_lm_rescore import label_to_char, line_candidates, rescore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--lm", type=Path, required=True)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    lm = CharNGramLM.load(a.lm)
    net = models.load_any(str(a.model), device=a.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    crops = sorted(a.val_dir.glob("*.png"))

    im0, seg0 = _synthesised_seg(crops[0])
    box0, _ = next(extract_polygons(im0, seg0, legacy=net.nn.use_legacy_polygons))
    net.predict(ts(box0).unsqueeze(0))
    om0 = np.asarray(net.outputs)
    n_labels = int(om0.shape[1] if om0.ndim == 3 else om0.shape[0])
    l2c = label_to_char(net.codec, n_labels)

    rows, craw = [], 0
    crescore = cn = 0
    for crop in crops:
        gt = crop.with_name(crop.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        if not gt:
            continue
        im, seg = _synthesised_seg(crop)
        try:
            cands = line_candidates(net, ts, l2c, im, seg, a.topk)
        except Exception:  # noqa: BLE001
            continue
        raw = rescore(cands, None, 0.0, a.beam, a.topk)
        resc = rescore(cands, lm, a.lam, a.beam, a.topk)
        rows.append((crop.stem, raw, resc, gt))
        craw += Levenshtein.distance(raw, gt)
        crescore += Levenshtein.distance(resc, gt)
        cn += len(gt)

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["stem", "pred_raw", "pred_rescored", "gt"])
        wr.writerows(rows)
    print(f"{len(rows)} lines -> {a.out}")
    print(f"corpus char acc: raw {1 - craw / cn:.4f}   rescored(λ={a.lam}) {1 - crescore / cn:.4f}")


if __name__ == "__main__":
    main()
