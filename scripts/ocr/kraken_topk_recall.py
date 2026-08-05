"""Top-k recall for a kraken/CTC recogniser (catmus) — the CTC analog of §6.8.

TrOCR (§6.8) is autoregressive: teacher-force the GT and read the ranked next-token
distribution at each position. Kraken is CTC — no autoregression — so the analog is
per-CHARACTER, per-FRAME: for each predicted character, read the top-k alphabet
symbols at its peak output frame (`net.outputs`), align the predicted string to the
GT (char-level Levenshtein), and for each SUBSTITUTION error ask "was the correct
character in kraken's top-k at that frame?". Insertions/deletions (segmentation-level
errors) have no clean per-frame top-k and are reported separately.

Interpretation: high top-k recall among substitutions = errors are same-position
confusions a reranker/LM could fix; low = the information isn't there.

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/kraken_topk_recall.py \
        --model models/ocr/catmus-medieval.mlmodel \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --output tests/ocr/evaluations/kraken_topk/catmus_topk.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from kraken.lib import models
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import extract_polygons
from rapidfuzz.distance import Levenshtein

from src.ocr.transcribe_line_crops import _synthesised_seg

K_LIST = [1, 2, 3, 5, 10]


def label_to_char(codec, n_labels: int) -> list[str]:
    """Precompute label-index -> character ('' marks the CTC blank / no glyph)."""
    out = []
    for i in range(n_labels):
        try:
            dec = codec.decode([(i, 0, 0, 1.0)])
            out.append(dec[0][0] if dec else "")
        except Exception:  # noqa: BLE001
            out.append("")
    return out


def line_topk(net, ts, l2c: list[str], im, seg, maxk: int):
    """Return (pred_string, [set-of-topk-chars per predicted char])."""
    box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
    preds = net.predict(ts(box).unsqueeze(0))[0]  # [(char, start, end, conf)]
    om = np.asarray(net.outputs)
    om = om[0] if om.ndim == 3 else om  # [labels, frames]
    pred_chars, topk_sets = [], []
    for ch, s, e, _ in preds:
        pred_chars.append(ch)
        e = max(e, s)
        span = om[:, s : e + 1]
        peak = s + int(span.max(axis=0).argmax())
        idxs = om[:, peak].argsort()[::-1][:maxk]
        topk_sets.append([l2c[int(i)] for i in idxs])
    return "".join(pred_chars), topk_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/ocr/catmus-medieval.mlmodel"))
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    net = models.load_any(str(args.model), device=args.device)
    b, c, h, w = net.nn.input
    # valid_norm=False matches kraken.rpred for catmus (a binarized '1' model);
    # valid_norm=True gives a different, noisier normalisation (verified: False
    # reproduces rpred's prediction string exactly).
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)

    crops = sorted(args.val_dir.glob("*.png"))
    if args.limit:
        crops = crops[: args.limit]

    # warm-up predict to populate net.outputs, then size the label->char table
    im0, seg0 = _synthesised_seg(crops[0])
    box0, _ = next(extract_polygons(im0, seg0, legacy=net.nn.use_legacy_polygons))
    net.predict(ts(box0).unsqueeze(0))
    om0 = np.asarray(net.outputs)
    n_labels = int(om0.shape[1] if om0.ndim == 3 else om0.shape[0])
    l2c = label_to_char(net.codec, n_labels)

    # counters
    gt_chars = 0
    ops = Counter()  # replace/insert/delete
    sub_in_topk = {k: 0 for k in K_LIST}  # substitutions with GT char in pred top-k
    n_sub = 0

    for crop in crops:
        gt = crop.with_name(crop.stem + ".gt.txt").read_text(encoding="utf-8").strip()
        im, seg = _synthesised_seg(crop)
        try:
            pred, topk_sets = line_topk(net, ts, l2c, im, seg, max(K_LIST))
        except Exception:  # noqa: BLE001
            continue
        gt_chars += len(gt)
        for op, i, j in Levenshtein.editops(pred, gt).as_list():
            if op == "replace":
                ops["replace"] += 1
                n_sub += 1
                cand = topk_sets[i] if i < len(topk_sets) else []
                for k in K_LIST:
                    if gt[j] in cand[:k]:
                        sub_in_topk[k] += 1
            elif op == "insert":
                ops["insert"] += 1  # GT char missing from pred (deletion by model)
            elif op == "delete":
                ops["delete"] += 1  # pred char with no GT (insertion by model)

    # matches = GT chars correctly produced; framed = GT positions that HAVE a
    # prediction frame (matches + substitutions). Deletions (68) have no frame.
    subs = ops["replace"]
    matches = gt_chars - subs - ops["insert"]  # ops[insert] = model deletions
    framed = matches + subs
    cer = round((subs + ops["insert"] + ops["delete"]) / gt_chars, 4) if gt_chars else 0
    result = {
        "model": str(args.model),
        "level": "character (CTC) — NOT token-level; compare err->top-k structure only",
        "n_lines": len(crops),
        "gt_chars": gt_chars,
        "cer_from_alignment": cer,
        "errors": {
            "substitutions": subs,
            "model_deletions": ops["insert"],
            "model_insertions": ops["delete"],
        },
        # top-1 char acc + top-k recall over framed positions (matches+subs)
        "top1_char": round(matches / framed, 4) if framed else 0,
        "topk_recall_all_framed": {
            f"top{k}": round((matches + sub_in_topk[k]) / framed, 4) if framed else 0
            for k in K_LIST
        },
        # err->top-k: among substitution errors, GT char in top-k
        "err_topk": {f"top{k}": round(sub_in_topk[k] / n_sub, 4) if n_sub else 0 for k in K_LIST},
        "note": "err->top-k over substitution errors only; the ~49% of errors that "
        "are CTC ins/del have no per-frame top-k.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
