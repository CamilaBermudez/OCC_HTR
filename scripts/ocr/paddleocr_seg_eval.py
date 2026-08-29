"""Does PaddleOCR line-detection give better crops than our kraken segmentation?

Downstream test (spec §6.9): for each validation line, crop it from the RAW page
two ways — with our segmentation box vs the spatially-matched PaddleOCR detection
box — run the *same* ViT+RoBERTa recogniser on both, and compare CER, first-word
accuracy (the line-initial-garble hypothesis, §6.7.1), and detection recall (how
many of our lines PaddleOCR actually found — the "missed lines" concern).

Both crops are raw rectangles from the same page image, so the ONLY variable is
the detection box geometry. Needs the PaddleOCR boxes precomputed by
`paddle_detect_pages.py` (run in the isolated paddle env).

    PROJECT_ROOT=. uv run python scripts/ocr/paddleocr_seg_eval.py \
        --paddle-boxes <paddle_boxes.json> \
        --seg-dir data/processed/segmented_images/segmentation_20260618_111517 \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --model-dir models/ocr/finetuned/vitroberta_T1_1font_20260731/best_model \
        --out tests/ocr/paddleocr_smoke_20260803/seg_eval.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

from src.ocr.image_prep import prepare_image

_LINE_RE = re.compile(r"(?P<page>.+)_line_(?P<idx>\d+)\.png$")


def bbox_of(poly) -> tuple[int, int, int, int]:
    a = np.array(poly).reshape(-1, 2)
    return int(a[:, 0].min()), int(a[:, 1].min()), int(a[:, 0].max()), int(a[:, 1].max())


def match_paddle(our, boxes):
    """Best PaddleOCR box for our line by IoU (so a neighbouring box can't win)."""
    ox0, oy0, ox1, oy1 = our
    oa = max(1, (ox1 - ox0) * (oy1 - oy0))
    best, best_iou = None, 0.0
    for pb in boxes:
        px0, py0, px1, py1 = bbox_of(pb)
        xo = max(0, min(ox1, px1) - max(ox0, px0))
        yo = max(0, min(oy1, py1) - max(oy0, py0))
        if xo <= 0 or yo <= 0:
            continue
        inter = xo * yo
        union = oa + (px1 - px0) * (py1 - py0) - inter
        iou = inter / max(1, union)
        if iou > best_iou:
            best, best_iou = (px0, py0, px1, py1), iou
    return best, best_iou


def cer(ref, hyp):
    return min(1.0, Levenshtein.distance(ref, hyp) / max(1, len(ref)))


def first_word(s):
    p = s.split()
    return p[0].lower() if p else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paddle-boxes", type=Path, required=True)
    ap.add_argument("--seg-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--iou-thresh",
        type=float,
        default=0.4,
        help="min IoU for a clean 1:1 line match (CER compared only on these).",
    )
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    dev = args.device
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(dev).eval()
    improc = AutoImageProcessor.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    rmode = "pad"
    mf = args.model_dir / "resize_mode.txt"
    if mf.is_file():
        rmode = mf.read_text().strip() or "pad"

    def read(crop: Image.Image) -> str:
        img = prepare_image(crop, improc, rmode)
        pv = improc(images=img, return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad():
            ids = model.generate(pv, num_beams=4, max_length=128)
        return tok.decode(ids[0], skip_special_tokens=True).strip()

    paddle = json.loads(args.paddle_boxes.read_text())
    pages = set(paddle.keys())
    seg_cache: dict[str, list] = {}
    raw_cache: dict[str, Image.Image] = {}

    rows = []
    for png in sorted(glob.glob(str(args.val_dir / "*.png"))):
        m = _LINE_RE.search(os.path.basename(png))
        if not m:
            continue
        page, idx = m.group("page"), int(m.group("idx"))
        if page not in pages:
            continue
        gt = Path(png[:-4] + ".gt.txt").read_text(encoding="utf-8").strip()
        if not gt:
            continue
        if page not in seg_cache:
            seg_cache[page] = json.loads((args.seg_dir / f"{page}.json").read_text())["lines"]
        our = bbox_of(seg_cache[page][idx]["boundary"])
        pmatch, iou = match_paddle(our, paddle[page]["boxes"])
        if page not in raw_cache:
            raw_cache[page] = Image.open(paddle[page]["raw"]).convert("RGB")
        raw = raw_cache[page]
        our_txt = read(raw.crop(our))
        row = {
            "page": page,
            "idx": idx,
            "gt": gt,
            "detected": pmatch is not None,
            "iou": round(iou, 2),
            "clean": iou >= args.iou_thresh,
            "our_cer": round(cer(gt, our_txt), 3),
            "our_txt": our_txt,
            "our_fw_ok": first_word(our_txt) == first_word(gt),
        }
        if pmatch is not None:
            pad_txt = read(raw.crop(pmatch))
            row.update(
                paddle_cer=round(cer(gt, pad_txt), 3),
                paddle_txt=pad_txt,
                paddle_fw_ok=first_word(pad_txt) == first_word(gt),
            )
        rows.append(row)

    n = len(rows)
    detected = [r for r in rows if r["detected"]]
    clean = [r for r in rows if r["clean"]]  # trustworthy 1:1 correspondence only

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    summary = {
        "n_lines": n,
        "detection_recall": round(len(detected) / n, 3) if n else 0,
        "n_clean_matches": len(clean),
        "iou_thresh": args.iou_thresh,
        "note": "CER/first-word compared ONLY on clean matches (IoU>=thresh) to avoid "
        "line-mismatch noise; both crops are raw rects from the same page.",
        "our_mean_cer_clean": mean([r["our_cer"] for r in clean]),
        "paddle_mean_cer_clean": mean([r["paddle_cer"] for r in clean]),
        "our_firstword_acc_clean": mean([1.0 if r["our_fw_ok"] else 0.0 for r in clean]),
        "paddle_firstword_acc_clean": mean([1.0 if r["paddle_fw_ok"] else 0.0 for r in clean]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1)
    )
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
