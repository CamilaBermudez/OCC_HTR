"""Recognition-only PaddleOCR over pre-cropped line images (300-val).

Runs the PP-OCRv5 Latin text-recognition model on each <stem>.png line crop and
writes <stem>.txt — recognition-ONLY (no re-detection), matching how catmus /
TrOCR read the whole crop, for a fair 300-val comparison. Requires a paddle env
(paddlepaddle + paddleocr 3.x), separate from the project torch env.

Usage:
    <paddle-venv>/bin/python scripts/ocr/paddleocr_recognize.py \
        --input-dir data/processed/annotated_samples/OCR/validation \
        --out-dir data/processed/transcription/paddleocr_latin_val300
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from paddleocr import TextRecognition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-name", default="latin_PP-OCRv5_mobile_rec")
    args = ap.parse_args()

    rec = TextRecognition(model_name=args.model_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    crops = sorted(glob.glob(str(args.input_dir / "*.png")))
    n = 0
    for crop in crops:
        stem = Path(crop).stem
        out = rec.predict(crop)
        r = out[0] if isinstance(out, list) else out
        text = (r.get("rec_text") or "").strip()
        (args.out_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        n += 1
    print(f"recognized {n} crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
