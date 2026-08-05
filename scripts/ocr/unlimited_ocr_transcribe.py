"""Unlimited-OCR (Baidu VLM, DeepSeek-OCR base) over line crops → <stem>.txt.

Recognition of pre-cropped single lines for the 300-val comparison. VLM, needs a
CUDA GPU (bf16). The model exposes a custom `model.infer(...)` (trust_remote_code);
its exact signature/return is pinned during a small probe run (--limit) before the
full set. Prompt is configurable — for single lines a plain OCR prompt beats the
repo's "document parsing." default.

Usage (probe then full):
    python scripts/ocr/unlimited_ocr_transcribe.py --limit 15 \
        --input-dir <crops> --out-dir <preds> --prompt '<image>\nFree OCR.'
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def extract_text(res) -> str:
    """The custom infer() may return a str, a dict, or a list — normalise."""
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in ("text", "rec_text", "result", "content"):
            if k in res:
                return str(res[k])
    if isinstance(res, list | tuple) and res:
        return extract_text(res[0])
    return str(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--model-id", default="baidu/Unlimited-OCR")
    ap.add_argument("--prompt", default="<image>\nFree OCR.")
    ap.add_argument("--limit", type=int, default=0, help="probe: only first N crops")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .cuda()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops = sorted(glob.glob(str(args.input_dir / "*.png")))
    if args.limit:
        crops = crops[: args.limit]

    for crop in crops:
        stem = Path(crop).stem
        res = model.infer(
            tok,
            prompt=args.prompt,
            image_file=crop,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=8192,
        )
        text = extract_text(res).strip().replace("\n", " ")
        (args.out_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
        if args.limit:
            print(f"{stem}: {text!r}")
    print(f"transcribed {len(crops)} crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
