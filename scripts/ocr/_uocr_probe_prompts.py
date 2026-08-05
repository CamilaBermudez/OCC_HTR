"""Throwaway: find an Unlimited-OCR prompt/crop_mode combo that yields text."""

import glob
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

mid = "baidu/Unlimited-OCR"
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(
        mid, trust_remote_code=True, use_safetensors=True, dtype=torch.bfloat16
    )
    .eval()
    .cuda()
)
work = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/uocr_work")
work.mkdir(parents=True, exist_ok=True)

crops = sorted(glob.glob("data/processed/annotated_samples/OCR/validation/*.png"))[:2]
prompts = [
    "<image>document parsing.",
    "<image>\nFree OCR.",
    "<image>\n<|grounding|>OCR this image.",
    "<image>\nOCR:",
    "<image>",
]
for crop in crops:
    print("\n#### CROP", Path(crop).name)
    for cm in (True, False):
        for pr in prompts:
            try:
                out = model.infer(
                    tok,
                    prompt=pr,
                    image_file=crop,
                    output_path=str(work),
                    eval_mode=True,
                    base_size=1024,
                    image_size=640,
                    crop_mode=cm,
                    max_length=4096,
                )
                txt = (out if isinstance(out, str) else str(out)).strip().replace("\n", " ")
            except Exception as e:  # noqa: BLE001
                txt = f"<ERR {type(e).__name__}: {str(e)[:60]}>"
            print(f"  crop_mode={cm} prompt={pr!r:45} -> {txt[:90]!r}")
