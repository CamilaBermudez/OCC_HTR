"""Dump Medusa 0.2 per-character confidence over a flat folder of line crops, for the
'does the model know when it's wrong?' calibration (spec §6.13). Runs the 9B VLM on the
cluster (CUDA); the light calibration + plotting is done locally from the JSON it writes.

For each image: greedy-generate with per-token log-probs (compute_transition_scores),
expand each token's prob to its decoded characters, then CLEAN exactly as
clean_medusa_output does (keep Medusa's FIRST non-noise line) and slice the per-char
confidences to that line. Writes {stem: {"text": ..., "confs": [...]}} to one JSON.

    PROJECT_ROOT=. python scripts/ocr/medusa_confidence_dump.py \
        --input-dir data/processed/annotated_samples/OCR/validation \
        --out preds/medusa_conf_val300.json --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.ocr.clean_medusa_output import is_noise
from src.ocr.medusa_transcribe import DEFAULT_PROMPT, _load_model_and_processor, detect_device


def _clean_with_confs(text: str, confs: list[float]) -> tuple[str, list[float]]:
    """Medusa's first non-noise line + the per-char confidences for exactly that line."""
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if not is_noise(stripped):
            start = offset + (len(line) - len(line.lstrip()))
            return stripped, confs[start : start + len(stripped)]
        offset += len(line) + 1  # + newline
    return "", []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="ENC-PSL/Medusa0.2Line-9B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = ap.parse_args()

    import logging

    logger = logging.getLogger("medusa_conf")
    logging.basicConfig(level=logging.INFO)
    device = detect_device(a.device)
    processor, model, _ = _load_model_and_processor(a.model_id, device, "none", logger)
    tok = getattr(processor, "tokenizer", processor)

    crops = sorted(a.input_dir.glob("*.png"))
    out: dict[str, dict] = {}
    for i, crop in enumerate(crops):
        img = Image.open(crop).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_PROMPT},
                    {"type": "image", "image": img},
                ],
            }
        ]
        kw = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        try:
            inputs = processor.apply_chat_template([messages], **kw, enable_thinking=False).to(
                device
            )
        except TypeError:
            inputs = processor.apply_chat_template([messages], **kw).to(device)
        with torch.no_grad():
            g = model.generate(
                **inputs,
                max_new_tokens=a.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        trans = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
        gen_ids = g.sequences[0, inputs["input_ids"].shape[1] :]
        probs = trans[0].exp().tolist()
        chars, confs = [], []
        for tid, pr in zip(gen_ids.tolist(), probs, strict=False):
            piece = tok.decode([tid], skip_special_tokens=True)
            for ch in piece:
                chars.append(ch)
                confs.append(float(pr))
        text, cconfs = _clean_with_confs("".join(chars), confs)
        out[crop.stem] = {"text": text, "confs": cconfs}
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(crops)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"DONE {len(out)} lines -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
