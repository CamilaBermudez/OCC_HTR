"""Top-k recall of a TrOCR-style model: when the top-1 token is wrong, was the
correct token in the top-k most likely next tokens? (spec §6.8)

Teacher-forced, per decoding position: feed the *correct* prefix (the GT tokens),
read the decoder logits, and at each position compare the model's ranked next-token
distribution to the ground-truth token. This isolates per-token perception — in
free-running generation, once the model diverges the "correct next token" is
ill-defined, so teacher forcing is the right frame for "was the answer in the
top-k".

Reports, over all real content-token positions (special tokens excluded):
  * top-1 accuracy and the top-k recall curve (k = 1,2,3,5,10);
  * THE headline: among top-1 *errors*, the fraction where the GT token was still
    in the top-k — i.e. how many mistakes are recoverable by a reranker/LM vs
    genuine "not even close" perception failures.

Reusable per model:
    PROJECT_ROOT=. uv run python scripts/ocr/topk_recall.py \
        --model-dir models/ocr/finetuned/vitroberta_T1_1font_20260731/best_model \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --device mps --output tests/ocr/evaluations/topk/<name>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.ocr.image_prep import prepare_image

K_LIST = (1, 2, 3, 5, 10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    ap.add_argument("--max-topk", type=int, default=10)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--examples", type=int, default=25, help="error examples to dump")
    ap.add_argument(
        "--resize-mode",
        choices=["pad", "stretch"],
        default=None,
        help="override resize (else resize_mode.txt, else pad). Models trained "
        "with the plain processor default need 'stretch'.",
    )
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    if args.device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        device = args.device

    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    resize_mode = "pad"
    mode_file = args.model_dir / "resize_mode.txt"
    if mode_file.is_file():
        resize_mode = mode_file.read_text(encoding="utf-8").strip() or "pad"
    if args.resize_mode:
        resize_mode = args.resize_mode

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    # replicate the finetune dataset: append eos only if the tokenizer doesn't
    probe = tokenizer("probe").input_ids
    adds_eos = bool(probe) and eos_id is not None and probe[-1] == eos_id
    will_append_eos = (not adds_eos) and (eos_id is not None)
    special = {
        i
        for i in (pad_id, eos_id, bos_id, tokenizer.cls_token_id, tokenizer.sep_token_id)
        if i is not None
    }

    pairs = sorted(args.val_dir.glob("*.png"))
    # per-k: total positions where GT in top-k ; and among-errors in top-k
    n_pos = 0
    n_err = 0
    in_topk = {k: 0 for k in K_LIST}
    err_in_topk = {k: 0 for k in K_LIST}
    examples: list[dict] = []

    for pi, png in enumerate(pairs):
        gt = png.with_suffix("").with_suffix(".gt.txt")
        if not gt.is_file():
            continue
        text = gt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        image = prepare_image(Image.open(png).convert("RGB"), image_processor, resize_mode)
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
        ids = tokenizer(text).input_ids
        if will_append_eos:
            ids = ids + [eos_id]
        labels = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(pixel_values=pixel_values, labels=labels).logits[0]  # [L, V]
        topk = logits.topk(max(K_LIST), dim=-1).indices  # [L, maxk]

        for i, gid in enumerate(ids):
            if gid in special:
                continue
            n_pos += 1
            ranked = topk[i].tolist()
            top1_ok = ranked[0] == gid
            if not top1_ok:
                n_err += 1
            for k in K_LIST:
                hit = gid in ranked[:k]
                in_topk[k] += hit
                if not top1_ok and hit:
                    err_in_topk[k] += hit
            if not top1_ok and len(examples) < args.examples:
                examples.append(
                    {
                        "line": png.stem,
                        "pos": i,
                        "gt_token": tokenizer.decode([gid]).strip(),
                        "pred_top1": tokenizer.decode([ranked[0]]).strip(),
                        "gt_rank": (ranked.index(gid) + 1) if gid in ranked else f">{max(K_LIST)}",
                        "top10": [tokenizer.decode([t]).strip() for t in ranked[:10]],
                    }
                )

    result = {
        "model_dir": str(args.model_dir),
        "n_lines": len(pairs),
        "n_positions": n_pos,
        "n_top1_errors": n_err,
        "top1_acc": round(in_topk[1] / n_pos, 4),
        "topk_recall": {k: round(in_topk[k] / n_pos, 4) for k in K_LIST},
        "error_recovery": {k: round(err_in_topk[k] / n_err, 4) for k in K_LIST if n_err},
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"model: {args.model_dir}")
    print(f"positions {n_pos} | top-1 errors {n_err} ({100*n_err/n_pos:.1f}%)")
    print(f"top-1 token accuracy: {100*result['top1_acc']:.1f}%")
    print("top-k recall (GT token in top-k, all positions):")
    for k in K_LIST:
        print(f"  top-{k:<2} {100*result['topk_recall'][k]:.1f}%")
    print(f"\nHEADLINE — among the {n_err} top-1 ERRORS, GT token was still in:")
    for k in K_LIST[1:]:
        print(f"  top-{k:<2} {100*result['error_recovery'][k]:.1f}%")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
