"""ViT+RoBERTa (TrOCR) transcription WITH per-token confidence over line crops.

For each ``<page>/<stem>.png`` crop, beam-search generate and record the text plus
the probability of every emitted subword token (via ``compute_transition_scores``,
normalized logits). Writes one JSON per page:

    {"page": "<page>", "lines": {"<stem>": {"text": "...",
                                             "tokens": [["non", 0.983], [" es", 1.0], ...]}}}

Feeds the viewer's model-comparison tab (spec §7.4.1). Run over the SAME kept
filtered crops catmus used, so the two models are compared on identical inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir", required=True, type=Path, help="kept-crops root (per-page subdirs)"
    )
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit-pages", type=int, default=0)
    args = ap.parse_args()

    proc = TrOCRProcessor.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(args.device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # provenance travels with the output (which model produced these confidences)
    (args.out_dir / "_provenance.json").write_text(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "input_dir": str(args.input_dir),
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
            }
        ),
        encoding="utf-8",
    )
    special = set(tok.all_special_ids)

    pages = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    if args.limit_pages:
        pages = pages[: args.limit_pages]

    t0 = time.time()
    n_lines = 0
    for pg in pages:
        out_json = args.out_dir / f"{pg.name}.json"
        if out_json.exists():  # resumable
            n_lines += len(json.loads(out_json.read_text()).get("lines", {}))
            continue
        crops = sorted(pg.glob("*.png"))
        lines: dict[str, dict] = {}
        for i in range(0, len(crops), args.batch_size):
            batch = crops[i : i + args.batch_size]
            imgs = [Image.open(c).convert("RGB") for c in batch]
            pv = proc(images=imgs, return_tensors="pt").pixel_values.to(args.device)
            with torch.no_grad():
                out = model.generate(
                    pv,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            # beam_indices only exists for beam search; greedy (num_beams=1) omits it,
            # and compute_transition_scores accepts None there.
            trans = model.compute_transition_scores(
                out.sequences, out.scores, getattr(out, "beam_indices", None), normalize_logits=True
            )
            for j, c in enumerate(batch):
                gen = out.sequences[j][1:]  # drop decoder-start
                probs = trans[j].exp().tolist()
                toks = [
                    [tok.decode([t]), round(float(p), 4)]
                    for t, p in zip(gen.tolist(), probs, strict=False)
                    if t not in special
                ]
                lines[c.stem] = {
                    "text": tok.decode(gen, skip_special_tokens=True).strip(),
                    "tokens": toks,
                }
        out_json.write_text(
            json.dumps({"page": pg.name, "lines": lines}, ensure_ascii=False), encoding="utf-8"
        )
        n_lines += len(lines)
        print(
            f"{pg.name}: {len(lines)} lines  (total {n_lines}, {time.time() - t0:.0f}s)", flush=True
        )

    print(f"DONE {n_lines} lines across {len(pages)} pages in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
