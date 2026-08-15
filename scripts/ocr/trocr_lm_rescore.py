"""TrOCR N-best + char-LM rescoring (spec §6.13, approach 2).

Autoregressive analog of the kraken rescorer: beam search emits N full-line
hypotheses per image with a sequence score; the char n-gram LM adds a language prior;
we pick argmax of ``ocr_score + lambda * LM_logscore``. Honest protocol: tune lambda on
a held-out dev split (100 of the 600 annotated, LM never trained on it), then report the
300-val at that fixed lambda with the LM retrained on all 600.

    PROJECT_ROOT=. uv run python scripts/ocr/trocr_lm_rescore.py \
        --model-dir models/ocr/finetuned/vitroberta_medical4000_stretch_tok \
        --val-dir data/processed/annotated_samples/OCR/validation \
        --annotated-dir data/processed/annotated_samples/OCR/full_annotated \
        --device mps --nbest 8 --lambdas 0,0.1,0.2,0.3,0.5,0.8,1.2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

from src.ocr.char_lm import CharNGramLM
from src.ocr.image_prep import prepare_image


def gen_nbest(model, proc, tok, path, resize_mode, n, device, max_len):
    """Return [(text, ocr_score)] — N beam hypotheses + length-normalised log-prob."""
    img = prepare_image(Image.open(path).convert("RGB"), proc, resize_mode)
    pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = model.generate(
            pixel_values=pv,
            num_beams=n,
            num_return_sequences=n,
            max_length=max_len,
            output_scores=True,
            return_dict_in_generate=True,
        )
    texts = tok.batch_decode(out.sequences, skip_special_tokens=True)
    scores = out.sequences_scores.tolist()  # length-normalised log-prob per hypothesis
    return list(zip((t.strip() for t in texts), scores, strict=False))


def pick(nbest, lm, lam):
    """argmax over hypotheses of ocr_score + lam * per-char LM logscore."""
    best, bs = nbest[0][0], -1e18
    for text, ocr in nbest:
        s = ocr + (lam * lm.logscore(text, per_char=True) if lam else 0.0)
        if s > bs:
            bs, best = s, text
    return best


def acc(cache, order_keys, lm, lam):
    cd = cn = wd = wn = 0
    for k in order_keys:
        gt, nbest = cache[k]
        pred = pick(nbest, lm, lam)
        cd += Levenshtein.distance(pred, gt)
        cn += len(gt)
        wd += Levenshtein.distance(pred.split(), gt.split())
        wn += max(1, len(gt.split()))
    return 1 - cd / cn, 1 - wd / wn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--annotated-dir", type=Path, required=True)
    ap.add_argument(
        "--dev-dir",
        type=Path,
        default=None,
        help="If given, use this folder as the held-out dev split (ViT-unseen). LM-train is "
        "then ALL of --annotated-dir. Makes lambda* honest (no ViT leakage into dev).",
    )
    ap.add_argument("--lambdas", default="0,0.1,0.2,0.3,0.5,0.8,1.2")
    ap.add_argument("--nbest", type=int, default=8)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    lambdas = [float(x) for x in args.lambdas.split(",")]
    resize_mode = "pad"
    mode_file = args.model_dir / "resize_mode.txt"
    if mode_file.is_file():
        resize_mode = mode_file.read_text(encoding="utf-8").strip() or "pad"

    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(args.device).eval()
    proc = AutoImageProcessor.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    def texts_of(cc):
        out = []
        for c in cc:
            t = c.with_suffix("").with_suffix(".gt.txt").read_text(encoding="utf-8").strip()
            if t:
                out.append(t)
        return out

    crops = sorted(args.annotated_dir.glob("*.png"))
    if args.dev_dir is not None:
        # Honest split: dev is a purpose-built ViT-unseen folder; LM-train = ALL of annotated-dir.
        dev_crops = sorted(args.dev_dir.glob("*.png"))
        train_crops = crops
        leaked = False
        lm_train = CharNGramLM(order=6).train(texts_of(train_crops))
        lm_full = CharNGramLM(order=6).train(texts_of(crops) + texts_of(dev_crops))
    else:
        # Legacy: carve dev out of annotated-dir (leaks into a ViT that trained on all 600).
        rng = random.Random(42)
        rng.shuffle(crops)
        dev_crops, train_crops = crops[: args.n_dev], crops[args.n_dev :]
        leaked = True
        lm_train = CharNGramLM(order=6).train(texts_of(train_crops))
        lm_full = CharNGramLM(order=6).train(texts_of(crops))

    def build_cache(crop_list):
        cache, keys = {}, []
        for c in crop_list:
            gt = c.with_suffix("").with_suffix(".gt.txt").read_text(encoding="utf-8").strip()
            if not gt:
                continue
            nb = gen_nbest(model, proc, tok, c, resize_mode, args.nbest, args.device, args.max_len)
            cache[c.stem] = (gt, nb)
            keys.append(c.stem)
        return cache, keys

    print(f"generating {args.nbest}-best for dev ({len(dev_crops)}) + val ...")
    dev_cache, dev_keys = build_cache(dev_crops)
    val_crops = sorted(args.val_dir.glob("*.png"))
    val_cache, val_keys = build_cache(val_crops)

    print(f"dev={len(dev_keys)} test={len(val_keys)} resize={resize_mode} nbest={args.nbest}")
    print(f"{'lambda':>7} | {'DEV char':>8} | {'DEV word':>8}")
    best = (0.0, -1.0)
    for lam in lambdas:
        ca, wa = acc(dev_cache, dev_keys, lm_train, lam)
        print(f"{lam:>7} | {ca:>8.4f} | {wa:>8.4f}")
        if wa > best[1]:
            best = (lam, wa)
    lam_star = best[0]
    tag = "[NB dev is leaked into the ViT model]" if leaked else "[dev is ViT-unseen — honest]"
    print(f"\n>>> lambda* = {lam_star} (best DEV word_acc)  {tag}")

    if not leaked:
        # Honest headline: fixed lambda* (chosen on ViT-unseen dev), LM retrained on all real-non-val.
        c0, w0 = acc(val_cache, val_keys, lm_full, 0.0)
        cs, ws = acc(val_cache, val_keys, lm_full, lam_star)
        print(
            f"\n300-VAL (HONEST, LM={len(texts_of(train_crops)) + len(dev_crops)}, fixed lambda*={lam_star}):"
        )
        print(f"  baseline (lambda=0):     char {c0:.4f}  word {w0:.4f}")
        print(
            f"  rescored (lambda={lam_star}):  char {cs:.4f}  word {ws:.4f}"
            f"   (dchar {cs - c0:+.4f}, dword {ws - w0:+.4f})"
        )

    # n-best diversity: if beams are degenerate (all identical), rescoring can't help.
    div = sum(len({t for t, _ in nb}) for _, nb in val_cache.values()) / max(1, len(val_cache))
    print(f"\navg distinct hypotheses / line (of {args.nbest}-best): {div:.2f}")

    # Honest diagnostic given NO clean dev exists: sweep lambda directly on the 300-val
    # (tuned-on-test) to see whether the LM helps TrOCR at all.
    print("\n300-VAL direct sweep (LM=600, tuned-on-test diagnostic):")
    print(f"{'lambda':>7} | {'char':>8} | {'word':>8}")
    for lam in lambdas:
        ca, wa = acc(val_cache, val_keys, lm_full, lam)
        print(f"{lam:>7} | {ca:>8.4f} | {wa:>8.4f}")


if __name__ == "__main__":
    main()
