"""Build a *kraken-style* augmentation pool from the real line crops.

Motivation (spec §6.5.22 follow-up): the kraken 600-real fine-tune with ketos's
built-in `--augment` reached 0.9710 on the 300-val — beating every synthetic-tier
kraken run and our font-re-render pools. Open question: is that because of the
kraken *augmentation style* (image-level perturbation of the REAL crops) rather than
our pipeline's style (re-rendering the text in medieval fonts)? To test it on
ViT+RoBERTa we need the same kind of images kraken trains on, as a static pool.

This script does NOT approximate kraken — it imports kraken's own line-recognition
augmenter (`kraken.lib.dataset.recognition.DefaultAugmenter`, the exact Compose
ketos `--augment` uses: PixelDropout + one-of blur + one-of optical/elastic/rotate,
overall p=0.5) and applies it offline to each real crop N times. kraken applies this
on the fly once/line/epoch and never stores it; we materialise N variants/line so a
static-pool trainer (TrOCR) sees comparable diversity. p=0.5 is kept as-is, so ~half
the variants pass through clean — mirroring kraken's own behaviour.

Output matches our other aug pools so `run_trocr_finetune.py` consumes it directly:
`<out>/<stem>_aug<NN>.png` + `<out>/labels.json` ({filename: text}).

    PROJECT_ROOT=. uv run python scripts/data_augmentation/run_kraken_style_augment.py \
        --real-folder data/processed/annotated_samples/OCR/full_annotated \
        --n-per-line 5 --out data/processed/synthetic_samples/augmented_images/kraken_style_600x5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("kraken_style_aug")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-folder", type=Path, required=True, help="<stem>.png + <stem>.gt.txt")
    ap.add_argument("--n-per-line", type=int, default=5, help="augmented variants per real line")
    ap.add_argument("--out", type=Path, required=True, help="output pool directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Deterministic: kraken's albumentations pipeline reads numpy's global RNG.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from kraken.lib.dataset.recognition import DefaultAugmenter

    augmenter = DefaultAugmenter()  # the exact ketos --augment Compose (p=0.5)
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = sorted(args.real_folder.glob("*.png"))
    labels: dict[str, str] = {}
    n_written = 0
    n_skipped = 0
    for png in pairs:
        gt = png.with_suffix(".gt.txt")
        if not gt.is_file():
            n_skipped += 1
            continue
        text = gt.read_text(encoding="utf-8").strip()
        if not text:
            n_skipped += 1
            continue
        rgb = Image.open(png).convert("RGB")
        chw = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)  # CHW uint8, as kraken expects
        for k in range(args.n_per_line):
            out = augmenter(chw, k)  # -> CHW float [0,1]
            arr = (out.permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype("uint8")
            name = f"{png.stem}_aug{k:02d}.png"
            Image.fromarray(arr).save(args.out / name)
            labels[name] = text
            n_written += 1

    (args.out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Wrote %d kraken-style aug images (%d lines x %d, %d skipped) -> %s",
        n_written,
        len(pairs) - n_skipped,
        args.n_per_line,
        n_skipped,
        args.out,
    )
    log.info("labels.json: %d entries", len(labels))


if __name__ == "__main__":
    main()
