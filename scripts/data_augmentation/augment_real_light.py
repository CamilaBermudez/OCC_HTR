"""Light augmentation of REAL manuscript line crops (spec §6.5.24 follow-up).

Unlike ``run_augment_images.py`` (which turns a CLEAN synthetic render into a
real-looking image via parchment-composite + aging + heavy warp), real crops are
ALREADY on parchment and already degraded. Applying that pipeline over-degrades
them (frozen-catmus CER 0.054 raw -> 0.109 gentle / 0.193 harsh). This preset keeps
ONLY mild geometry + light blur/noise (no composite / aging / ink-bleed / heavy
warp), which holds real crops at real legibility (catmus CER 0.050 ~ 0.054 raw).

Produces N light-augmented copies per ``<stem>.png`` (with a sibling ``<stem>.gt.txt``)
plus a flat ``labels.json`` {aug_name: gt_text}, ready for the kraken/ketos finetune
(``--augmented-folder`` + ``--labels-json``, run with ketos ``--no-augment`` to test
offline light-aug vs ketos's online ``--augment``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image


def light_transform() -> A.ReplayCompose:
    """The validated light preset (catmus CER ~0.050 on real crops)."""
    return A.ReplayCompose(
        [
            A.Morphological(scale=(1, 1), operation="dilation", p=0.15),
            A.Affine(
                translate_percent={"x": (-0.01, 0.01), "y": (-0.02, 0.02)},
                rotate=(-1.5, 1.5),
                scale=(0.98, 1.02),
                border_mode=cv2.BORDER_REPLICATE,
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.3),
            A.GaussNoise(std_range=(0.004, 0.010), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.06, contrast_limit=0.06, p=0.4),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-folder", required=True, help="Folder of <stem>.png + <stem>.gt.txt")
    ap.add_argument("--output-folder", required=True)
    ap.add_argument("--n-augmentations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    real = Path(args.real_folder)
    out = Path(args.output_folder)
    out.mkdir(parents=True, exist_ok=True)
    tf = light_transform()

    pngs = sorted(real.glob("*.png"))
    labels: dict[str, str] = {}
    n_saved = 0
    for i, png in enumerate(pngs):
        gt_path = png.with_suffix("").with_suffix(".gt.txt")
        if not gt_path.is_file():
            continue
        gt = gt_path.read_text(encoding="utf-8").strip()
        if not gt:
            continue
        arr = np.array(Image.open(png).convert("RGB"))
        for j in range(args.n_augmentations):
            tf.set_random_seed(args.seed + i * args.n_augmentations + j)
            aug = tf(image=arr)["image"]
            name = f"{png.stem}_aug{j:02d}.png"
            Image.fromarray(aug).save(out / name)
            (out / name.replace(".png", ".gt.txt")).write_text(gt, encoding="utf-8")
            labels[name] = gt
            n_saved += 1

    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    print(f"saved {n_saved} light-aug images from {len(pngs)} real crops -> {out}")


if __name__ == "__main__":
    main()
