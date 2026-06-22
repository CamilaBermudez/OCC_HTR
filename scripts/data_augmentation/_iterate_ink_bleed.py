"""Throwaway iteration script for apply_ink_bleed.

Runs composite_on_parchment + apply_ink_bleed on the same 8 source lines as
extreme_damage_samples, with fixed seeds, so I can visually compare against
the user's reference photo and the bleed_*.png currently on disk. Not part
of the pipeline — delete when the look is locked.
"""

import os
import sys
from pathlib import Path

import cv2

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.environ.setdefault("PROJECT_ROOT", str(project_root))

from src.data_augmentation.augmentation_techniques import (  # noqa: E402
    apply_aged_parchment_effects,
    apply_augmentation_techniques,
    apply_ink_bleed,
    composite_on_parchment,
)

_ = (
    apply_aged_parchment_effects,
    apply_ink_bleed,
    composite_on_parchment,
)  # kept for ad-hoc tweaks

SRC_DIR = project_root / "data/processed/synthetic_text/medieval_text_20260607_224326"
PARCH_DIR = project_root / "tests/augmentation/parchment_crops"
OUT_DIR = project_root / "tests/augmentation/ink_bleed_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

src_files = sorted(SRC_DIR.glob("*.png"))
parch_files = sorted(PARCH_DIR.glob("*.png"))
assert src_files and parch_files

# Reuse first 8 source lines (matches extreme_damage_samples).
chosen = src_files[:8]

for i, sp in enumerate(chosen):
    seed = 100 + i
    img = cv2.imread(str(sp))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Run the actual production pipeline so morphological erosion / dropout
    # / warp / scan capture all fire — otherwise we'd only be testing the
    # custom-Lambda steps in isolation.
    out = apply_augmentation_techniques(img, parch_files, bleed_source_files=src_files, seed=seed)[
        "image"
    ]
    out_path = OUT_DIR / f"bleed_{i:02d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_path.name}")
