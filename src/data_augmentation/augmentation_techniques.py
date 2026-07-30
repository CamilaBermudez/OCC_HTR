"""Augmentation utilities for synthetic OCR/HTR training data.

Components:
- `get_parchment_crops`: one-time setup; extract empty-parchment crops from
  manuscript page scans by sampling random windows and keeping the ones with
  lowest Canny edge density.
- `composite_on_parchment`: translucent multiplicative-ink composite onto a
  random parchment crop, with stroke-density noise and optional verso
  bleed-through. Designed to be wrapped in `A.Lambda` inside the pipeline.
- `apply_augmentation_techniques`: the full Option-A augmentation pipeline
  (ink degradation + parchment composite + warp + scan-capture effects)
  wrapped in `A.ReplayCompose` so every call records exactly what fired.
- `batch_augment_directory`: walk a directory of source images, apply the
  pipeline N times per source with deterministic per-call seeds, save the
  augmented images, and write a consolidated reproducibility log (JSON +
  optional plain-text run log).
"""

import datetime
import functools
import json
import logging
import os
import random
import subprocess
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def setup_augmentation_logging(logs_dir: str | Path, run_name: str):
    """File + console logger for an augmentation run. Mirrors the pattern in
    `src/data_preprocessing/crop_image_segments.py`."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"{run_name}_augmentation.log"

    logger = logging.getLogger("augmentation")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for h in (file_handler, console):
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger, str(log_file)


def _get_git_commit() -> str:
    """Short git SHA at PROJECT_ROOT, or 'unknown' if unavailable. Used to pin
    the exact pipeline code that produced an augmentation run."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.environ.get("PROJECT_ROOT", "."),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


_REPLAY_NOISY_KEYS = {
    "matrix",
    "bbox_matrix",
    "noise_map",
    "mask_drop_mask",
    "mask_drop_values",
    "shape",
    "output_shape",
    "interpolation",
    "mask_interpolation",
    "fill",
    "fill_mask",
}


def _to_python(v):
    """Coerce numpy scalars / containers into JSON-friendly Python primitives."""
    if hasattr(v, "item") and not hasattr(v, "__len__"):
        return v.item()
    if isinstance(v, dict):
        return {k: _to_python(vv) for k, vv in v.items()}
    if isinstance(v, list | tuple):
        return [_to_python(x) for x in v]
    return v


def _clean_transform_node(node):
    """Walk one transform's replay dict and keep only fields useful for review:
    class name, applied flag, scalar params, nested transforms (for OneOf)."""
    if not isinstance(node, dict):
        return node
    out = {
        "name": node.get("__class_fullname__", "?").rsplit(".", 1)[-1],
        "applied": node.get("applied", False),
    }
    if "transforms" in node:
        out["transforms"] = [_clean_transform_node(t) for t in node["transforms"]]
    params = node.get("params") or {}
    scalar_params = {}
    for k, v in params.items():
        if k in _REPLAY_NOISY_KEYS:
            continue
        if hasattr(v, "shape") and getattr(v, "ndim", 0) >= 1:
            continue  # drop arrays
        scalar_params[k] = _to_python(v)
    if scalar_params:
        out["params"] = scalar_params
    return out


def get_parchment_crops(
    input_dir: Path,
    output_dir: Path,
    run_name: str,
    crop_size: int = 200,
    candidates_page: int = 40,
    keep_page: int = 3,
    edge_threshold: float = 6.0,
    min_brightness: float = 100.0,
    max_blue_fraction: float = 0.002,
    seed: int = 0,
    plot_: bool = False,
):
    """Extract low-edge ("empty parchment") crops from manuscript pages.

    For each image, sample random square windows and score them by mean
    Canny edge density. Lower scores indicate emptier parchment (better).

    `min_brightness` rejects crops whose mean grayscale value is below
    the threshold. Uniformly-dark regions (page borders, book spine /
    gutter, margins) have near-zero Canny edges and would otherwise
    slip through the edge-density filter, contaminating the parchment
    pool with solid black patches.

    `max_blue_fraction` rejects crops whose share of saturated-blue
    pixels (B − R > 30 AND B > 80) exceeds the threshold. Illuminated
    initials and their marginal frames are painted with strong blue
    pigment; a crop that catches even a corner of one will inject a
    blue rectangle into every augmented sample that lands on it. Canny
    barely sees these solid-colour regions, so they otherwise pass the
    edge-density filter.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    page_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    assert page_files, f"No pages found in {input_dir}"

    # Clear any prior run so reruns don't accumulate stale crops.
    for old in save_dir.glob("parchment_*.png"):
        old.unlink()

    saved: list[tuple[Path, float, np.ndarray]] = []
    for page_path in page_files:
        page = cv2.imread(str(page_path))
        if page is None:
            continue
        page = cv2.cvtColor(page, cv2.COLOR_BGR2RGB)
        h, w = page.shape[:2]
        if h < crop_size or w < crop_size:
            continue

        candidates: list[tuple[float, np.ndarray]] = []
        n_pixels = crop_size * crop_size
        for _ in range(candidates_page):
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            crop = page[y : y + crop_size, x : x + crop_size]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            # Reject solid-dark regions (page borders / book spine) BEFORE
            # scoring by Canny — uniformly dark regions have ~0 edge
            # density and would otherwise outrank real parchment.
            if float(gray.mean()) < min_brightness:
                continue
            # Reject illuminated-initial frames (saturated blue pigment).
            # Canny is blind to large solid colour regions, so a blue patch
            # would slip through the edge filter.
            blue_pixels = int(
                (
                    (crop[..., 2].astype(int) - crop[..., 0].astype(int) > 30) & (crop[..., 2] > 80)
                ).sum()
            )
            if blue_pixels / n_pixels > max_blue_fraction:
                continue
            edges = cv2.Canny(gray, 50, 150)
            score = float(edges.mean())
            candidates.append((score, crop))

        candidates.sort(key=lambda c: c[0])
        for score, crop in candidates[:keep_page]:
            if score > edge_threshold:
                continue
            idx = len(saved)
            out_path = save_dir / f"parchment_{idx:03d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved.append((out_path, score, crop))

    print(f"Saved {len(saved)} crops to {save_dir}")
    if saved:
        mean_score = float(np.mean([s for _, s, _ in saved]))
        print(f"Mean edge score across kept crops: {mean_score:.2f}")

    if plot_ and saved:
        n = min(len(saved), 16)
        cols = 4
        rows = (n + cols - 1) // cols
        _, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
        axes = np.array(axes).reshape(rows, cols)
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]
            if i < n:
                path, score, crop = saved[i]
                ax.imshow(crop)
                ax.set_title(f"{path.name}  edge={score:.2f}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    return saved


def composite_on_parchment(image, parchment_files, **kwargs):
    """Composite a rendered glyph image onto a real parchment crop.

    Designed to be wrapped by `A.Lambda(image=partial(composite_on_parchment,
    parchment_files=...))`. `**kwargs` absorbs Albumentations' Lambda extras
    (`cols`, `rows`, etc.) so the call signature is forgiving.

    Expects `image` in RGB (uint8). Returns RGB (uint8) of the same shape.
    """
    assert parchment_files, "parchment_files is empty — run get_parchment_crops first"

    bg = cv2.imread(str(random.choice(parchment_files)))
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    bh, bw = bg.shape[:2]
    if bh < h or bw < w:
        bg = cv2.resize(bg, (max(w, bw), max(h, bh)))
        bh, bw = bg.shape[:2]
    y0 = random.randint(0, bh - h)
    x0 = random.randint(0, bw - w)
    bg_crop = bg[y0 : y0 + h, x0 : x0 + w].astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ink = np.clip((0.55 - gray) / 0.35, 0.0, 1.0)

    # Low-frequency noise — word/letter-scale density variation (some
    # words have more pigment than others, like a quill being refilled
    # mid-line).
    nh, nw = max(8, h // 15), max(8, w // 15)
    coarse = np.random.rand(nh, nw).astype(np.float32)
    noise = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.clip(noise * 1.3 + 0.1, 0.4, 1.0)
    # High-frequency noise — within-stroke density variation that breaks
    # the digital uniformity of the rendered glyph. Real quill strokes
    # skip and pool: a single downstroke isn't a solid black bar.
    fh, fw = max(16, h // 4), max(16, w // 4)
    fine_coarse = np.random.rand(fh, fw).astype(np.float32)
    fine_noise = cv2.resize(fine_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    # Clip toward 1.0 so the noise mostly *lightens* parts of strokes
    # (skipping quill) rather than darkening them further.
    fine_noise = np.clip(fine_noise * 0.7 + 0.5, 0.55, 1.0)
    ink = ink * noise * fine_noise

    bleed = np.zeros_like(ink)
    if random.random() < 0.6:
        ghost = cv2.flip(ink, 1)
        ghost = cv2.GaussianBlur(ghost, (0, 0), sigmaX=4.0)
        bleed = ghost * 0.22

    ink = ink[..., None]
    bleed = bleed[..., None]

    # Iron-gall ink reads as a saturated red-brown, not jet black. Channel
    # multipliers (R > G > B with a wide spread) push dense-ink pixels toward
    # ~RGB(80, 50, 25) on a typical parchment crop — matches the warm sepia
    # of the reference manuscripts rather than the near-black we had before.
    ink_mult_brown = np.array([0.32, 0.22, 0.13], dtype=np.float32)
    # Rubric red (cinnabar / minium): the second ink the scribes used for
    # capitulum headers and section initials. Lower B than brown and higher
    # R, so dense-ink pixels read as a saturated dark red on parchment.
    ink_mult_red = np.array([0.55, 0.15, 0.10], dtype=np.float32)
    # Per-pixel selector: only pixels with very strong red dominance get
    # the red multiplier. Brown ink (60, 40, 20) and warm parchment have
    # R > G > B by ~20, so a naive R - max(G,B) > 0 test would blend them
    # toward red too — needs a hard offset to reject. Rubric red glyphs
    # upstream are painted at RUBRIC_RED ≈ (140, 30, 25), giving R - max
    # (G,B) ≈ 110 which trips this cleanly.
    img_f = image.astype(np.float32)
    red_score = np.clip(
        (img_f[..., 0] - np.maximum(img_f[..., 1], img_f[..., 2]) - 40.0) / 50.0,
        0.0,
        1.0,
    )[..., None]
    ink_mult_per_pixel = (1.0 - red_score) * ink_mult_brown + red_score * ink_mult_red

    bleed_mult = np.array([0.88, 0.83, 0.72], dtype=np.float32)
    out = bg_crop * (bleed_mult**bleed) * (ink_mult_per_pixel**ink)
    return out.clip(0, 255).astype(np.uint8)


def apply_aged_parchment_effects(image, **kwargs):
    """Add centuries-of-aging visual effects to a composited line image.

    Each sub-effect has its own internal probability so different samples
    show different aging combinations. Designed for line-level crops:
    page-level effects (torn edges, wax seal, tide lines) are omitted
    because they don't translate to line crops.

    Sub-effects layered in order:
      1. Iron-gall ink browning (p=0.85): the dark sepia ink of iron gall
         oxidises toward warm brown over time. Shift R up and B down on
         pixels weighted by ink density, so only the ink browns — the
         parchment isn't touched.
      2. Global ink wear (p=0.55): low-frequency noise field that fades
         ink unevenly across the WHOLE line — some letters retain density,
         others are heavily worn. This simulates widespread parchment wear
         from centuries of handling, not localised damage. Without this,
         only the crease band shows wear and the rest of the line stays
         too clean.
      3. Foxing spots (p=0.60): 2–7 small Gaussian patches of brownish-rust
         tone scattered randomly. Each is sized 2.5–8 px and applied at
         30–55% opacity. Simulates mold/oxidation stains.
      4. Micro-pitting around dense ink (p=0.50): tiny dark dots clustered
         near and within dense-ink regions. Simulates iron-gall corrosion
         that has eaten micro-pits into the parchment. 2–6% of dense-ink
         neighbourhood pixels become pits.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Local parchment colour (target for the ink-wear blend).
    flat = img.reshape(-1, 3)
    brightness = flat.mean(axis=1)
    bright_threshold = np.percentile(brightness, 75)
    bright_pixels = flat[brightness >= bright_threshold]
    parchment_color = (
        bright_pixels.mean(axis=0)
        if len(bright_pixels) > 0
        else np.array([218, 200, 170], dtype=np.float32)
    )

    # Ink density mask used by multiple effects.
    gray = (img.mean(axis=2) / 255.0).astype(np.float32)
    ink_mask = np.clip((0.45 - gray) / 0.40, 0.0, 1.0)

    # 1. Iron-gall ink browning.
    if random.random() < 0.85:
        iron_gall_shift = np.array([1.08, 1.00, 0.85], dtype=np.float32)
        strength = random.uniform(0.20, 0.45) * ink_mask[..., None]
        img = img * (1.0 - strength) + (img * iron_gall_shift) * strength

    # 1b. Horizontal ink density variation — smoothed per-column opacity
    #     so some words/areas show denser ink, others fainter. Real
    #     manuscripts always have this; without it, every letter on a
    #     line gets identical pigment density (synthetic-looking). No
    #     internal probability — applied on every aged sample.
    col_field = np.random.uniform(0.55, 1.0, w).astype(np.float32)
    kernel_size = max(31, w // 8 * 2 + 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    col_smooth = np.convolve(col_field, kernel, mode="same")
    # Convert to per-column "how much ink to fade out": (1 - opacity).
    col_fade = (1.0 - col_smooth.reshape(1, -1)) * ink_mask
    col_fade_3d = col_fade[..., None]
    img = img * (1.0 - col_fade_3d) + parchment_color * col_fade_3d
    # Recompute ink mask since the ink density has changed.
    gray = (img.mean(axis=2) / 255.0).astype(np.float32)
    ink_mask = np.clip((0.45 - gray) / 0.40, 0.0, 1.0)

    # 2. Global ink wear — low-frequency noise field that fades ink
    #    unevenly across the whole line, not just one localised band.
    #    With ~18% probability, switch to an EXTREME mode that pushes
    #    the wear field toward 1.0 across most pixels and applies full
    #    strength — this is the "hardest damage" tail that matches the
    #    near-ghost lines seen in worn manuscripts.
    if random.random() < 0.60:
        extreme_wear = random.random() < 0.18
        noise_h = max(6, h // 8)
        noise_w = max(16, w // 25)
        coarse = np.random.rand(noise_h, noise_w).astype(np.float32)
        wear_field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
        wear_field = np.clip(wear_field, 0.0, 1.0)
        if extreme_wear:
            # Power < 1 PUSHES values up toward 1: most pixels end up in
            # the high-wear regime. Combined with global_strength near 1.0
            # this fades ink-rich pixels nearly to parchment.
            wear_field = wear_field**0.55
            global_strength = random.uniform(0.92, 1.0)
        else:
            wear_field = wear_field**1.4
            global_strength = random.uniform(0.55, 0.98)
        wear_fade = global_strength * wear_field * ink_mask
        wear_3d = wear_fade[..., None]
        img = img * (1.0 - wear_3d) + parchment_color * wear_3d

    # 3. Foxing spots (more spots, larger, stronger at the upper end).
    if random.random() < 0.65:
        n_spots = random.randint(3, 12)
        y_grid = np.arange(h, dtype=np.float32).reshape(-1, 1)
        x_grid = np.arange(w, dtype=np.float32).reshape(1, -1)
        for _ in range(n_spots):
            cx = random.uniform(0, w)
            cy = random.uniform(0, h)
            radius = random.uniform(2.5, 10.0)
            spot_color = np.array(
                [
                    random.uniform(120, 170),
                    random.uniform(75, 115),
                    random.uniform(45, 85),
                ],
                dtype=np.float32,
            )
            strength_max = random.uniform(0.30, 0.70)
            dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
            spot_mask = strength_max * np.exp(-dist_sq / (2 * radius**2))
            spot_3d = spot_mask[..., None]
            img = img * (1.0 - spot_3d) + spot_color * spot_3d

    # 4. Micro-pitting around dense ink (iron-gall corrosion).
    if random.random() < 0.55:
        dense_ink = (ink_mask > 0.6).astype(np.float32)
        dense_ink = cv2.dilate(dense_ink, np.ones((3, 3), dtype=np.uint8), iterations=1)
        pit_noise = np.random.rand(h, w).astype(np.float32)
        pit_threshold = 1.0 - random.uniform(0.02, 0.10)
        pit_mask = ((pit_noise > pit_threshold) & (dense_ink > 0.5)).astype(np.float32)
        pit_color = np.array([35, 25, 15], dtype=np.float32)
        pit_strength = 0.80
        pit_3d = (pit_mask * pit_strength)[..., None]
        img = img * (1.0 - pit_3d) + pit_color * pit_3d

    return img.clip(0, 255).astype(np.uint8)


def apply_torn_edges(image, **kwargs):
    """Simulate a ragged torn edge along the top, bottom, or both sides of
    the line crop. The torn region is filled with a dark color so the
    "missing" parchment reads as the void behind the page.

    Produces an irregular zigzag edge (random vertex spacing 4–22 px,
    depth up to ~1/6 of image height) with a slight blur for natural
    softness. Useful for ~10–20% of training samples so the HTR model
    learns to handle line crops where the page edge is damaged or where
    the crop intersects a tear.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    max_tear = max(4, h // 6)
    sides = random.choice(["top", "bottom", "both"])

    # Build a binary mask: 1 where the parchment shows, 0 inside the tear.
    mask = np.ones((h, w), dtype=np.float32)

    def _zigzag_polygon(from_top: bool):
        """Return polygon vertices for the tear on the chosen side."""
        pts: list[tuple[int, int]] = []
        x = 0
        while x < w:
            y_tear = random.randint(0, max_tear)
            y = y_tear if from_top else (h - 1 - y_tear)
            pts.append((min(x, w - 1), y))
            x += random.randint(4, 22)
        # Close the polygon along the corresponding canvas edge.
        if from_top:
            return [(0, 0)] + pts + [(w - 1, 0)]
        return [(0, h - 1)] + pts + [(w - 1, h - 1)]

    if sides in ("top", "both"):
        poly = np.array(_zigzag_polygon(from_top=True), dtype=np.int32)
        cv2.fillPoly(mask, [poly], 0.0)
    if sides in ("bottom", "both"):
        poly = np.array(_zigzag_polygon(from_top=False), dtype=np.int32)
        cv2.fillPoly(mask, [poly], 0.0)

    # Soft blur so the tear edge doesn't read as a clean digital line.
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)

    # Blend: parchment where mask=1, dark "void" where mask=0.
    dark_void = np.array([15, 10, 8], dtype=np.float32)
    mask_3d = mask[..., None]
    img = img * mask_3d + dark_void * (1.0 - mask_3d)

    return img.clip(0, 255).astype(np.uint8)


def apply_ink_bleed(image, bleed_source_files=None, **kwargs):
    """Simulate verso ink bleed-through: faint mirrored ghost text from
    the back of the parchment showing through to the front.

    The recto (current line) stays SHARP and READABLE. The bleed appears
    as faint blurred ghost text in the parchment background, including
    in margins/gaps where there's no recto ink — exactly what you see on
    a real folio with multiple rows of verso text bleeding through.

    Recipe:
      1. Build a synthetic "verso page" by stacking 2-4 *different*
         rendered lines from the source pool at random vertical
         positions, each squashed to ~50-80% height so several rows fit
         vertically. Without this stacking the ghost is a single thin
         band; the reference photos show ghost text across the whole
         crop height.
      2. Mirror horizontally (verso reads mirrored on the recto).
      3. Heavily Gaussian-blur the combined mask — diffusion through
         parchment fibres softens stroke edges into a wash.
      4. Gate by a parchment-only mask (gray>0.30) so existing recto
         ink isn't muddied.
      5. Multiplicative darkening plus a subtle warm-brown tint so the
         ghost reads as oxidised iron-gall, not grey.
    """
    if not bleed_source_files:
        return image
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # 1. Synthetic verso page — multiple ghost lines stacked vertically.
    #    Each line gets a light pre-blur before placement so its own edges
    #    don't survive as hard rectangles after the global blur (this was
    #    the corner-block artifact in the previous iteration).
    verso = np.zeros((h, w), dtype=np.float32)
    n_lines = random.randint(2, 4)
    placed = 0
    pre_blur_k = max(3, (h // 18) | 1)
    for _ in range(n_lines * 2):  # extra attempts in case some loads fail
        if placed >= n_lines:
            break
        bleed_raw = cv2.imread(str(random.choice(bleed_source_files)))
        if bleed_raw is None:
            continue
        bleed_gray = cv2.cvtColor(bleed_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        line_mask = np.clip((220.0 - bleed_gray) / 180.0, 0.0, 1.0)
        line_h_scale = random.uniform(0.45, 0.80)
        new_h = max(8, int(round(h * line_h_scale)))
        line_resized = cv2.resize(line_mask, (w, new_h), interpolation=cv2.INTER_LINEAR)
        line_resized = cv2.GaussianBlur(line_resized, (pre_blur_k, pre_blur_k), 0)
        y_off = random.randint(-new_h // 2, h - new_h // 2)
        y0, y1 = max(0, y_off), min(h, y_off + new_h)
        src_y0 = max(0, -y_off)
        src_y1 = src_y0 + (y1 - y0)
        if y1 > y0 and src_y1 > src_y0:
            verso[y0:y1, :] = np.maximum(verso[y0:y1, :], line_resized[src_y0:src_y1, :])
            placed += 1
    if placed == 0:
        return image

    # 2. Mirror horizontally — verso reads mirrored on the recto.
    if random.random() < 0.85:
        verso = verso[:, ::-1].copy()

    # 3. Heavy diffusion through parchment fibres. Big kernel so individual
    #    letter forms blur into a wash and the ghost reads as a stain
    #    rather than a sharp imprint.
    blur_k = max(13, (h // 3) | 1)
    verso = cv2.GaussianBlur(verso, (blur_k, blur_k), 0)
    # Gentle renormalisation: re-scale only if the blur dropped the peak
    # well below 1.0, but cap the scale so a single dense stroke can't
    # blow out the whole field.
    vmax = float(verso.max())
    if vmax > 0.05:
        verso = verso / max(vmax, 0.55)
        verso = np.clip(verso, 0.0, 1.0)

    # 4. Parchment-only gate — preserve sharp recto ink.
    gray_orig = (img.mean(axis=2) / 255.0).astype(np.float32)
    parchment_gate = np.clip((gray_orig - 0.30) / 0.50, 0.0, 1.0)
    ghost_factor = verso * parchment_gate

    # 5. Multiplicative darkening + warm brown tint. Strength tuned so the
    #    ghost is clearly visible across the whole crop but stays faint
    #    enough that the recto remains the dominant signal.
    darken_amt = random.uniform(0.55, 0.75)
    multiplier = 1.0 - (1.0 - darken_amt) * ghost_factor
    img = img * multiplier[..., None]

    # Warm brown tint: iron-gall ink browns as it migrates through the
    # parchment, so the ghost reads brown rather than grey.
    tint = np.array([1.00, 0.94, 0.82], dtype=np.float32)
    tint_mix = (ghost_factor * 0.5)[..., None]
    img = img * (1.0 - tint_mix) + (img * tint) * tint_mix

    return img.clip(0, 255).astype(np.uint8)


def apply_page_creases(image, **kwargs):
    """Simulate a centuries-old parchment fold with realistic ink degradation.

    Models the visible wear of a manuscript that has been folded and
    unfolded many times across centuries. The fold is revealed primarily
    through TEXTURE and INK DEGRADATION, not a dark shadow:

      1. Wavy crease centerline: low-frequency sinusoid (≤3 px) along x —
         not perfectly straight, like a real fold.
      2. Mechanical-wear noise modulation: high-frequency noise multiplies
         the base fade profile, so the ink doesn't fade smoothly but
         unevenly — the impression of "rubbed" texture rather than a
         digital gradient.
      3. Abrasion spots: 10–25 small Gaussian patches of heavy fading,
         concentrated near the crease, create the "interrupted strokes /
         missing fragments" appearance where letters cross the fold.
      4. Asymmetric warp: subtle paper-fiber compression — pixels above
         and below the crease drift toward it (0.6–1.4 px).
      5. Discoloration: a warm-yellow tint weighted by the fade profile,
         simulating age-related contamination concentrated in the fold.

    Ink along the crease is blended toward the local parchment color so
    letters appear partially rubbed away rather than blacked out. The
    fading is strongest at the crease itself and diminishes outward.
    """
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Estimate local parchment color (target for the ink-fade blend).
    flat = img.reshape(-1, 3)
    brightness = flat.mean(axis=1)
    bright_threshold = np.percentile(brightness, 75)
    bright_pixels = flat[brightness >= bright_threshold]
    parchment_color = (
        bright_pixels.mean(axis=0)
        if len(bright_pixels) > 0
        else np.array([218, 200, 170], dtype=np.float32)
    )

    # Capture ink density on the ORIGINAL (un-faded) input. The extreme
    # smudge-patch step later needs to know where the letters actually
    # were; recomputing it on the already-band-faded image gives a
    # near-uniform low value and the smudge falls on parchment instead
    # of text.
    parchment_brightness = float(parchment_color.mean()) / 255.0
    gray_orig = (img.mean(axis=2) / 255.0).astype(np.float32)
    ink_density_orig = np.clip(
        (parchment_brightness - gray_orig) / max(parchment_brightness, 0.01), 0.0, 1.0
    )

    # 2D coordinate grids.
    y_grid = np.arange(h, dtype=np.float32).reshape(-1, 1)
    x_grid = np.arange(w, dtype=np.float32).reshape(1, -1)

    # Wavy crease centerline.
    crease_y = random.uniform(h * 0.30, h * 0.70)
    wave_freq = random.uniform(0.002, 0.008)
    wave_amp = random.uniform(1.5, 3.5)
    wave_phase = random.uniform(0, 2 * np.pi)
    crease_y_per_x = crease_y + wave_amp * np.sin(x_grid * wave_freq * 2 * np.pi + wave_phase)
    dist_from_crease = y_grid - crease_y_per_x

    # With ~18% probability, switch to an EXTREME mode that widens the
    # fade band to cover most of the line height, raises fade strength
    # close to 1.0, and ups the abrasion-spot count/size/strength. This
    # is the "centuries-old crease where this row of text is nearly gone"
    # tail of the damage distribution.
    extreme_crease = random.random() < 0.18

    # Base fade profile (Gaussian falloff from the crease).
    if extreme_crease:
        # Sigma >= ~40% of line height covers essentially the whole line
        # — most ink pixels feel the fade.
        fade_sigma = max(8.0, h * 0.25) * random.uniform(0.85, 1.15)
    else:
        fade_sigma = max(8.0, h * 0.10) * random.uniform(0.85, 1.15)
    base_fade = np.exp(-(dist_from_crease**2) / (2 * fade_sigma**2))

    # Mechanical-wear noise: medium-frequency multiplicative texture so the
    # fade isn't a smooth gradient. In EXTREME mode use a higher-frequency
    # grid so the texture reads as parchment-fibre rubbing rather than a
    # smooth wash. Modulates between 0.4× and 1.0×.
    if extreme_crease:
        noise_h = max(12, h // 3)
        noise_w = max(64, w // 8)
    else:
        noise_h = max(8, h // 6)
        noise_w = max(32, w // 20)
    coarse_noise = np.random.rand(noise_h, noise_w).astype(np.float32)
    wear_noise = cv2.resize(coarse_noise, (w, h), interpolation=cv2.INTER_CUBIC)
    wear_noise = 0.4 + 0.6 * wear_noise

    # Smooth fade mask (noise-modulated Gaussian).
    if extreme_crease:
        fade_strength = random.uniform(0.65, 0.85)
    else:
        fade_strength = random.uniform(0.60, 0.85)
    smooth_fade = base_fade * wear_noise * fade_strength

    # Abrasion spots — small Gaussian patches of heavy fading concentrated
    # near the crease line, creating the "letter fragments missing" effect.
    spot_mask = np.zeros((h, w), dtype=np.float32)
    if extreme_crease:
        n_spots = random.randint(30, 60)
        spot_radius_range = (3.5, 8.0)
        spot_strength_range = (0.80, 0.97)
        spot_y_jitter = fade_sigma * 0.6
    else:
        n_spots = random.randint(10, 25)
        spot_radius_range = (1.5, 4.5)
        spot_strength_range = (0.55, 0.90)
        spot_y_jitter = fade_sigma * 0.4
    crease_y_flat = crease_y_per_x[0]  # (w,)
    for _ in range(n_spots):
        spot_x = random.randint(0, w - 1)
        spot_y_target = crease_y_flat[spot_x] + random.uniform(-spot_y_jitter, spot_y_jitter)
        spot_radius = random.uniform(*spot_radius_range)
        spot_strength = random.uniform(*spot_strength_range)
        dist_sq = (x_grid - spot_x) ** 2 + (y_grid - spot_y_target) ** 2
        spot = spot_strength * np.exp(-dist_sq / (2 * spot_radius**2))
        spot_mask = np.maximum(spot_mask, spot)

    # Combine smooth fade with abrasion spots; spots win where they're
    # stronger (so missing-fragment effect dominates locally).
    fade_cap = 0.88 if extreme_crease else 0.92
    combined_fade = np.clip(np.maximum(smooth_fade, spot_mask), 0, fade_cap)

    # Asymmetric warp (paper fiber compression).
    warp_sigma = fade_sigma * 0.7
    warp_amplitude = random.uniform(0.6, 1.4)
    displacement_y = (
        -np.sign(dist_from_crease)
        * warp_amplitude
        * np.exp(-(dist_from_crease**2) / (2 * warp_sigma**2))
    )
    x_coords, y_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    y_coords_warped = y_coords + displacement_y
    img = cv2.remap(
        img,
        x_coords,
        y_coords_warped,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Compute an ink-density mask on the WARPED image so the fade can
    # specifically target ink pixels (not parchment). Without this, the
    # blend toward parchment lightens the whole band uniformly, making it
    # look like a brighter stripe rather than rubbed-away letters.
    gray = (img.mean(axis=2) / 255.0).astype(np.float32)
    ink_density = np.clip((parchment_brightness - gray) / max(parchment_brightness, 0.01), 0.0, 1.0)

    # EXTREME mode only: add multiple irregular SMUDGE PATCHES *before*
    # the band fade. Each patch is a region where letter edges are
    # heavily blurred AND ink is nearly fully faded toward parchment —
    # this is what makes individual characters/syllables in the damaged
    # area disappear while their neighbours remain readable. Applied
    # BEFORE band fade because once band fade converts ink pixels to
    # parchment-colour, the local-fade step has no remaining ink budget
    # to consume and the smudge becomes visually invisible.
    if extreme_crease:
        # Higher-frequency noise produces many small patches (a few
        # characters wide each).
        smudge_noise = np.random.rand(max(6, h // 4), max(16, w // 12)).astype(np.float32)
        smudge_field = cv2.resize(smudge_noise, (w, h), interpolation=cv2.INTER_CUBIC)
        smudge_field = np.clip(smudge_field, 0.0, 1.0)
        # Restrict damage to 1–3 LOCALIZED zones along x. Real fold damage
        # is concentrated in specific portions of a row; without this gate
        # the smudge spreads across the whole line and reads as ink bleed
        # rather than rubbed-fold damage.
        n_zones = random.randint(1, 3)
        zone_weight = np.zeros((1, w), dtype=np.float32)
        zone_sigma = w * random.uniform(0.06, 0.11)
        x_axis = np.arange(w, dtype=np.float32).reshape(1, -1)
        for _ in range(n_zones):
            zone_cx = random.uniform(w * 0.08, w * 0.92)
            zone_weight = np.maximum(
                zone_weight,
                np.exp(-((x_axis - zone_cx) ** 2) / (2 * zone_sigma**2)),
            )
        # Bias HARD by ink density (dilated) so patches reliably form
        # ON the letters within the chosen zones.
        ink_dilated = cv2.dilate(ink_density_orig, np.ones((5, 5), dtype=np.uint8), iterations=2)
        smudge_field = smudge_field * zone_weight * (0.05 + ink_dilated)
        # Threshold relative to the in-zone mass: 80th percentile of
        # weighted values keeps only the most-damaged patch interiors.
        smudge_threshold = float(np.percentile(smudge_field, 80))
        smudge_mask = (smudge_field > smudge_threshold).astype(np.float32)
        soft_k = max(5, (min(h, w) // 30) | 1)
        smudge_mask = cv2.GaussianBlur(smudge_mask, (soft_k, soft_k), 0)
        smudge_mask = np.clip(smudge_mask, 0.0, 1.0)

        # 1. Local Gaussian blur (smudges letter edges). Moderate kernel
        #    so letter shapes are still recognisable inside the zone.
        blur_k = max(7, (min(h, w) // 14) | 1)
        img_blurred = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
        m3 = (smudge_mask * 0.85)[..., None]  # blend rather than full replace
        img = img * (1.0 - m3) + img_blurred * m3

        # 2. Moderate local ink fade — degraded but not erased. Even
        #    inside the most-damaged smudge zone, the model should still
        #    have enough ink residue to learn the letter shapes.
        gray_s = (img.mean(axis=2) / 255.0).astype(np.float32)
        ink_density_s = np.clip(
            (parchment_brightness - gray_s) / max(parchment_brightness, 0.01), 0.0, 1.0
        )
        local_fade = smudge_mask * (0.30 + 0.30 * ink_density_s) * random.uniform(0.80, 0.95)
        local_fade_3d = np.clip(local_fade, 0.0, 0.80)[..., None]
        img = img * (1.0 - local_fade_3d) + parchment_color * local_fade_3d

        # Refresh ink_density for the band fade below, so its ink-targeting
        # uses the post-smudge image.
        gray = (img.mean(axis=2) / 255.0).astype(np.float32)
        ink_density = np.clip(
            (parchment_brightness - gray) / max(parchment_brightness, 0.01), 0.0, 1.0
        )

    # Modulate the combined fade by ink density: parchment pixels get only
    # a tiny share of the fade (keeps the band continuity), ink pixels get
    # most of it (gets blended strongly toward parchment colour, which is
    # how the "rubbed-away letter fragments" effect arises).
    ink_targeted_fade = combined_fade * (0.10 + 0.90 * ink_density)
    fade_3d = ink_targeted_fade[..., None]
    img = img * (1.0 - fade_3d) + parchment_color * fade_3d

    # (Fold-groove hairline removed: at line-crop scale a narrow Gaussian
    #  darkening reads as a digital cut across the text — the band wear
    #  and abrasion spots already convey the fold without it.)

    # Subtle warm-yellow discoloration concentrated at the crease, weighted
    # by the base fade so it's strongest at the centre and falls off outward.
    discoloration = base_fade * random.uniform(0.10, 0.20)
    yellow_shift = np.array([1.0, 0.96, 0.85], dtype=np.float32)
    disc_3d = discoloration[..., None]
    img = img * (1.0 - disc_3d) + (img * yellow_shift) * disc_3d

    return img.clip(0, 255).astype(np.uint8)


def apply_augmentation_techniques(input_image, parchment_files, bleed_source_files=None, seed=None):
    """
    The pipeline:
      1. Ink degradation — morphological erosion/dilation + PixelDropout.
      2. Substrate swap — translucent multiplicative ink composited onto a
         random real parchment crop with optional verso bleed-through.
      3. Tonal jitter — warm-only HueSaturationValue.
      4. Page warp — elastic distortion + small shift/rotate.
      5. Scan capture — Gaussian blur, Gaussian noise, plasma brightness.

    Args:
        input_image: Rendered glyph image (RGB, uint8).
        parchment_files: Non-empty list of parchment crop paths, typically
            produced by `get_parchment_crops`.
        seed: Optional RNG seed for reproducibility.

    Returns:
        dict with keys 'image' (augmented array) and 'replay' (which
        transforms fired and the values they sampled — see ReplayCompose).
    """
    assert parchment_files, "parchment_files is empty — run get_parchment_crops first"

    if seed is not None:
        # Seed the global RNGs so both the custom Lambda (uses random + numpy)
        # and Albumentations' internal RNG (derives from the global state when
        # no explicit RNG is set) are reproducible.
        random.seed(seed)
        np.random.seed(seed)

    # Bind parchment_files into composite_on_parchment so it matches the
    # signature A.Lambda expects: fn(image=..., **albumentations_kwargs).
    bound_composite = functools.partial(composite_on_parchment, parchment_files=parchment_files)
    # Bind a (possibly empty) bleed-source list into apply_ink_bleed.
    bound_ink_bleed = functools.partial(apply_ink_bleed, bleed_source_files=bleed_source_files)

    transform_real_bg = A.ReplayCompose(
        [
            # 1. Ink degradation BEFORE composite. Strongly bias toward
            #    erosion — the rendered Brokenscript font is uniformly thick,
            #    while real quill strokes have hairline thins, especially
            #    on upstrokes. Erosion brings the synthetic glyphs closer
            #    to the delicate quill character of the reference scans.
            A.Morphological(scale=(1, 2), operation="dilation", p=0.25),
            A.Morphological(scale=(1, 3), operation="erosion", p=0.85),
            A.PixelDropout(dropout_prob=0.02, drop_value=255, p=0.5),
            # 2. Substrate swap with translucent ink + bleed-through (custom).
            A.Lambda(image=bound_composite, name="composite_on_parchment", p=1.0),
            # 2a. Centuries-of-aging effects: iron-gall ink browning,
            #     foxing spots, micro-pitting around dense ink. Each
            #     sub-effect has its own internal probability.
            A.Lambda(image=apply_aged_parchment_effects, name="aged_parchment", p=0.7),
            # 2b. Whole-line ink bleed (ink diffused into parchment fibres,
            #     letter edges fuzzed). Distinct phenomenon from rubbed-fold
            #     damage — letters stay mostly readable but visibly "wet".
            A.Lambda(image=bound_ink_bleed, name="ink_bleed", p=0.15),
            # 2c. Hard damage (heavy verso bleed + uneven tone + yellow tint).
            #     Fires on a minority of samples so the HTR model sees a mix
            #     of clean and severely damaged folios.
            A.Lambda(image=apply_page_creases, name="page_creases", p=0.40),
            # 2d. Torn / ragged edge on top, bottom, or both — fills the "tear"
            #     with a near-black void. DISABLED (p=0.0, 2026-07-30): the real
            #     line crops have no black torn borders, so this produced a
            #     synth-only artefact (jagged black bands) absent from the target
            #     distribution. Kept in code (reversible) but off. See spec §6.5.18.
            A.Lambda(image=apply_torn_edges, name="torn_edges", p=0.0),
            # 3. Tonal jitter — warm direction only (no pink/magenta).
            A.HueSaturationValue(
                hue_shift_limit=(0, 8),
                sat_shift_limit=(-15, 5),
                val_shift_limit=(-15, 0),
                p=0.7,
            ),
            # 4. Page warp
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=50,
                        sigma=5,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1.0,
                    ),
                    A.ElasticTransform(
                        alpha=120,
                        sigma=12,
                        border_mode=cv2.BORDER_REPLICATE,
                        p=1.0,
                    ),
                ],
                p=0.7,
            ),
            A.Affine(
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                scale=1.0,
                rotate=(-2.5, 2.5),
                border_mode=cv2.BORDER_REPLICATE,
                p=1.0,  # always-on — every real scan has slight rotation
            ),
            # 5. Scan capture — Affine, blur, and noise are always-on because
            #    every real scan exhibits at least slight rotation, defocus,
            #    and sensor noise. These three guarantee an "unlucky" seed
            #    still produces output that looks like a real scanned page.
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.012, 0.028), p=1.0),
            A.PlasmaBrightnessContrast(
                brightness_range=(-0.15, 0.05),
                contrast_range=(-0.1, 0.1),
                p=0.7,
            ),
        ],
    )
    if seed is not None:
        # Albumentations' per-transform RNG isn't tied to the global random
        # state, so set_random_seed must be called separately for the Compose
        # output to match between runs.
        transform_real_bg.set_random_seed(seed)

    return transform_real_bg(image=input_image)


def batch_augment_directory(
    input_dir: Path,
    output_dir: Path,
    run_name: str,
    parchment_files: list[Path],
    n_augmentations: int = 1,
    seed: int | None = None,
    logs_dir: str | Path | None = None,
    sample_size: int | None = None,
):
    """Apply the augmentation pipeline to every image in `input_dir`.
    Each call uses a derived per-image-per-variant seed, so the entire
    batch is reproducible when `seed` is set. When `seed` is None, every call
    is non-deterministic.

    Args:
        sample_size: When None (the default), every source image is processed.
            When set to a positive integer, only the first `sample_size`
            images are taken — useful for quickly previewing what the
            pipeline produces without writing the whole dataset.

    Note: output samples inherit the size of the *input renders*. Real crops
    are ~400×39 px, so the renderer (``medieval_text_generation``, font_size
    ~24 / margin ~7) is what sets the target size — the augmentation preserves
    it. See spec §6.5.18.

    Returns:
        List of output Paths actually saved.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    assert parchment_files, "parchment_files is empty — run get_parchment_crops first"

    # Logger setup — file + console if logs_dir, console only otherwise.
    if logs_dir:
        logger, log_file = setup_augmentation_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("augmentation")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        log_file = None

    logger.info(f"=== Augmentation Started | Run: {run_name} ===")

    image_paths = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    assert image_paths, f"No images found in {input_dir}"

    # The full source-image pool is also used as the ink-bleed source
    # pool: every other rendered line is a candidate "verso text" that
    # can show through the current line as bleed-through. Captured here
    # before sample_size trimming so the bleed pool retains variety even
    # in preview mode.
    bleed_source_files = list(image_paths)

    # Sampling mode: keep only the first N source images for a quick
    # preview run. Useful for verifying the pipeline produces what you
    # expect before kicking off the full batch.
    if sample_size is not None and sample_size > 0:
        image_paths = image_paths[:sample_size]
        logger.info(
            f"Sample mode: limited to {len(image_paths)} source image(s) "
            f"(requested sample_size={sample_size})"
        )

    # Config summary — everything needed to reproduce the run later.
    config_summary = {
        "run": run_name,
        "git": _get_git_commit(),
        "input_dir": str(input_dir),
        "output_dir": str(save_dir),
        "n_sources": len(image_paths),
        "n_augmentations": n_augmentations,
        "total_expected": len(image_paths) * n_augmentations,
        "base_seed": seed,
        "sample_size": sample_size,
        "parchment_dir": str(parchment_files[0].parent) if parchment_files else None,
        "n_parchment_files": len(parchment_files),
    }
    logger.info(f"Config: {json.dumps(config_summary)}")

    saved: list[Path] = []
    n_skipped = 0
    log_entries: dict[str, dict] = {}
    progress = tqdm(image_paths, total=len(image_paths), desc="Augmenting", unit="src")
    for i, img_path in enumerate(progress):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path.name}")
            n_skipped += 1
            progress.set_postfix(saved=len(saved), skipped=n_skipped)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for j in range(n_augmentations):
            call_seed = None if seed is None else seed + i * n_augmentations + j
            try:
                out = apply_augmentation_techniques(
                    img, parchment_files, bleed_source_files=bleed_source_files, seed=call_seed
                )
            except Exception as exc:
                logger.error(f"Augmentation failed for {img_path.name} variant {j}: {exc}")
                n_skipped += 1
                continue
            out_path = save_dir / f"{img_path.stem}_aug{j:02d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR))
            saved.append(out_path)

            # Record per-output augmentation details for the consolidated log.
            transforms = [_clean_transform_node(t) for t in out["replay"]["transforms"]]
            n_applied = sum(1 for t in transforms if t.get("applied"))
            log_entries[out_path.name] = {
                "source": img_path.name,
                "variant_index": j,
                "seed": call_seed,
                "n_applied": n_applied,
                "transforms": transforms,
            }

        progress.set_postfix(saved=len(saved), skipped=n_skipped)
    progress.close()

    # Consolidated JSON log for the whole run — one entry per saved image.
    log = {
        "summary": {
            **config_summary,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "saved": len(saved),
            "skipped": n_skipped,
        },
        "outputs": log_entries,
    }
    log_path = save_dir / "augmentation_log.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    logger.info(
        f"Augmentation complete: {len(saved)} saved / "
        f"{len(image_paths) * n_augmentations} expected ({n_skipped} skipped)"
    )
    logger.info(f"Output dir: {save_dir}")
    logger.info(f"Augmentation log (JSON): {log_path}")
    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return saved
