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
    seed: int = 0,
    plot_: bool = False,
):
    """Extract low-edge ("empty parchment") crops from manuscript pages.
    For each image, sample random square windows, score them by
    mean Canny edge density. Lower scores indicate emptier parchment(better).
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
        for _ in range(candidates_page):
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            crop = page[y : y + crop_size, x : x + crop_size]
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
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

    nh, nw = max(8, h // 15), max(8, w // 15)
    coarse = np.random.rand(nh, nw).astype(np.float32)
    noise = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.clip(noise * 1.3 + 0.1, 0.4, 1.0)
    ink = ink * noise

    bleed = np.zeros_like(ink)
    if random.random() < 0.6:
        ghost = cv2.flip(ink, 1)
        ghost = cv2.GaussianBlur(ghost, (0, 0), sigmaX=4.0)
        bleed = ghost * 0.22

    ink = ink[..., None]
    bleed = bleed[..., None]

    ink_mult = np.array([0.27, 0.22, 0.17], dtype=np.float32)
    bleed_mult = np.array([0.88, 0.83, 0.72], dtype=np.float32)
    out = bg_crop * (bleed_mult**bleed) * (ink_mult**ink)
    return out.clip(0, 255).astype(np.uint8)


def apply_augmentation_techniques(input_image, parchment_files, seed=None):
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

    transform_real_bg = A.ReplayCompose(
        [
            # 1. Ink degradation BEFORE composite. Bias toward erosion —
            #    real quill strokes have hairline thins the rendered font lacks.
            A.Morphological(scale=(1, 2), operation="dilation", p=0.5),
            A.Morphological(scale=(1, 3), operation="erosion", p=0.7),
            A.PixelDropout(dropout_prob=0.02, drop_value=255, p=0.5),
            # 2. Substrate swap with translucent ink + bleed-through (custom).
            A.Lambda(image=bound_composite, name="composite_on_parchment", p=1.0),
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
                    A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                    A.ElasticTransform(alpha=120, sigma=12, p=1.0),
                ],
                p=0.7,
            ),
            A.Affine(
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                scale=1.0,
                rotate=(-4, 4),
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
):
    """Apply the augmentation pipeline to every image in `input_dir`.
    Each call uses a derived per-image-per-variant seed, so the entire
    batch is reproducible when `seed` is set. When `seed` is None, every call
    is non-deterministic.

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
        "parchment_dir": str(parchment_files[0].parent) if parchment_files else None,
        "n_parchment_files": len(parchment_files),
    }
    logger.info(f"Config: {json.dumps(config_summary)}")

    saved: list[Path] = []
    n_skipped = 0
    log_entries: dict[str, dict] = {}
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Skipping unreadable image: {img_path.name}")
            n_skipped += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for j in range(n_augmentations):
            call_seed = None if seed is None else seed + i * n_augmentations + j
            try:
                out = apply_augmentation_techniques(img, parchment_files, seed=call_seed)
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

        # Progress every 20 source images (or on the final image).
        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            logger.info(
                f"Progress: {i + 1}/{len(image_paths)} sources | "
                f"saved: {len(saved)} | skipped: {n_skipped}"
            )

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
