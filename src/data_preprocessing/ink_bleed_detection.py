"""Detect ink bleed in a folder of filtered line images.

Ink bleed in medieval manuscripts shows up two ways, both with the same
fingerprint in a grayscale image:

* **Bleed-through** — ink from the verso (back) of the page shows through
  as faint dark patches in the parchment background.
* **Ink spreading / feathering** — ink diffusing into the parchment
  fibres, softening character edges and adding mid-tone halos.

Both produce *non-uniform parchment* and *intermediate-intensity pixels*
that fall between the clean parchment peak and the ink peak of the
intensity histogram.

We quantify this with two metrics computed against Otsu's threshold:

1. ``bg_std_norm`` — standard deviation of the parchment pixels (those
   above Otsu), normalised by the available dynamic range
   ``255 - otsu_threshold``. High → patchy parchment → likely bleed.
2. ``intermediate_ratio`` — fraction of pixels in the band
   ``[otsu - 30, otsu]``. These are pixels darker than clean parchment
   but not solid ink. High → lots of mid-tones → likely bleed.

The two metrics are clipped at 0.30 (an empirically reasonable upper
bound for "clearly bleed"), scaled into ``[0, 1]``, and combined into a
single ``bleed_score`` via a weighted sum. ``has_bleed`` is
``bleed_score >= bleed_threshold`` (default 0.35; tune for the dataset).

The output JSON has both the boolean flag and the raw metrics so the
threshold can be re-calibrated without re-running detection.
"""

import datetime
import json
import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def setup_bleed_detection_logging(logs_dir: str | Path, run_name: str):
    """File + console logger, same pattern as the other src/ scripts."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"{run_name}_ink_bleed_detection.log"

    logger = logging.getLogger("ink_bleed_detection")
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
    """Short git SHA at PROJECT_ROOT, or 'unknown' if unavailable."""
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


def compute_bleed_metrics(gray: np.ndarray) -> dict:
    """Compute per-image metrics that correlate with ink bleed.

    Returns a dict with ``otsu_threshold``, ``bg_mean``, ``bg_std``,
    ``bg_std_norm``, ``bg_p5``, ``intermediate_ratio``, ``ink_fraction``,
    and ``degenerate`` (True if the image is all one tone).
    """
    otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bg_mask = gray > otsu_t
    fg_mask = ~bg_mask

    bg_pixels = gray[bg_mask]
    fg_pixels = gray[fg_mask]

    if bg_pixels.size == 0 or fg_pixels.size == 0:
        return {
            "otsu_threshold": float(otsu_t),
            "bg_mean": float(bg_pixels.mean()) if bg_pixels.size else 0.0,
            "bg_std": 0.0,
            "bg_std_norm": 0.0,
            "bg_p5": 0.0,
            "intermediate_ratio": 0.0,
            "ink_fraction": float(fg_mask.sum()) / float(gray.size),
            "degenerate": True,
        }

    bg_mean = float(bg_pixels.mean())
    bg_std = float(bg_pixels.std())
    bg_p5 = float(np.percentile(bg_pixels, 5))
    bg_std_norm = bg_std / max(255 - float(otsu_t), 1.0)

    band_lo = max(0, int(otsu_t) - 30)
    intermediate_mask = (gray >= band_lo) & (gray <= otsu_t)
    intermediate_ratio = float(intermediate_mask.sum()) / float(gray.size)

    ink_fraction = float(fg_mask.sum()) / float(gray.size)

    return {
        "otsu_threshold": float(otsu_t),
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "bg_std_norm": float(bg_std_norm),
        "bg_p5": bg_p5,
        "intermediate_ratio": intermediate_ratio,
        "ink_fraction": ink_fraction,
        "degenerate": False,
    }


def compute_bleed_score(
    metrics: dict, *, w_bg_std: float = 0.6, w_intermediate: float = 0.4
) -> float:
    """Composite ``[0, 1]`` score from the per-image metrics.

    Both ``bg_std_norm`` and ``intermediate_ratio`` are clipped at 0.30
    (empirically, values above this are clearly "bleed") then scaled
    into ``[0, 1]`` and combined via a weighted sum. ``w_bg_std`` defaults
    to 0.6 because background uniformity tends to be the stronger signal.
    """
    if metrics.get("degenerate"):
        return 0.0
    bg_score = min(1.0, metrics["bg_std_norm"] / 0.30)
    inter_score = min(1.0, metrics["intermediate_ratio"] / 0.30)
    return float(w_bg_std * bg_score + w_intermediate * inter_score)


def detect_ink_bleed(
    images_dir: str | Path,
    output_base_dir: str | Path,
    *,
    bleed_threshold: float = 0.35,
    w_bg_std: float = 0.6,
    w_intermediate: float = 0.4,
    logs_dir: str | Path | None = None,
) -> Path:
    """Score every image in ``images_dir`` for ink bleed; write a JSON.

    Args:
        images_dir: Directory of line images (``*.png``, ``*.jpg``,
            ``*.jpeg`` at top level).
        output_base_dir: Parent directory under which a
            ``bleed_detection_<timestamp>/`` subdirectory is created.
        bleed_threshold: ``has_bleed = bleed_score >= bleed_threshold``.
            Default 0.35; tune for your dataset by inspecting the score
            distribution in the output JSON.
        w_bg_std, w_intermediate: Weights for the two sub-scores. Must
            be non-negative; the function does NOT enforce they sum to 1.
        logs_dir: Optional plain-text run log directory.

    Returns:
        Path to the saved ``ink_bleed.json``.
    """
    images_dir = Path(images_dir)
    output_base_dir = Path(output_base_dir)
    assert images_dir.is_dir(), f"Images dir not found: {images_dir}"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bleed_detection_{timestamp}"
    # Write the JSON directly into output_base_dir (no timestamped
    # subdir) so it sits next to the other artefacts of the same
    # filtering run. The timestamp goes into the filename instead, so
    # multiple runs don't overwrite each other.
    output_base_dir.mkdir(parents=True, exist_ok=True)

    if logs_dir:
        logger, log_file = setup_bleed_detection_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("ink_bleed_detection")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        log_file = None

    logger.info(f"=== Ink-bleed detection started | Run: {run_name} ===")

    image_files = sorted(
        list(images_dir.rglob("*.png"))
        + list(images_dir.rglob("*.jpg"))
        + list(images_dir.rglob("*.jpeg"))
    )
    assert image_files, f"No image files (*.png, *.jpg) found under {images_dir}"

    config_summary = {
        "run": run_name,
        "git": _get_git_commit(),
        "images_dir": str(images_dir),
        "output_dir": str(output_base_dir),
        "bleed_threshold": bleed_threshold,
        "w_bg_std": w_bg_std,
        "w_intermediate": w_intermediate,
    }
    logger.info(f"Config: {json.dumps(config_summary, ensure_ascii=False)}")
    logger.info(f"Found {len(image_files)} images")

    per_image: dict[str, dict] = {}
    skipped: list[str] = []
    n_with_bleed = 0

    for img_path in tqdm(image_files, desc="Scoring", unit="img"):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # Key entries by the relative path from images_dir so nested
        # per-page subdirs don't collide and so a JSON consumer can
        # walk straight back to the file on disk.
        rel_key = str(img_path.relative_to(images_dir))
        if gray is None:
            skipped.append(rel_key)
            continue
        metrics = compute_bleed_metrics(gray)
        score = compute_bleed_score(metrics, w_bg_std=w_bg_std, w_intermediate=w_intermediate)
        has_bleed = bool(score >= bleed_threshold and not metrics.get("degenerate", False))
        per_image[rel_key] = {
            "has_bleed": has_bleed,
            "bleed_score": score,
            "metrics": metrics,
        }
        if has_bleed:
            n_with_bleed += 1

    scores = np.array([entry["bleed_score"] for entry in per_image.values()])
    score_summary = {
        "min": float(scores.min()) if scores.size else 0.0,
        "p25": float(np.percentile(scores, 25)) if scores.size else 0.0,
        "p50": float(np.percentile(scores, 50)) if scores.size else 0.0,
        "p75": float(np.percentile(scores, 75)) if scores.size else 0.0,
        "p95": float(np.percentile(scores, 95)) if scores.size else 0.0,
        "max": float(scores.max()) if scores.size else 0.0,
        "mean": float(scores.mean()) if scores.size else 0.0,
    }

    summary = {
        **config_summary,
        "n_images": len(image_files),
        "n_processed": len(per_image),
        "n_skipped": len(skipped),
        "skipped_files": skipped,
        "n_with_bleed": n_with_bleed,
        "fraction_with_bleed": n_with_bleed / max(len(per_image), 1),
        "bleed_score_distribution": score_summary,
    }

    out = {"summary": summary, "images": per_image}
    output_path = output_base_dir / f"ink_bleed_{timestamp}.json"
    output_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        f"Processed {len(per_image)} images "
        f"(with bleed: {n_with_bleed}, skipped: {len(skipped)})"
    )
    logger.info(f"Score distribution: {json.dumps(score_summary)}")
    logger.info(f"Output JSON: {output_path}")
    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return output_path
