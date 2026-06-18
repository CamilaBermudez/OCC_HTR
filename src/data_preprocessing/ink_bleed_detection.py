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

Both metrics are **min-max normalised across the run** (rescaled so the
worst image in the batch maps to 1.0 and the cleanest maps to 0.0) and
then combined into a single ``bleed_score`` via a weighted sum.

``has_bleed`` flags the top ``(100 - bleed_percentile)%`` of scores —
i.e. with the default ``bleed_percentile=75.0`` the top quarter of
images by bleed score is marked. The effective score cutoff is recorded
in the summary as ``effective_threshold`` so a downstream consumer can
reproduce the decision.

The output JSON has the boolean flag, the bleed score, the raw
per-image metrics, AND the per-run min/max used for normalisation, so
the threshold can be re-calibrated without re-running detection.
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


def _minmax(value: float, lo: float, hi: float) -> float:
    """Min-max rescale ``value`` from ``[lo, hi]`` into ``[0, 1]``, clipped."""
    span = hi - lo
    if span <= 0:
        return 0.0
    return float(np.clip((value - lo) / span, 0.0, 1.0))


def compute_bleed_score(
    bg_score: float,
    inter_score: float,
    *,
    w_bg_std: float = 0.6,
    w_intermediate: float = 0.4,
) -> float:
    """Composite ``[0, 1]`` score from pre-normalised sub-scores.

    Both inputs are expected to already live in ``[0, 1]`` (e.g. after
    min-max normalisation across the run). ``w_bg_std`` defaults to 0.6
    because background uniformity tends to be the stronger signal.
    """
    return float(w_bg_std * bg_score + w_intermediate * inter_score)


def detect_ink_bleed(
    images_dir: str | Path,
    output_base_dir: str | Path,
    *,
    bleed_percentile: float = 75.0,
    w_bg_std: float = 0.6,
    w_intermediate: float = 0.4,
    logs_dir: str | Path | None = None,
) -> Path:
    """Score every image in ``images_dir`` for ink bleed; write a JSON.

    The two per-image metrics (``bg_std_norm`` and ``intermediate_ratio``)
    are min-max normalised across the run before being combined into a
    composite ``bleed_score`` via a weighted sum. An image is flagged as
    ``has_bleed`` when its score is in the top ``(100 - bleed_percentile)%``
    of the run — i.e. the effective threshold is
    ``np.percentile(scores, bleed_percentile)``.

    Args:
        images_dir: Directory of line images (``*.png``, ``*.jpg``,
            ``*.jpeg`` at top level).
        output_base_dir: Parent directory under which a
            ``bleed_detection_<timestamp>/`` subdirectory is created.
        bleed_percentile: Score-distribution percentile that defines the
            cutoff. Default 75.0 (flag top 25% as bleed). 50.0 = half;
            90.0 = top 10% only. Always flags *something* — that's the
            nature of percentile thresholds.
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
        "bleed_percentile": bleed_percentile,
        "w_bg_std": w_bg_std,
        "w_intermediate": w_intermediate,
    }
    logger.info(f"Config: {json.dumps(config_summary, ensure_ascii=False)}")
    logger.info(f"Found {len(image_files)} images")

    # Pass 1: load each image and compute its raw metrics. We hold the
    # results in memory because the min-max normalisation in pass 2
    # needs the per-metric min/max across the whole run.
    raw_results: list[tuple[str, dict]] = []
    skipped: list[str] = []

    for img_path in tqdm(image_files, desc="Scoring", unit="img"):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        # Key entries by the relative path from images_dir so nested
        # per-page subdirs don't collide and so a JSON consumer can
        # walk straight back to the file on disk.
        rel_key = str(img_path.relative_to(images_dir))
        if gray is None:
            skipped.append(rel_key)
            continue
        raw_results.append((rel_key, compute_bleed_metrics(gray)))

    # Per-metric min/max across the run, excluding degenerate images
    # (single-tone images would skew the floor to 0 and pull every
    # real metric down).
    healthy = [m for _, m in raw_results if not m.get("degenerate")]
    if healthy:
        bg_vals = np.array([m["bg_std_norm"] for m in healthy])
        inter_vals = np.array([m["intermediate_ratio"] for m in healthy])
        bg_min, bg_max = float(bg_vals.min()), float(bg_vals.max())
        inter_min, inter_max = float(inter_vals.min()), float(inter_vals.max())
    else:
        bg_min = bg_max = inter_min = inter_max = 0.0

    metric_normalization = {
        "bg_std_norm": {"min": bg_min, "max": bg_max},
        "intermediate_ratio": {"min": inter_min, "max": inter_max},
    }

    # Pass 2: rescale via min-max, compute composite scores. Degenerate
    # images keep score 0 so they never fall above the percentile cutoff.
    per_image: dict[str, dict] = {}
    for rel_key, metrics in raw_results:
        if metrics.get("degenerate"):
            score = 0.0
        else:
            bg_sub = _minmax(metrics["bg_std_norm"], bg_min, bg_max)
            inter_sub = _minmax(metrics["intermediate_ratio"], inter_min, inter_max)
            score = compute_bleed_score(
                bg_sub, inter_sub, w_bg_std=w_bg_std, w_intermediate=w_intermediate
            )
        per_image[rel_key] = {
            "bleed_score": score,
            "metrics": metrics,
        }

    scores = np.array([entry["bleed_score"] for entry in per_image.values()])
    # Percentile-derived threshold. We compute over the FULL score
    # distribution (including degenerate 0s); the alternative — only
    # healthy scores — would silently raise the bar when a run contains
    # many blank images.
    if scores.size:
        effective_threshold = float(np.percentile(scores, bleed_percentile))
    else:
        effective_threshold = 0.0

    n_with_bleed = 0
    for entry in per_image.values():
        # Strict inequality at the boundary: ties don't get flagged.
        # With a percentile this is mostly cosmetic but matters for tiny
        # runs where many scores collapse to the same value.
        is_degenerate = entry["metrics"].get("degenerate", False)
        has_bleed = bool(entry["bleed_score"] > effective_threshold and not is_degenerate)
        entry["has_bleed"] = has_bleed
        if has_bleed:
            n_with_bleed += 1

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
        "effective_threshold": effective_threshold,
        "metric_normalization": metric_normalization,
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
