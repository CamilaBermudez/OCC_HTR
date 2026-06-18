import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data_preprocessing.image_segmentation import (
    _find_column_spans_by_projection,
    _line_baseline_midpoint,
)


def setup_simple_logging(logs_dir: str, run_name: str | None = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(logs_dir) / f"{run_name}_filtering.log"

    logger = logging.getLogger("filtering")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    for h in [handler, console]:
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger, str(log_file)


def log_filter_funnel(
    logger,
    bounds_density_down,
    bounds_density_up,
    bounds_size_down,
    bounds_size_up,
    long_imgs,
    clean_img,
    img_remove_keys,
):
    def get_key(t):
        return (t[0], t[1])  # t = (folder, stem, density, size, height, width)

    density_low = {get_key(t) for t in bounds_density_down}
    density_high = {get_key(t) for t in bounds_density_up}
    density_all = density_low | density_high

    size_low = {get_key(t) for t in bounds_size_down}
    size_high = {get_key(t) for t in bounds_size_up}
    size_all = size_low | size_high

    long_set = {get_key(t) for t in long_imgs}
    clean_set = {get_key(t) for t in clean_img}

    density_size = density_all & size_all
    density_long = density_all & long_set
    size_long = size_all & long_set
    all_three = density_all & size_all & long_set

    to_remove = img_remove_keys

    logger.info("Filter Breakdown:")
    logger.info(
        f"    Density outliers: {len(density_all)} (low: {len(density_low)}, high: {len(density_high)})"
    )
    logger.info(
        f"    Size outliers:    {len(size_all)} (low: {len(size_low)}, high: {len(size_high)})"
    )
    logger.info(f"    Long images:      {len(long_set)}")
    logger.info(f"    Clean/kept:       {len(clean_set)}")

    logger.info(" Intersections:")
    logger.info(f"   Density ∩ Size:     {len(density_size)}")
    logger.info(f"   Density ∩ Long:     {len(density_long)}")
    logger.info(f"   Size ∩ Long:        {len(size_long)}")
    logger.info(f"   All three:          {len(all_three)}")

    logger.info(" Final Removal:")
    logger.info(f"   Unique to remove:   {len(to_remove)}")

    # ASCII funnel visualization
    total = len(density_all | size_all | long_set | clean_set)
    if total > 0:
        funnel = [
            (" Total analyzed", total),
            (" Density filter", len(density_all)),
            (" Size filter", len(size_all)),
            (" Long filter", len(long_set)),
            (" Final removal", len(to_remove)),
            (" Kept", total - len(to_remove)),
        ]
        max_val = max(v for _, v in funnel)
        logger.info(" Funnel:")
        for label, count in funnel:
            bar = "█" * int(40 * count / max_val) if max_val > 0 else ""
            logger.info(f"   {label:20s} {count:5d} {bar}")


def calculate_text_density(image: np.ndarray | str | Path) -> tuple[float, int, int, int]:
    if isinstance(image, str | Path):
        img = cv.imread(str(image), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image}")
    else:
        img = image.copy()

    total_pixels = img.size
    height, width = img.shape
    if total_pixels == 0:
        return 0.0, 0, height, width

    text_pixels = np.count_nonzero(img == 0)
    return float(text_pixels / total_pixels), total_pixels, height, width


def process_image_folder(input_path: str | Path) -> list[dict]:
    """
    Returns: List of dicts with folder name, total lines and list of (density, size) tuples
    """
    input_path = Path(input_path)
    image_folders = [f for f in input_path.iterdir() if f.is_dir()]
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    glb_density = []

    for img_folder in tqdm(image_folders, desc="Processing folders", unit="folder"):
        image_files = sorted(
            [f for f in img_folder.iterdir() if f.suffix.lower() in image_extensions],
            key=lambda x: x.stem,
        )
        folder_density = []

        for img_path in tqdm(
            image_files, desc=f"Images in {img_folder.name}", unit="file", leave=False
        ):
            try:
                density, size, height, width = calculate_text_density(image=img_path)
                folder_density.append((density, size, height, width))
            except Exception:
                continue
        if folder_density:
            glb_density.append(
                {
                    "folder": img_folder.name,
                    "total_lines": len(folder_density),
                    "density": folder_density,
                }
            )

    return glb_density


def descr_statistics_list(
    density_size_data: list[dict], type: str, percentiles: list[float]
) -> pd.Series:
    """Extract either density or size values and compute descriptive statistics."""
    idx = 0 if type == "density" else 1
    full_list = [item[idx] for sublist in density_size_data for item in sublist["density"]]
    return pd.Series(full_list).describe(percentiles=percentiles)


def _threshold_to_percentile_key(threshold: float) -> str:
    pct = threshold * 100
    return f"{int(pct)}%" if pct == int(pct) else f"{pct}%"


def filter_based_on_thresholds(
    density_size_data: list[dict],
    density_thresholds: list[float],
    size_thresholds: list[float],
    stats_density: pd.Series,
    stats_size: pd.Series,
    src_dir: Path,
) -> tuple[list[tuple], list[tuple], list[tuple], list[tuple], list[tuple], list[tuple]]:
    bounds_density_down, bounds_density_up = [], []
    bounds_size_down, bounds_size_up = [], []
    long_imgs, clean_img = [], []

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    folder_files = {}
    for sublist in density_size_data:
        folder_path = src_dir / sublist["folder"]
        if folder_path.is_dir():
            folder_files[sublist["folder"]] = sorted(
                [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions],
                key=lambda x: x.stem,
            )

    density_keys = {}
    if len(density_thresholds) >= 1:
        density_keys["low"] = _threshold_to_percentile_key(density_thresholds[0])
    if len(density_thresholds) == 2:
        density_keys["high"] = _threshold_to_percentile_key(density_thresholds[1])

    size_keys = {}
    if len(size_thresholds) >= 1:
        size_keys["low"] = _threshold_to_percentile_key(size_thresholds[0])
    if len(size_thresholds) == 2:
        size_keys["high"] = _threshold_to_percentile_key(size_thresholds[1])

    for sublist in density_size_data:
        folder_name = sublist["folder"]
        image_files = folder_files.get(folder_name, [])

        if not image_files:
            continue

        for j, (density, size, height, width) in enumerate(sublist["density"]):
            if j >= len(image_files):
                continue

            stem = image_files[j].stem
            _tuple = (folder_name, stem, density, size, height, width)

            if "low" in density_keys and density <= stats_density[density_keys["low"]]:
                bounds_density_down.append(_tuple)
            if "high" in density_keys and density >= stats_density[density_keys["high"]]:
                bounds_density_up.append(_tuple)

            if "low" in size_keys and size <= stats_size[size_keys["low"]]:
                bounds_size_down.append(_tuple)
            if "high" in size_keys and size >= stats_size[size_keys["high"]]:
                bounds_size_up.append(_tuple)

            if height > width:
                long_imgs.append(_tuple)
            else:
                clean_img.append(_tuple)

    return (
        bounds_density_down,
        bounds_density_up,
        bounds_size_down,
        bounds_size_up,
        long_imgs,
        clean_img,
    )


def move_files_to_timestamped_folder(
    src_dir: Path, dst_base_dir: Path, list_img_remove: set[tuple], file_type: str, timestamp: str
) -> tuple[int, int]:
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    exclude_keys = {(item[0], item[1]) for item in list_img_remove}

    copied, removed = 0, 0
    src_files = [f for f in src_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts]

    for src_file in src_files:
        file_key = (src_file.parent.name, src_file.stem)
        folder = "removed" if file_key in exclude_keys else "kept"

        rel_path = src_file.relative_to(src_dir)
        dst_file = dst_base_dir / timestamp / file_type / folder / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

        if folder == "removed":
            removed += 1
        else:
            copied += 1

    return copied, removed


def generate_analysis_df(
    dst_dir: Path,
    timestamp: str,
    bounds_density_down,
    bounds_density_up,
    bounds_size_down,
    bounds_size_up,
    long_imgs,
    clean_img,
) -> pd.DataFrame:
    def get_key(t):
        return t[0], t[1], t[2], t[3], t[4], t[5]

    doc_reasons = defaultdict(set)
    for t in bounds_density_down:
        doc_reasons[get_key(t)].add("density_low")
    for t in bounds_density_up:
        doc_reasons[get_key(t)].add("density_high")
    for t in bounds_size_down:
        doc_reasons[get_key(t)].add("size_low")
    for t in bounds_size_up:
        doc_reasons[get_key(t)].add("size_high")
    for t in long_imgs:
        doc_reasons[get_key(t)].add("long_img")
    for t in clean_img:
        doc_reasons[get_key(t)].add("clean")

    density_keys = set(get_key(t) for t in bounds_density_down) | set(
        get_key(t) for t in bounds_density_up
    )
    size_keys = set(get_key(t) for t in bounds_size_down) | set(get_key(t) for t in bounds_size_up)
    bounds_density_size_keys = density_keys & size_keys
    long_imgs_keys = set(get_key(t) for t in long_imgs)

    img_remove_keys = long_imgs_keys | bounds_density_size_keys

    df_all = pd.DataFrame(
        [
            {
                "folder": k[0],
                "stem": k[1],
                "density": k[2],
                "size": k[3],
                "height": k[4],
                "width": k[5],
                "reasons": list(v),
                "was_removed": k in img_remove_keys,
            }
            for k, v in doc_reasons.items()
        ]
    )

    output_path = dst_dir / timestamp / "filter_tracking.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)

    return df_all


def filter_noisy_lines(
    src_dir: Path,
    dst_dir: Path,
    size_thresholds: list[float] | None = None,
    density_thresholds: list[float] | None = None,
    export_tracking: bool = True,
    timestamp: str | None = None,
) -> tuple[set[tuple], pd.DataFrame]:
    if size_thresholds is None:
        size_thresholds = [0.05]
    if density_thresholds is None:
        density_thresholds = [0.001, 0.99]
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    density_size_data = process_image_folder(src_dir)
    stats_density = descr_statistics_list(density_size_data, "density", density_thresholds)
    stats_size = descr_statistics_list(density_size_data, "size", size_thresholds)

    bounds = filter_based_on_thresholds(
        density_size_data=density_size_data,
        density_thresholds=density_thresholds,
        size_thresholds=size_thresholds,
        stats_density=stats_density,
        stats_size=stats_size,
        src_dir=src_dir,
    )
    (
        bounds_density_down,
        bounds_density_up,
        bounds_size_down,
        bounds_size_up,
        long_imgs,
        clean_img,
    ) = bounds

    def get_key(t):
        return (t[0], t[1])

    bounds_density_ = {get_key(t) for t in bounds_density_down} | {
        get_key(t) for t in bounds_density_up
    }
    bounds_size_ = {get_key(t) for t in bounds_size_down} | {get_key(t) for t in bounds_size_up}
    bounds_density_size_ = bounds_size_ & bounds_density_
    long_imgs_set = {get_key(t) for t in long_imgs}

    img_remove = long_imgs_set | bounds_density_size_

    filter_counts = {
        "bounds_density_down": bounds_density_down,
        "bounds_density_up": bounds_density_up,
        "bounds_size_down": bounds_size_down,
        "bounds_size_up": bounds_size_up,
        "long_imgs": long_imgs,
        "clean_img": clean_img,
    }

    df_tracking = pd.DataFrame()
    if export_tracking:
        df_tracking = generate_analysis_df(
            dst_dir=dst_dir,
            timestamp=timestamp,
            bounds_density_down=bounds_density_down,
            bounds_density_up=bounds_density_up,
            bounds_size_down=bounds_size_down,
            bounds_size_up=bounds_size_up,
            long_imgs=long_imgs,
            clean_img=clean_img,
        )

    return img_remove, df_tracking, filter_counts


_LINE_FILENAME_RE = re.compile(r"^(.+)_line_(\d+(?:p\d+)?)$")


def _parse_line_filename(stem: str) -> tuple[str, float] | None:
    """Parse '<page>_line_<id>' → (page, id_as_float) or None.

    Supports the existing integer convention (line_24) and the
    fractional convention we introduce here for split right-halves
    (line_24p5 → 24.5). 'p' is used as the decimal separator because
    a literal '.' would break the trailing '.png' parser and a '_'
    would conflict with how an underscored multi-token page name
    would look.
    """
    m = _LINE_FILENAME_RE.match(stem)
    if not m:
        return None
    page, idx_str = m.group(1), m.group(2)
    if "p" in idx_str:
        whole, frac = idx_str.split("p", 1)
        idx = float(whole) + float(f"0.{frac}")
    else:
        idx = float(idx_str)
    return page, idx


def _format_split_line_index(idx_float: float) -> str:
    """Format e.g. 24.5 → '24p5' for the right-half filename."""
    whole = int(idx_float)
    frac = idx_float - whole
    # Strip trailing zeros from the fractional part: 0.50 → '5', not '50'.
    frac_str = f"{frac:.6f}".split(".")[1].rstrip("0") or "0"
    return f"{whole}p{frac_str}"


def _column_of_each_line(json_data: dict, page_width: int) -> list[int]:
    """Return a per-line list of column indices.

    Uses the same projection-based column detection that runs at
    segmentation time, so the column boundaries here match whatever
    reorder_lines_reading_order saw when it numbered the lines in
    reading order.
    """
    lines = json_data.get("lines", []) or []
    if not lines:
        return []
    line_ranges: list[tuple[int, int]] = []
    for line in lines:
        bl = line.get("baseline")
        if isinstance(bl, list) and len(bl) >= 2:
            line_ranges.append((int(bl[0][0]), int(bl[-1][0])))
        else:
            bnd = line.get("boundary")
            if isinstance(bnd, list) and bnd:
                xs = [int(p[0]) for p in bnd]
                line_ranges.append((min(xs), max(xs)))
            else:
                line_ranges.append((0, 0))
    spans = _find_column_spans_by_projection(line_ranges, page_width)
    if not spans:
        return [0] * len(lines)
    column_midpoints = [(left + right) / 2.0 for left, right in spans]
    column_of_line: list[int] = []
    for line in lines:
        cx, _ = _line_baseline_midpoint(line)
        nearest = 0
        best = abs(cx - column_midpoints[0])
        for i, mid in enumerate(column_midpoints[1:], start=1):
            d = abs(cx - mid)
            if d < best:
                best = d
                nearest = i
        column_of_line.append(nearest)
    return column_of_line


def _split_wide_image_pair(img_paths: list[Path], right_half_stem: str) -> int:
    """Split each image in `img_paths` at its horizontal midpoint.

    Left half overwrites the original file; right half is written next
    to it with stem `right_half_stem`. Returns the number of files
    actually split (some paths may not exist in every kept folder).
    """
    n_split = 0
    for img_path in img_paths:
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
            mid = w // 2
            left = im.crop((0, 0, mid, h))
            right = im.crop((mid, 0, w, h))
        right_path = img_path.parent / f"{right_half_stem}{img_path.suffix}"
        # Save right first so we can't end up with only the overwritten
        # left half if something fails halfway through.
        right.save(right_path, optimize=True)
        left.save(img_path, optimize=True)
        n_split += 1
    return n_split


def split_wide_kept_images(
    binarised_kept_dir: Path,
    original_kept_dir: Path | None,
    kraken_json_dir: Path,
    *,
    percentile: float = 99.0,
    min_ratio_to_median: float = 1.5,
    logger: logging.Logger | None = None,
) -> dict:
    """Detect and split unusually-wide line crops in the kept folders.

    Wide lines are usually kraken merging the top row of two adjacent
    columns into a single record. We detect them by width statistics
    (above ``percentile`` AND above ``min_ratio_to_median`` × the
    median), then for each one look up the corresponding kraken JSON
    to determine column structure, split the image at its midpoint,
    and save a right half with a fractional reading-order index that
    sorts between the last index of the current column and the first
    index of the next column (e.g. between 24 and 25 → 24p5).

    Skips lines that already live in the rightmost column (no next
    column to give the right half a 'between' position) or that come
    from a single-column page.
    """
    log = logger or logging.getLogger("filtering")
    if not binarised_kept_dir.is_dir():
        log.warning(f"split_wide: kept dir not found: {binarised_kept_dir}")
        return {"n_wide": 0, "n_split": 0, "n_skipped": 0}

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    kept_files = [
        f for f in binarised_kept_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts
    ]
    if not kept_files:
        return {"n_wide": 0, "n_split": 0, "n_skipped": 0}

    widths: list[int] = []
    for p in kept_files:
        try:
            with Image.open(p) as im:
                widths.append(im.size[0])
        except Exception:
            widths.append(0)
    widths_arr = np.asarray(widths)
    median = float(np.median(widths_arr))
    p_cutoff = float(np.percentile(widths_arr, percentile))
    floor = max(p_cutoff, min_ratio_to_median * median)
    log.info(
        f"Wide-line detection: median width={median:.0f}, "
        f"p{percentile:g}={p_cutoff:.0f}, effective threshold={floor:.0f}"
    )

    candidates = [(p, w) for p, w in zip(kept_files, widths, strict=False) if w > floor]
    log.info(f"Found {len(candidates)} wide candidates above the threshold")

    # Cache per-page (column_of_line, lines, page_width) to avoid re-reading
    # each page's JSON for every wide line on it.
    page_cache: dict[str, tuple[list[int], list[dict], int]] = {}

    n_split = 0
    n_skipped = 0
    for img_path, width in candidates:
        parsed = _parse_line_filename(img_path.stem)
        if parsed is None:
            log.warning(f"split_wide: cannot parse line filename: {img_path.name}")
            n_skipped += 1
            continue
        page_name, line_idx_float = parsed
        if not float(line_idx_float).is_integer():
            # A previously-split right half — never split again.
            n_skipped += 1
            continue
        line_idx = int(line_idx_float)

        if page_name not in page_cache:
            json_path = kraken_json_dir / f"{page_name}.json"
            if not json_path.exists():
                log.warning(f"split_wide: no JSON for {page_name} ({json_path})")
                n_skipped += 1
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"split_wide: cannot read JSON for {page_name}: {e}")
                n_skipped += 1
                continue
            lines_in_json = data.get("lines", []) or []
            # Page width proxy: rightmost line edge + small buffer; we
            # don't have the source image at hand and the projection
            # algorithm only needs an upper-bound width.
            max_x = 0
            for line in lines_in_json:
                bl = line.get("baseline")
                if isinstance(bl, list) and len(bl) >= 2:
                    max_x = max(max_x, int(bl[-1][0]))
            page_width = max_x + 100 if max_x else 1
            col_of_line = _column_of_each_line(data, page_width)
            page_cache[page_name] = (col_of_line, lines_in_json, page_width)

        col_of_line, lines_in_json, _ = page_cache[page_name]
        if line_idx >= len(col_of_line):
            log.warning(
                f"split_wide: line index {line_idx} out of range "
                f"for page {page_name} ({len(col_of_line)} lines)"
            )
            n_skipped += 1
            continue

        current_col = col_of_line[line_idx]
        n_cols = max(col_of_line) + 1 if col_of_line else 1
        if n_cols < 2:
            n_skipped += 1
            continue
        if current_col >= n_cols - 1:
            # Wide line is in the rightmost column; no next column to
            # place the right half into.
            log.info(f"split_wide: skipping {img_path.name} — already in rightmost column")
            n_skipped += 1
            continue

        last_of_current = max(i for i, c in enumerate(col_of_line) if c == current_col)
        first_of_next = min(i for i, c in enumerate(col_of_line) if c == current_col + 1)
        right_idx = (last_of_current + first_of_next) / 2.0
        right_stem = f"{page_name}_line_{_format_split_line_index(right_idx)}"

        # Also split the matching file in the original (color) kept dir
        # if present, so the two staged outputs stay in lockstep.
        paths_to_split = [img_path]
        if original_kept_dir is not None:
            try:
                rel = img_path.relative_to(binarised_kept_dir)
                paths_to_split.append(original_kept_dir / rel)
            except ValueError:
                pass

        n_done = _split_wide_image_pair(paths_to_split, right_stem)
        if n_done:
            n_split += 1
            log.info(
                f"split: {img_path.name} (w={width}) → "
                f"left={img_path.stem}, right={right_stem}  "
                f"(column {current_col}, right-half index {right_idx})"
            )

    return {
        "n_wide": len(candidates),
        "n_split": n_split,
        "n_skipped": n_skipped,
        "threshold": float(floor),
        "median_width": median,
        "p_cutoff": p_cutoff,
    }


def run_filtering_pipeline(
    binarized_src: Path,
    extracted_src: Path,
    dst_base_dir: Path,
    logs_dir: str | None = None,
    run_name: str | None = None,
    size_thresholds: list[float] | None = None,
    density_thresholds: list[float] | None = None,
    kraken_json_dir: Path | None = None,
    wide_percentile: float = 99.0,
    wide_min_ratio_to_median: float = 1.5,
) -> dict:
    if size_thresholds is None:
        size_thresholds = [0.03]
    if density_thresholds is None:
        density_thresholds = [0.001, 0.997]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"filter_{timestamp}"

    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Filtering Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("filtering")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
            )
        log_file = None

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.environ.get("PROJECT_ROOT", "."),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    if not binarized_src.is_dir():
        logger.error(f"Binarized source folder not found: {binarized_src}")
        return {"success": False}

    config_summary = {
        "run": run_name,
        "git": git_commit,
        "binarized_input": str(binarized_src.name),
        "extracted_input": str(extracted_src.name),
        "output_base": str(dst_base_dir.name),
        "timestamp": timestamp,
        "density_thresholds": density_thresholds,
        "size_thresholds": size_thresholds,
    }
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")

    run_output_dir = dst_base_dir / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {run_output_dir}")

    logger.info("Analyzing images for filtering...")
    img_remove, df_tracking, filter_counts = filter_noisy_lines(
        src_dir=binarized_src,
        dst_dir=dst_base_dir,
        size_thresholds=size_thresholds,
        density_thresholds=density_thresholds,
        export_tracking=True,
        timestamp=timestamp,
    )

    total_analyzed = len(df_tracking) if not df_tracking.empty else 0

    if not df_tracking.empty:
        log_filter_funnel(
            logger=logger,
            bounds_density_down=filter_counts["bounds_density_down"],
            bounds_density_up=filter_counts["bounds_density_up"],
            bounds_size_down=filter_counts["bounds_size_down"],
            bounds_size_up=filter_counts["bounds_size_up"],
            long_imgs=filter_counts["long_imgs"],
            clean_img=filter_counts["clean_img"],
            img_remove_keys=img_remove,
        )

    marked_removed = df_tracking["was_removed"].sum() if not df_tracking.empty else 0
    logger.info(
        f"Analysis complete: {total_analyzed} images | Marked for removal: {marked_removed}"
    )

    logger.info("Moving binarized images...")
    kept_bin, removed_bin = move_files_to_timestamped_folder(
        src_dir=binarized_src,
        dst_base_dir=dst_base_dir,
        list_img_remove=img_remove,
        file_type="binarized",
        timestamp=timestamp,
    )
    logger.info(f"Binarized: {kept_bin} kept | {removed_bin} removed")

    # Step 3: Move original extracted lines to timestamped folders
    if extracted_src.is_dir():
        logger.info("Moving original extracted lines...")
        kept_orig, removed_orig = move_files_to_timestamped_folder(
            src_dir=extracted_src,
            dst_base_dir=dst_base_dir,
            list_img_remove=img_remove,
            file_type="original",
            timestamp=timestamp,
        )
        logger.info(f"   ✓ Original: {kept_orig} kept | {removed_orig} removed")
    else:
        logger.warning(f"⚠ Extracted source folder not found: {extracted_src}")
        kept_orig = removed_orig = 0

    # Optional wide-line split step. Runs on the KEPT folders only,
    # after files are staged into the destination, so we never mutate
    # the upstream binarised/extracted sources. Disabled when no
    # kraken_json_dir is provided.
    wide_stats: dict | None = None
    if kraken_json_dir is not None:
        logger.info("Scanning kept images for wide-line splits...")
        binarised_kept_dir = run_output_dir / "binarized" / "kept"
        original_kept_dir_path = (
            run_output_dir / "original" / "kept" if extracted_src.is_dir() else None
        )
        wide_stats = split_wide_kept_images(
            binarised_kept_dir=binarised_kept_dir,
            original_kept_dir=original_kept_dir_path,
            kraken_json_dir=Path(kraken_json_dir),
            percentile=wide_percentile,
            min_ratio_to_median=wide_min_ratio_to_median,
            logger=logger,
        )
        logger.info(
            f"Wide-line split: {wide_stats['n_split']} pairs created "
            f"from {wide_stats['n_wide']} candidates "
            f"({wide_stats['n_skipped']} skipped)"
        )

    # Final summary. The kept/removed totals are per-IMAGE (each image
    # has one binarized file + one original file), so we report the
    # canonical per-image counts from the binarised source — NOT the
    # sum across both file types, which would double-count every image.
    # The original-vs-binarised pair is sanity-checked here and a
    # warning is raised if they don't match (which would suggest one of
    # the source folders is missing files).
    total_kept = kept_bin
    total_removed = removed_bin
    if extracted_src.is_dir() and (kept_orig != kept_bin or removed_orig != removed_bin):
        logger.warning(
            "Binarised vs. original counts disagree — possible missing files. "
            f"Binarised: kept={kept_bin}, removed={removed_bin}; "
            f"Original: kept={kept_orig}, removed={removed_orig}"
        )

    logger.info("Filtering complete")
    logger.info(f"Kept: {total_kept} | Removed: {total_removed}")
    logger.info(f"Results: {run_output_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return {
        "total_analyzed": total_analyzed,
        "marked_removed": int(marked_removed),
        "kept": total_kept,
        "removed": total_removed,
        "output_dir": str(run_output_dir),
        "tracking_csv": str(run_output_dir / "filter_tracking.csv"),
        "wide_line_split": wide_stats,
    }
