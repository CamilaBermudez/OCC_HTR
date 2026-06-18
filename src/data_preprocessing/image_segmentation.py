import datetime
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from kraken import serialization
from kraken.containers import BaselineLine, BBoxLine, Segmentation
from PIL import Image
from tqdm import tqdm

from src.utils.path_utils import format_filename, format_for_cli


def setup_simple_logging(logs_dir: str, run_name: str | None = None):
    """Minimal logging setup: file + console, INFO level only"""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(logs_dir) / f"{run_name}_segmentation.log"

    logger = logging.getLogger("segmentation")
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


def segment_image(img_path: Path, output_path: Path, mask_path: Path | None = None) -> bool:
    try:
        load_dotenv()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = os.environ.get("PYTHON_IO_ENCODING", "utf-8")

        img_path = Path(img_path)
        output_path = Path(output_path)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)

        mask_cmd = None
        if mask_path is not None and Path(mask_path).exists():
            mask_cmd = format_for_cli(Path(mask_path))[0]

        kraken_bin = os.environ.get("KRAKEN_BIN") or shutil.which("kraken")
        if not kraken_bin:
            return False

        cmd = [kraken_bin, "-i", input_cmd, output_cmd, "segment", "-bl"]
        if mask_cmd:
            cmd.extend(["-m", mask_cmd])

        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, timeout=300)
        return True
    except Exception:
        import traceback

        traceback.print_exc()  # shows full error + line number
        return False


def _line_baseline_midpoint(line: dict) -> tuple[float, float]:
    """Return the (cx, cy) midpoint of a line's baseline.

    Falls back to the boundary polygon's centroid when the baseline is
    missing or malformed. ``(0.0, 0.0)`` is a defensive last resort that
    keeps the sort stable rather than crashing on a degenerate record.
    """
    bl = line.get("baseline")
    if isinstance(bl, list) and len(bl) >= 2:
        (x0, y0), (x1, y1) = bl[0], bl[-1]
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bnd = line.get("boundary")
    if isinstance(bnd, list) and bnd:
        xs = [p[0] for p in bnd]
        ys = [p[1] for p in bnd]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return 0.0, 0.0


def _line_left_x(line: dict) -> float:
    """Return the x-coordinate of the line's left edge (baseline start).

    For column detection this is more stable than the baseline midpoint:
    every line in a given column starts at the column's left margin
    regardless of how long the line is, while the midpoint shifts with
    line length and can make a single column look bimodal.

    Falls back to the boundary polygon's leftmost x when the baseline
    is missing.
    """
    bl = line.get("baseline")
    if isinstance(bl, list) and bl:
        return float(bl[0][0])
    bnd = line.get("boundary")
    if isinstance(bnd, list) and bnd:
        return float(min(p[0] for p in bnd))
    return 0.0


def _find_column_spans_by_projection(
    line_x_ranges: list[tuple[int, int]],
    page_width: int,
    coverage_fraction: float = 0.10,
) -> list[tuple[int, int]]:
    """Detect column boundaries via x-axis projection.

    For each x-coordinate, count how many lines' horizontal span
    ``(x0, x1)`` covers it. Columns appear as continuous runs where the
    projection is high; column gutters (and page margins) are valleys.

    This handles within-column variations that histogram peak detection
    can't: indented lines (different x0) still span most of the same
    column body, so their projection contribution lands inside the
    column boundary rather than creating a spurious second peak.

    Args:
        line_x_ranges: per-line ``(x0, x1)`` integer span.
        page_width: image width in pixels.
        coverage_fraction: minimum fraction of all lines that must span
            an x-coordinate for that x to count as "inside a column".
            Default 0.10 (10%): a region needs sustained coverage by
            at least a tenth of all lines to qualify. Lower for sparse
            data (e.g. marginalia); higher to suppress column outliers.

    Returns:
        Sorted list of ``(col_left, col_right)`` integer ranges.
    """
    n = len(line_x_ranges)
    if n == 0 or page_width <= 0:
        return []

    # Difference array: add 1 at x0, subtract 1 at x1+1. Cumulative sum
    # gives per-x coverage in O(n + page_width) rather than O(n·width).
    # Both end-indices are clamped to [0, page_width] — when x1 reaches
    # exactly page_width the `+1` would otherwise overflow the diff
    # array, and any line whose right edge exceeds page_width simply
    # contributes coverage up to the page edge.
    diff = [0] * (page_width + 1)
    for x0, x1 in line_x_ranges:
        a = max(0, min(page_width, x0))
        b = max(0, min(page_width, x1 + 1))
        if b > a:
            diff[a] += 1
            diff[b] -= 1
    proj = [0] * page_width
    running = 0
    for x in range(page_width):
        running += diff[x]
        proj[x] = running

    threshold = max(2, int(coverage_fraction * n))
    columns: list[tuple[int, int]] = []
    in_col = False
    col_start = 0
    for x in range(page_width):
        if proj[x] >= threshold:
            if not in_col:
                col_start = x
                in_col = True
        elif in_col:
            columns.append((col_start, x - 1))
            in_col = False
    if in_col:
        columns.append((col_start, page_width - 1))

    return columns


def reorder_lines_reading_order(
    lines: list[dict],
    page_width: int | None = None,
) -> list[dict]:
    """Reorder kraken line dicts into Western reading order, supporting
    pages with any number of columns (2, 3, 4, ...).

    Kraken emits ``lines`` in detection order, which on a multi-column
    manuscript is often "right column first" or interleaved — that's
    why a page's line 0 can land in the rightmost column. We sort lines
    so the resulting list goes:

        column 1 top → column 1 bottom →
        column 2 top → column 2 bottom →
        column N top → column N bottom

    Algorithm:

    1. Compute each line's ``(cx, cy)`` baseline midpoint and ``(x0, x1)``
       horizontal span (boundary fallbacks when the baseline is absent).
    2. Project all line spans onto the x-axis and find columns as
       continuous runs of high coverage (≥ 10% of lines spanning that x).
       This discovers however many columns the page actually has and is
       robust against indented or short lines, because a line still
       *spans* its column's body even if it doesn't start at the column's
       left margin.
    3. Assign each line to the column whose midpoint is nearest its cx.
    4. Within each column, sort top-to-bottom by ``cy`` row bucket, then
       left-to-right by ``cx`` for ties — so a row that kraken split into
       a left half and a right half comes out left-then-right.

    A column that contains two stacked sub-blocks (e.g. lines at
    cy=100-300, gap, more lines at cy=500-700) is handled correctly:
    those lines all live inside the same column span, then the cy sort
    interleaves them top-down.

    Args:
        lines: kraken line dicts; each should have ``baseline`` (preferred)
            or ``boundary``.
        page_width: image width in pixels — used as the projection
            range. If omitted, falls back to the rightmost observed
            line edge plus one.

    Returns:
        A *new* list with the same dicts arranged in reading order;
        per-line dicts themselves are not mutated.
    """
    if not lines:
        return list(lines)

    # Each entry: ((cx, cy), line, (x0, x1))
    # cx/cy drive within-column row sorting; (x0, x1) drives column detection.
    decorated: list[tuple[tuple[float, float], dict, tuple[int, int]]] = []
    for line in lines:
        cx, cy = _line_baseline_midpoint(line)
        x0_f = _line_left_x(line)
        bl = line.get("baseline")
        if isinstance(bl, list) and len(bl) >= 2:
            x1_f = float(bl[-1][0])
        else:
            bnd = line.get("boundary")
            if isinstance(bnd, list) and bnd:
                x1_f = float(max(p[0] for p in bnd))
            else:
                x1_f = x0_f
        decorated.append(((cx, cy), line, (int(x0_f), int(x1_f))))

    if not (page_width and page_width > 0):
        # Fall back to the rightmost line edge as the implicit page width.
        right_edges = [d[2][1] for d in decorated]
        page_width = max(right_edges) + 1 if right_edges else 1
    page_width_int = int(page_width)

    spans = _find_column_spans_by_projection(
        [d[2] for d in decorated],
        page_width_int,
    )

    # Fallback to a single column if projection found nothing (degenerate
    # data where no x-coordinate had enough coverage).
    if not spans:
        spans = [(0, page_width_int - 1)]

    column_midpoints = [(cl + cr) / 2.0 for cl, cr in spans]

    # Assign each line to the LEFTMOST column whose right edge sits at
    # or past the line's left edge (x0). This handles three cases in a
    # single rule:
    #   - normal line: x0 is inside its column's span → assigned to that
    #     column.
    #   - indented line: x0 is shifted right of the column's left margin
    #     but still inside the column's span → same column.
    #   - merged-column line (e.g. a wide top-of-page baseline kraken
    #     produces by joining the first row of two adjacent columns into
    #     one record): x0 is at the *left* edge of the leftmost column
    #     the line touches; nearest-midpoint by cx puts it in the next
    #     column over, but "first column whose right edge ≥ x0" places
    #     it in the column where it visually starts. The filter step's
    #     wide-image splitter then cuts it in half and inserts the right
    #     half between this column and the next.
    # We fall back to nearest-cx for lines whose x0 is past the right
    # edge of every column (rare; happens for stray marginal noise).
    columns: list[list] = [[] for _ in spans]
    for entry in decorated:
        x0 = entry[2][0]
        assigned = None
        for i, (_, col_right) in enumerate(spans):
            if col_right >= x0:
                assigned = i
                break
        if assigned is None:
            cx = entry[0][0]
            assigned = 0
            best = abs(cx - column_midpoints[0])
            for i, mid in enumerate(column_midpoints[1:], start=1):
                d = abs(cx - mid)
                if d < best:
                    best = d
                    assigned = i
        columns[assigned].append(entry)

    ordered: list[dict] = []
    for col in columns:
        # Within a column, sort by row first (top-to-bottom), then by cx
        # (left-to-right) for segments that land on the same row. Without
        # the secondary cx key, two segments at nearly identical cy keep
        # whatever order kraken returned, which on a split line means
        # the right half can end up before the left half.
        #
        # The "row" key buckets cy by roughly half the typical line
        # spacing in this column so a few-pixel baseline drift doesn't
        # falsely separate two same-row segments. We estimate line
        # spacing from the median gap between consecutive cy values
        # within the column itself, with sensible fallbacks for short
        # columns.
        cys = sorted(entry[0][1] for entry in col)
        diffs = [cys[i + 1] - cys[i] for i in range(len(cys) - 1) if cys[i + 1] - cys[i] > 0]
        if diffs:
            median_line_gap = sorted(diffs)[len(diffs) // 2]
        else:
            median_line_gap = 0.0
        row_bucket = max(8.0, median_line_gap / 2.0)
        # entry = ((cx, cy), line_dict, x0) → sort by (row bucket, cx)
        col.sort(key=lambda d: (int(d[0][1] / row_bucket), d[0][0]))
        ordered.extend(entry[1] for entry in col)
    return ordered


def apply_reading_order_to_json(json_path: Path, image_path: Path | None = None) -> None:
    """Rewrite a kraken segmentation JSON so its ``lines`` are in reading
    order. The page width comes from the source image — passed explicitly
    via ``image_path`` or resolved from the JSON's own ``imagename``
    field (kraken records the original path there at segmentation time).

    The JSON-side fallback exists because the JSON stem (``20_f_015v_016``)
    rarely matches the original image filename verbatim (``20 - f. 015v - 016.jpg``),
    so a naive caller doing ``rglob(f'{json_path.stem}.*')`` finds nothing
    and silently skips the rewrite.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if "lines" not in data or not data["lines"]:
        return

    if image_path is None:
        recorded = data.get("imagename")
        if not recorded:
            raise FileNotFoundError(
                f"No image_path passed and JSON has no 'imagename' field: {json_path}"
            )
        image_path = Path(recorded)
        if not image_path.is_absolute():
            image_path = json_path.parent.parent / image_path  # back out of the segmentation dir
            if not image_path.exists():
                image_path = Path.cwd() / recorded

    with Image.open(image_path) as im:
        page_width = im.size[0]
    data["lines"] = reorder_lines_reading_order(data["lines"], page_width=page_width)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def format_from_JSON_to_ALTO_XML(input_json_path, input_img_path, output_alto_path):
    with open(input_json_path, encoding="utf-8") as f:
        data = json.load(f)
    im = Image.open(input_img_path)

    lines = []
    for line in data["lines"]:
        if line.get("type") == "baselines":
            lines.append(
                BaselineLine(
                    id=line["id"],
                    baseline=line["baseline"],
                    boundary=line["boundary"],
                    text=line.get("text"),
                    tags=line.get("tags"),
                )
            )
        else:
            lines.append(
                BBoxLine(
                    id=line["id"],
                    bbox=line["bbox"],
                    text=line.get("text"),
                    tags=line.get("tags"),
                    split=line.get("split"),
                )
            )

    seg = Segmentation(
        type=data["type"],
        imagename=data["imagename"],
        text_direction=data["text_direction"],
        script_detection=data["script_detection"],
        lines=lines,
        regions=data.get("regions", {}),
        line_orders=data.get("line_orders", []),
    )

    alto = serialization.serialize(seg, image_size=im.size)
    with open(output_alto_path, "w", encoding="utf-8") as f:
        f.write(alto)


def segment_all_images(
    input_folder: str | Path,
    output_folder: str | Path,
    masks_folder: str | Path,
    logs_dir: str | None = None,
    run_name: str | None = None,
) -> dict:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    masks_folder = Path(masks_folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"seg_{timestamp}"

    if logs_dir:
        logger, log_file = setup_simple_logging(logs_dir, run_name)
        logger.info(f"=== Segmentation Started | Run: {run_name} ===")
    else:
        logger = logging.getLogger("segmentation")
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

    if not input_folder.is_dir():
        logger.error(f"Input folder not found: {input_folder}")
        return {"success": False}

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        [f for f in input_folder.iterdir() if f.suffix.lower() in image_extensions]
    )

    config_summary = {
        "run": run_name,
        "git": git_commit,
        "input": str(input_folder.name),
        "output": str(output_folder.name),
        "images_count": len(image_files),
    }
    if log_file:
        logger.info(f"Config: {json.dumps(config_summary)}")

    output_folder = Path(output_folder) / f"segmentation_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting segmentation for {len(image_files)} images...")

    alto_dir = output_folder / "alto_format"
    alto_dir.mkdir(parents=True, exist_ok=True)

    if not output_folder.exists():
        logger.error("Failed to create directory. Check permissions or parent path.")

    success_count = 0
    for img_path in tqdm(image_files, desc="Segmenting", unit="file"):
        base_name = img_path.stem
        output_path, output_filename, processed_name = format_filename(base_name, output_folder)
        input_cmd, output_cmd = format_for_cli(img_path, output_path)
        alto_path = alto_dir / f"{processed_name}.xml"
        mask_path = masks_folder / f"{processed_name}.png"
        if not mask_path.exists():
            mask_path = None

        if segment_image(input_cmd, output_cmd, mask_path=mask_path):
            success_count += 1
            # Kraken returns lines in detection order, which on a multi-
            # column page typically isn't reading order (the visible
            # numbering would start on the right column, for example).
            # Rewrite the JSON in-place so downstream consumers
            # (plot_bounds, crop_segments, ALTO conversion) all read
            # left-column-top-down, right-column-top-down.
            try:
                apply_reading_order_to_json(output_path, img_path)
            except Exception as e:
                logger.warning(f"reading-order rewrite failed for {img_path.name}: {e}")
            format_from_JSON_to_ALTO_XML(
                input_json_path=output_path, input_img_path=img_path, output_alto_path=alto_path
            )

    logger.info(f"Segmentation complete: {success_count}/{len(image_files)} succeeded")
    logger.info(f"Output: {output_folder}")
    if log_file:
        logger.info(f"Log file: {log_file}")

    return {"total": len(image_files), "success": success_count, "output": str(output_folder)}
