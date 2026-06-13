"""Add a non-rectangular alpha mask to fully-opaque glyph crops.

Only touches PNGs that have no transparency at all (alpha=255 everywhere) —
crops that were already cut by hand are skipped.

Algorithm per PNG:
  1. Estimate parchment colour from a 2-pixel border ring (median RGB) —
     adapts to each crop's local lighting rather than a global threshold.
  2. Compute Euclidean colour distance from parchment for each pixel.
  3. Map distance → soft alpha (0 near parchment, 1 for clearly-different ink).
  4. Snap alphas below `low_thresh` to 0 to kill speckle.
  5. Keep only the largest connected ink component (drops isolated specks
     and corner noise that wasn't well-represented in the border sample).
  6. Dilate the kept mask by 1 px to preserve stroke anti-aliasing edges,
     then intersect with the soft alpha so opaque interior stays opaque
     and the edge keeps its anti-aliased ramp.
  7. Write RGBA back to disk in-place.

Run via `uv run python scripts/data_augmentation/_auto_segment_glyph_alpha.py`.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_opening, label

DEFAULT_FOLDERS = [
    "O_",
    "am",
    "an",
    "au",
    "cum",
    "em",
    "ma",
    "me",
    "mi",
    "mu",
    "nu",
    "um",
    "un",
    "x",
]


def auto_segment(
    arr_rgba: np.ndarray,
    *,
    border_thickness: int = 2,
    near_dist: float = 25.0,
    far_dist: float = 70.0,
    low_thresh: float = 0.20,
    dilation: int = 1,
) -> np.ndarray:
    """Return a new RGBA array with alpha derived from ink-vs-parchment colour distance."""
    rgb = arr_rgba[..., :3].astype(np.float32)
    h, w = rgb.shape[:2]

    # Border-ring parchment reference. Median is robust to a stray ink stroke
    # that clips the edge of the crop.
    border_mask = np.ones((h, w), dtype=bool)
    bt = max(1, min(border_thickness, h // 4, w // 4))
    border_mask[bt:-bt, bt:-bt] = False
    parchment = np.median(rgb[border_mask], axis=0)

    dist = np.linalg.norm(rgb - parchment, axis=2)
    alpha = np.clip((dist - near_dist) / max(1.0, far_dist - near_dist), 0.0, 1.0)
    alpha = np.where(alpha < low_thresh, 0.0, alpha)

    # Clean up speckle, then keep every connected component that's large
    # enough to plausibly be an actual letter — not just the single
    # largest. Multi-letter ligatures (am, cum, me, …) usually contain
    # separate components per letter that aren't physically touching;
    # keeping only the largest would discard the rest. We also always
    # keep the largest component regardless of size, so a near-empty
    # image still produces *some* mask.
    binary = alpha > 0.5
    binary = binary_opening(binary, iterations=1)  # drop single-pixel noise
    # Close small holes inside letter strokes (faded ink pixels that
    # didn't pass the colour threshold leave the binarised stroke pitted
    # with single-pixel gaps; without closing, the final composite reads
    # as horizontal stripes instead of solid letters). 2-iteration
    # closing fills holes up to ~4 px wide while keeping inter-letter
    # gaps in multi-letter ligatures.
    binary = binary_closing(binary, iterations=2)
    labeled, n = label(binary)
    if n == 0:
        keep = np.zeros_like(binary)
    else:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background label
        largest = int(sizes.max())
        # Per-component size threshold: at least 15% of the largest
        # component OR at least 0.3% of the image area, whichever is
        # smaller. 15% catches near-equal-size letter components in
        # ligatures (a vs m, c vs u vs m); the area floor protects very
        # asymmetric crops where one letter dwarfs the others.
        h_px, w_px = binary.shape
        area_floor = max(8, int(0.003 * h_px * w_px))
        min_size = min(int(0.15 * largest), area_floor)
        keep_labels = {int(i) for i in np.where(sizes >= min_size)[0]}
        keep_labels.add(int(np.argmax(sizes)))  # always keep largest
        keep = np.isin(labeled, list(keep_labels))

    if dilation > 0 and keep.any():
        keep = binary_dilation(keep, iterations=dilation)

    alpha = np.where(keep, alpha, 0.0)

    out = arr_rgba.copy()
    out[..., 3] = (alpha * 255.0).astype(np.uint8)
    return out


def process_folder(folder: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Returns (processed, skipped_already_cropped, skipped_missing)."""
    if not folder.is_dir():
        return (0, 0, 1)
    processed = skipped = 0
    for png in sorted(folder.glob("*.png")):
        im = Image.open(png).convert("RGBA")
        arr = np.asarray(im, dtype=np.uint8)
        # Already cropped (any transparency present) — leave alone.
        if (arr[..., 3] != 255).any():
            skipped += 1
            continue
        new_arr = auto_segment(arr)
        opaque_frac = (new_arr[..., 3] == 255).mean()
        transp_frac = (new_arr[..., 3] == 0).mean()
        print(
            f"  {png.name:<16}  shape={im.size}  "
            f"opaque={opaque_frac:.2f}  transparent={transp_frac:.2f}"
        )
        if not dry_run:
            Image.fromarray(new_arr, mode="RGBA").save(png)
        processed += 1
    return (processed, skipped, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glyphs-root",
        default="glyphs",
        help="Root directory containing the pattern-stamp folders.",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=DEFAULT_FOLDERS,
        help="Subfolder names to process. Defaults to all pattern-stamp folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but do not overwrite any PNGs.",
    )
    args = parser.parse_args()

    root = Path(args.glyphs_root)
    total_proc = total_skip = total_miss = 0
    for f in args.folders:
        print(f"=== {f} ===")
        p, s, m = process_folder(root / f, dry_run=args.dry_run)
        total_proc += p
        total_skip += s
        total_miss += m
    print(
        f"\nDone: processed={total_proc}  "
        f"skipped_already_cropped={total_skip}  "
        f"missing_folders={total_miss}  "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
