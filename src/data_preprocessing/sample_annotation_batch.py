"""Sample a fresh batch of line stems for hand-annotation.

Each batch produces a folder of ``<stem>.png`` + ``<stem>.gt.txt``
pairs, where the ``.gt.txt`` is pre-filled with the catmus baseline
OCR text (as a starting point that the annotator corrects in place),
plus an ``_INDEX.csv`` listing the picked stems and a ``_README.md``
recording provenance.

Key design points, derived from lessons on batches 3-6:

- **Line images come from the FILTERED / KEPT folder**, not the raw
  extraction folder. The filtered folder has manually corrected crops
  (e.g. double-column re-segmentations) and has already dropped the
  unusable crops that ink-bleed detection flagged. Using the raw
  extraction folder means shipping annotators corrupted crops that
  never made it into the OCR / training pipeline — batch 6's line 101
  page 67 was a case in point.

- **Exclusion by pool folder(s)**: rather than track a growing global
  ignore list by hand, the caller points at one or more
  ``exclude_folders`` (the current corrected pool + the permanent
  validation set) and every ``.gt.txt`` stem in any of them is
  excluded automatically. So each new batch is guaranteed
  non-overlapping with both the training pool AND the held-out
  benchmark set, without the caller having to manage state.
  Non-existent folders are skipped with a warning (useful during
  bootstrap when e.g. the validation set hasn't been created yet).

- **Optional content filter**: when the goal is a *targeted* batch
  (e.g. capital C or capital E, unusual glyphs, specific abbreviations),
  pass a regex via ``pattern``. Only OCR seeds matching that regex are
  eligible. When ``pattern`` is ``None``, every non-empty seed is
  eligible.

- **Reproducibility**: the ``seed`` argument fully determines which
  100 stems are picked from the eligible pool. Two runs of the same
  seed → the same batch. Previous batches recorded their seed in
  ``_README.md`` so we can avoid re-using seeds and accidentally
  producing an overlapping batch.

Follows the ``scripts/`` <-> ``src/`` split — the CLI wrapper in
``scripts/data_preprocessing/run_sample_annotation_batch.py`` parses
argparse; the module here hosts the workflow.
"""

import csv
import datetime
import json
import logging
import os
import random
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path


def setup_simple_logging(
    logs_dir: str | Path,
    task_name: str = "sample_annotation_batch",
    run_name: str | None = None,
):
    """File + console logger, same shape as the other src.* modules."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logs_dir) / f"{run_name}_{task_name}.log"

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.info("=== %s Run Started | Run: %s ===", task_name.upper(), run_name)
    logger.info("Log file: %s", log_file)
    return logger, str(log_file)


def _iter_candidates(
    source_lines_dir: Path,
    ocr_seed_dir: Path,
    excluded_stems: set[str],
    pattern: re.Pattern[str] | None,
) -> Iterator[tuple[str, str, Path, str]]:
    """Yield ``(page, stem, png_path, ocr_seed_text)`` for eligible lines.

    Walks ``source_lines_dir`` per-page (subdir per folio) and pairs each
    PNG with its matching ``.txt`` in ``ocr_seed_dir/<page>/``. A line
    is eligible iff:

    - The stem is not in ``excluded_stems``.
    - The OCR seed file exists AND has non-empty stripped content.
    - If ``pattern`` is set, the OCR seed matches it.
    """
    for page_dir in sorted(source_lines_dir.iterdir()):
        if not page_dir.is_dir():
            continue
        ocr_page = ocr_seed_dir / page_dir.name
        for png in sorted(page_dir.glob("*.png")):
            stem = png.stem
            if stem in excluded_stems:
                continue
            ocr_file = ocr_page / f"{stem}.txt"
            if not ocr_file.is_file():
                continue
            text = ocr_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            if pattern is not None and not pattern.search(text):
                continue
            yield page_dir.name, stem, png, text


def _load_excluded_stems(
    exclude_folders: list[Path], logger: logging.Logger | None = None
) -> set[str]:
    """Return the union of stems across every folder in ``exclude_folders``.

    Uses ``.gt.txt`` files as the canonical "already annotated" marker —
    matching each ``<stem>.gt.txt`` regardless of whether a ``.png``
    sibling exists (defensive against half-populated pool states).

    Missing folders are logged as a warning and skipped, not treated as
    an error. That way a caller can list both the training pool and
    the validation set from day one; if the validation set doesn't
    exist yet (bootstrap), the sampler still works.
    """
    stems: set[str] = set()
    for folder in exclude_folders:
        if not folder.is_dir():
            if logger is not None:
                logger.warning("Exclude folder does not exist, skipping: %s", folder)
            continue
        stems.update(p.name[: -len(".gt.txt")] for p in folder.glob("*.gt.txt"))
    return stems


def _write_readme(
    out_dir: Path,
    *,
    run_name: str,
    n_target: int,
    n_picked: int,
    n_pages: int,
    seed: int,
    n_excluded: int,
    pattern: re.Pattern[str] | None,
    pattern_label: str | None,
    source_lines_dir: Path,
) -> None:
    """Write the batch ``_README.md`` describing what was sampled and why."""
    pattern_paragraph = ""
    if pattern is not None:
        label = f" ({pattern_label})" if pattern_label else ""
        pattern_paragraph = f"\nFilter{label}: OCR seed text matches regex `{pattern.pattern}`.\n"
    lines_source_note = (
        f"\nLine PNGs sourced from `{source_lines_dir}` — the "
        "filtered / manually-corrected line-crop folder, NOT the raw "
        "extraction folder. This ensures annotators see the same "
        "crops the OCR pipeline runs on (post ink-bleed filter + "
        "double-column corrections).\n"
    )
    (out_dir / "_README.md").write_text(
        f"# Annotation batch ({run_name})\n\n"
        f"{n_picked} line samples (target: {n_target}) sampled with "
        f"seed={seed} across {n_pages} pages.\n"
        f"{pattern_paragraph}"
        f"{lines_source_note}\n"
        f"Excludes the {n_excluded} already-annotated stems in the\n"
        "corrected pool. All `.gt.txt` files are guaranteed non-empty\n"
        "at sampling time.\n\n"
        "## Files\n"
        "- `<stem>.png` — line image\n"
        "- `<stem>.gt.txt` — pre-filled with OCR seed; correct in place\n"
        "- `_INDEX.csv` — page + stem listing\n\n"
        "After review, move verified pairs into the corrected pool\n"
        "(`data/processed/annotated_samples/OCR/full_annotated/`).\n",
        encoding="utf-8",
    )


def run_sample_annotation_batch(
    source_lines_dir: str | Path,
    ocr_seed_dir: str | Path,
    exclude_folders: list[str | Path],
    output_root: str | Path,
    n_target: int = 100,
    seed: int = 42,
    pattern: str | None = None,
    pattern_label: str | None = None,
    logs_dir: str | Path | None = None,
    task_name: str = "sample_annotation_batch",
    run_name: str | None = None,
    log_config: bool = True,
    output_subfolder_prefix: str = "real_val_sample",
) -> dict:
    """Sample a batch of line stems for hand-annotation.

    Args:
        source_lines_dir: root of ``<page>/*.png`` line crops
            (typically ``filtered_images/.../original/kept``).
        ocr_seed_dir: root of ``<page>/*.txt`` OCR predictions to
            pre-fill the ``.gt.txt`` labels.
        exclude_folders: list of folders whose ``<stem>.gt.txt`` files
            mark already-annotated / held-out stems. Typically the
            training pool (``full_annotated/``) AND the permanent
            validation set (``validation/``) — both are checked so a
            training batch can never accidentally pick a held-out stem.
        output_root: where to create the batch subfolder. The batch
            folder is ``<output_root>/<output_subfolder_prefix>_<run_name>/``.
        output_subfolder_prefix: prefix for the output subdir name.
            Default ``"real_val_sample"`` matches historic training
            batches; use ``"validation"`` for the permanent held-out
            benchmark set so the two are visually distinct.
        n_target: how many stems to pick. If fewer than this are
            eligible, all eligible ones are used and a warning is logged.
        seed: RNG seed for reproducibility.
        pattern: optional regex to filter OCR seeds. Only seeds
            matching this pattern are eligible.
        pattern_label: human-readable label for the pattern, written
            into ``_README.md`` (e.g. "capital C or E targeted").

    Returns:
        Dict with ``out_dir``, ``n_picked``, ``n_pages``, ``n_eligible``,
        ``n_excluded``, and paths to the log and index files.
    """
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    source_lines_dir = Path(source_lines_dir)
    ocr_seed_dir = Path(ocr_seed_dir)
    exclude_folders_ = [Path(p) for p in exclude_folders]
    output_root = Path(output_root)
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger, log_file = setup_simple_logging(
        logs_dir=str(logs_dir), task_name=task_name, run_name=run_name
    )

    if log_config:
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(project_root),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            git_commit = "unknown"
        config = {
            "run_name": run_name,
            "git_commit": git_commit,
            "timestamp": datetime.datetime.now().isoformat(),
            "source_lines_dir": str(source_lines_dir),
            "ocr_seed_dir": str(ocr_seed_dir),
            "exclude_folders": [str(p) for p in exclude_folders_],
            "output_root": str(output_root),
            "output_subfolder_prefix": output_subfolder_prefix,
            "n_target": n_target,
            "seed": seed,
            "pattern": pattern,
            "pattern_label": pattern_label,
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    assert source_lines_dir.is_dir(), f"Source lines folder not found: {source_lines_dir}"
    assert ocr_seed_dir.is_dir(), f"OCR seed folder not found: {ocr_seed_dir}"
    assert exclude_folders_, "At least one exclude-folder is required."

    compiled = re.compile(pattern) if pattern else None
    excluded = _load_excluded_stems(exclude_folders_, logger=logger)
    logger.info(
        "Excluding %d stems from %d folder(s): %s",
        len(excluded),
        len(exclude_folders_),
        [str(p) for p in exclude_folders_],
    )

    candidates = list(_iter_candidates(source_lines_dir, ocr_seed_dir, excluded, compiled))
    logger.info("Eligible candidates: %d", len(candidates))

    if len(candidates) < n_target:
        logger.warning(
            "Only %d eligible candidates < target %d — sampling all eligible.",
            len(candidates),
            n_target,
        )
        take = candidates
    else:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        take = candidates[:n_target]

    picked = sorted(take, key=lambda x: (x[0], x[1]))

    out_dir = output_root / f"{output_subfolder_prefix}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    pages_seen: set[str] = set()
    for page, stem, png, text in picked:
        shutil.copy2(png, out_dir / png.name)
        (out_dir / f"{stem}.gt.txt").write_text(text + "\n", encoding="utf-8")
        index_rows.append(
            {
                "page": page,
                "line_stem": stem,
                "image": png.name,
                "ocr_seed_txt": f"{stem}.gt.txt",
            }
        )
        pages_seen.add(page)

    index_path = out_dir / "_INDEX.csv"
    with open(index_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["page", "line_stem", "image", "ocr_seed_txt"])
        w.writeheader()
        w.writerows(index_rows)

    _write_readme(
        out_dir,
        run_name=run_name,
        n_target=n_target,
        n_picked=len(picked),
        n_pages=len(pages_seen),
        seed=seed,
        n_excluded=len(excluded),
        pattern=compiled,
        pattern_label=pattern_label,
        source_lines_dir=source_lines_dir,
    )

    logger.info(
        "Wrote %d pairs + _INDEX.csv + _README.md to %s (pages: %d)",
        len(picked),
        out_dir,
        len(pages_seen),
    )

    return {
        "out_dir": str(out_dir),
        "n_picked": len(picked),
        "n_pages": len(pages_seen),
        "n_eligible": len(candidates),
        "n_excluded": len(excluded),
        "index_path": str(index_path),
        "log_path": log_file,
    }
