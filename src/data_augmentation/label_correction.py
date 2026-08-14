"""Build a {augmented_image_name: corrected_label} JSON for a finished
augmentation run.

For each augmented image (named ``<source_stem>_aug<NN>.png``), look up the
source line's text in the medieval-text-generation labels JSON, apply
configurable character substitutions, and write the result to disk.

The default substitutions are the classical Latin/medieval Iberian
orthography → modern normalisations:

    v → u    V → U
    j → i    J → I

Pass a custom mapping to extend or replace this set.

The output directory is named ``labels_<timestamp>`` where ``<timestamp>``
is extracted from the augmented images folder name (e.g., ``aug_20260608_120000``
→ ``labels_20260608_120000``). This keeps every augmentation run paired
with its corresponding label run by the timestamp alone.
"""

import datetime
import json
import logging
import os
import re
import subprocess
from pathlib import Path

# Default character substitutions.
#
# v/V are typeset where u/U is meant in medieval Iberian orthography; same
# for j/J vs i/I. We fold those to lowercase u/i.
#
# Additionally, the uppercase letters {I, T, A, E, S, O, H, M, D, Q, F}
# appear in the cometa transcriptions but are NOT in the catmus-medieval
# OCR model's output codec — they trigger a codec-mismatch warning at
# fine-tuning time. Medieval scribes rarely distinguished case, so we
# fold those to lowercase rather than forcing the output layer to grow.
#
# Override or extend via `correct_labels(substitutions=...)`.
# Diplomatic normalization ONLY — match the real 600 GT convention (spec §6.5.26):
# u/v and i/j collapse (real uses u:1274/v:2, i:1040/j:0), but CAPITALS ARE PRESERVED
# (real uses E:71, C:35, D/F/I/L/M/R/S/U…). The old map also lowercased
# T/A/E/S/O/H/M/D/Q/F and I→i, U→u, which deleted capitals the GT expects → the model
# never learned to emit them (label had 0 'E' vs 71 in real). Capital v/j normalize to
# capital u/i (V→U, J→I), not to lowercase. Long-s / rotunda-r are handled at render time
# (labelled s/r, matching real ſ:0 ꝛ:0). No abbreviation expansion — label = image.
DEFAULT_SUBSTITUTIONS: dict[str, str] = {
    "v": "u",
    "j": "i",
    "V": "U",
    "J": "I",
}


_AUG_FILENAME_RE = re.compile(r"^(.+)_aug\d+\.png$")


def setup_label_correction_logging(logs_dir: str | Path, run_name: str):
    """File + console logger, same pattern as the other src/ scripts."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"{run_name}_label_correction.log"

    logger = logging.getLogger("label_correction")
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


def correct_labels(
    input_json: str | Path,
    augmented_folder: str | Path,
    output_base_dir: str | Path,
    *,
    substitutions: dict[str, str] | None = None,
    text_field: str = "original_text",
    logs_dir: str | Path | None = None,
) -> Path:
    """Build a corrected-labels JSON for an augmented-images directory.

    Args:
        input_json: Path to the medieval-text-generation ``labels.json``,
            keyed by source image name (the line-render). The script reads
            ``labels[source_name][text_field]`` to get the base text.
        augmented_folder: Path to the augmented images directory
            (typically ``aug_<timestamp>/``). Files matching the pattern
            ``<source_stem>_aug<NN>.png`` are processed; anything else
            (e.g., ``augmentation_log.json``) is ignored.
        output_base_dir: Parent directory under which a ``labels_<timestamp>``
            subdirectory is created (timestamp extracted from
            ``augmented_folder.name``).
        substitutions: Mapping of source character to replacement character.
            Defaults to ``DEFAULT_SUBSTITUTIONS`` (v→u, V→U, j→i, J→I).
        text_field: Which field of the source labels entry to use as the
            base text. Default ``"original_text"`` (clean transcription).
            Use ``"medieval_text"`` to keep long s / rotunda r and only
            apply the additional substitutions on top.
        logs_dir: Optional directory for the plain-text run log.

    Returns:
        The Path to the saved ``labels.json``.
    """
    input_json = Path(input_json)
    augmented_folder = Path(augmented_folder)
    output_base_dir = Path(output_base_dir)
    substitutions = substitutions if substitutions is not None else dict(DEFAULT_SUBSTITUTIONS)
    assert substitutions, "substitutions is empty — pass a non-empty mapping"

    # Extract timestamp from "aug_<timestamp>" folder name so the label
    # output folder can be paired with it by the timestamp alone.
    folder_name = augmented_folder.name
    if folder_name.startswith("aug_"):
        timestamp = folder_name[len("aug_") :]
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"labels_{timestamp}"
    output_dir = output_base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger setup — file + console if logs_dir, console only otherwise.
    if logs_dir:
        logger, log_file = setup_label_correction_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("label_correction")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        log_file = None

    logger.info(f"=== Label correction started | Run: {run_name} ===")

    assert input_json.is_file(), f"Input JSON not found: {input_json}"
    assert augmented_folder.is_dir(), f"Augmented folder not found: {augmented_folder}"

    config_summary = {
        "run": run_name,
        "git": _get_git_commit(),
        "input_json": str(input_json),
        "augmented_folder": str(augmented_folder),
        "output_dir": str(output_dir),
        "substitutions": substitutions,
        "text_field": text_field,
    }
    logger.info(f"Config: {json.dumps(config_summary, ensure_ascii=False)}")

    # Load the source labels — accept either the nested medieval-text format
    # {"summary": ..., "labels": {...}} or a flat {image_name: {...}} dict.
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], dict):
        source_labels = data["labels"]
    else:
        source_labels = data

    trans_table = str.maketrans(substitutions)

    augmented_files = sorted(augmented_folder.glob("*.png"))
    assert augmented_files, f"No *.png files in {augmented_folder}"

    output_labels: dict[str, str] = {}
    n_unmatched_filename = 0
    n_missing_source = 0
    n_missing_text = 0

    for aug_file in augmented_files:
        match = _AUG_FILENAME_RE.match(aug_file.name)
        if not match:
            n_unmatched_filename += 1
            continue
        source_name = match.group(1) + ".png"
        source_entry = source_labels.get(source_name)
        if source_entry is None:
            n_missing_source += 1
            continue
        base_text = source_entry.get(text_field)
        if not base_text:
            n_missing_text += 1
            continue
        output_labels[aug_file.name] = base_text.translate(trans_table)

    output_path = output_dir / "labels.json"
    output_path.write_text(json.dumps(output_labels, indent=2, ensure_ascii=False))

    logger.info(
        f"Corrected {len(output_labels)} labels "
        f"(unmatched filename: {n_unmatched_filename}, "
        f"missing source: {n_missing_source}, "
        f"missing text: {n_missing_text})"
    )
    logger.info(f"Output JSON: {output_path}")
    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return output_path
