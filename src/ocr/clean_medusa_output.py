"""Strip chat-template leakage from Medusa 0.2 transcription outputs.

Roughly half of Medusa's raw ``.txt`` outputs on this corpus contain
Qwen3.5-VL chat-template artefacts that weren't consumed during
decoding — patterns like:

    yssi comensan las paraulas de
    user
    assistant
    <think>

    </think>

    yssi comensan las paraulas de

The genuine transcription typically appears twice (once before the
leaked markers and once as the model's final answer after ``</think>``),
so after removing marker lines and empty lines, deduplicating
consecutive identical lines collapses the echo back to a single clean
line.

Cleaning is idempotent: files with no artefacts pass through untouched.

Follows the same ``scripts/`` <-> ``src/`` split as the other OCR
modules: the CLI wrapper in ``scripts/ocr/run_clean_medusa_output.py``
parses argparse, and this module hosts the logic + logging.
"""

import datetime
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

# Exact-match strings that indicate chat-template leakage. Anything else
# is treated as genuine transcription content. The prompt strings are
# hard-coded rather than read from run_medusa_transcribe.py because the
# model sometimes echoes the prompt regardless of what we passed in.
NOISE_LINES: frozenset[str] = frozenset(
    {
        "user",
        "assistant",
        "response",
        "<think>",
        "</think>",
        ">",
        "Transcribe the handwritten text in this line image.",
        "Output ONLY the transcription.",
    }
)


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "clean_medusa_output", run_name: str | None = None
):
    """File + console logger, same shape as the other src.ocr modules."""
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


def clean_text(raw: str) -> tuple[str, bool]:
    """Return ``(cleaned_text, was_modified)``.

    The cleaner is intentionally minimal so nothing unexpected survives:
    strip whitespace on every line, drop empty and marker-only lines,
    then collapse consecutive duplicates. That matches the observed
    "clean line -> markers -> clean line" pattern without needing to
    reason about ``</think>`` positions or generation order.
    """
    lines = [line.strip() for line in raw.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            continue
        if line in NOISE_LINES:
            continue
        if kept and kept[-1] == line:
            continue
        kept.append(line)
    cleaned = "\n".join(kept) + ("\n" if kept else "")
    return cleaned, cleaned != raw


def run_clean_medusa_output(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    logs_dir: str | Path | None = None,
    task_name: str = "clean_medusa_output",
    run_name: str | None = None,
    log_config: bool = True,
) -> dict:
    """Walk ``input_dir`` for ``*.txt`` files and clean any with artefacts.

    Args:
        input_dir: folder of Medusa ``.txt`` outputs (walked recursively).
        output_dir: if set, write cleaned files here mirroring the input
            layout. If ``None``, rewrite the input files in place.
        dry_run: report what would change without touching any files.
        logs_dir: log directory. Default: ``logs/clean_medusa_output``.
        run_name: log basename + summary run identifier.

    Returns:
        Dict with ``n_total``, ``n_changed``, ``in_place``, ``log_path``.
    """
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else None
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name
    if run_name is None:
        run_name = f"clean_medusa_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            "input_dir": str(input_dir),
            "output_dir": str(output_dir) if output_dir else None,
            "dry_run": dry_run,
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    assert input_dir.is_dir(), f"Not a directory: {input_dir}"
    in_place = output_dir is None

    n_total = n_changed = 0
    for src in sorted(input_dir.rglob("*.txt")):
        n_total += 1
        raw = src.read_text(encoding="utf-8")
        cleaned, modified = clean_text(raw)
        if modified:
            n_changed += 1
        if dry_run:
            continue
        if in_place:
            if modified:
                src.write_text(cleaned, encoding="utf-8")
        else:
            dst = output_dir / src.relative_to(input_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if modified:
                dst.write_text(cleaned, encoding="utf-8")
            else:
                shutil.copy2(src, dst)

    verb = "would clean" if dry_run else "cleaned"
    logger.info(
        "Scanned %d .txt files; %s %d that had chat-template artefacts.",
        n_total,
        verb,
        n_changed,
    )
    if not dry_run and not in_place:
        logger.info("Output written to: %s", output_dir)

    return {
        "n_total": n_total,
        "n_changed": n_changed,
        "in_place": in_place,
        "dry_run": dry_run,
        "log_path": log_file,
    }
