"""Strip chat-template leakage from Medusa 0.2 transcription outputs.

Roughly half of Medusa's raw ``.txt`` outputs on this corpus contain
Qwen3.5-VL chat-template artefacts and/or multiple transcription
attempts that weren't consumed during decoding. Two shapes show up:

1. Chat markers between two copies of the same transcription:

       yssi comensan las paraulas de
       user
       assistant
       <think>

       </think>

       yssi comensan las paraulas de

2. Multiple attempts, prompt echoes, or stray tokens after the first
   answer — often followed by a truncated re-attempt:

       que no erre la tua ma ⁊ le malaute si      <- first attempt
       string                                     <- stray marker
       que no erre la tua                         <- truncated re-attempt

For both shapes, empirically the FIRST non-artefact line is Medusa's
best transcription — the fullest, most accurate answer before the
model drifts. Everything after is noise, prompt echoes, or partial
regenerations. Taking only that first line collapses both patterns
back to one clean transcription.

Noise detection covers:
- Exact-match markers (``user``, ``assistant``, ``<think>``, ``</think>``,
  ``>``, ``I``, ``string``, ``response``, the prompt strings, etc.)
- Prefix rule: any line starting with ``>`` (quoted-reply / continuation
  artefact — matches ``>Et as veguadas ...`` even though the exact-match
  ``>`` rule doesn't).

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
# is treated as genuine transcription content (subject to the prefix
# rule in ``is_noise`` below). The prompt strings are hard-coded rather
# than read from run_medusa_transcribe.py because the model sometimes
# echoes the prompt regardless of what we passed in.
NOISE_LINES: frozenset[str] = frozenset(
    {
        "user",
        "assistant",
        "response",
        "<think>",
        "</think>",
        ">",
        "I",
        "string",
        "text",
        "output",
        "answer",
        "Transcribe the handwritten text in this line image.",
        "Output ONLY the transcription.",
        "Output ONLY the transcription",
    }
)


def is_noise(line: str) -> bool:
    """Return True if ``line`` is chat-template noise rather than transcription.

    Combines exact-match membership in ``NOISE_LINES`` with a prefix
    rule for lines starting with ``>`` (Qwen's quoted-reply artefact —
    catches ``>Et as veguadas ...`` which the exact-match ``>`` alone
    misses). Blank lines are also treated as noise.
    """
    if not line:
        return True
    if line in NOISE_LINES:
        return True
    if line.startswith(">"):
        return True
    return False


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
    """Return ``(cleaned_text, was_modified)`` — Medusa's first meaningful line.

    Walks the raw output line by line and returns the first line that
    isn't chat-template noise. Everything after is discarded: this drops
    prompt echoes, truncated re-attempts, near-duplicate variants, and
    ``>``-prefixed continuations that would otherwise inflate CER by
    making the compared string 2-3x longer than the reference.

    If every line is noise, the cleaned text is empty — a valid signal
    that Medusa produced nothing usable for that image (the evaluator
    then treats it as full-length substitution error).
    """
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not is_noise(line):
            cleaned = line + "\n"
            return cleaned, cleaned != raw
    return "", raw != ""


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
