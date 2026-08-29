"""Compute character- and word-level accuracy for OCR prediction folders
against a hand-verified ground-truth folder of ``<stem>.gt.txt`` files.

Follows the same ``scripts/`` <-> ``src/`` split as ``transcribe_img`` and
``dictionary_evaluation``: the CLI wrapper in
``scripts/ocr/run_evaluate_ocr.py`` parses argparse and applies defaults,
and this module hosts the logic + logging so notebooks or other scripts
can reuse the workflow programmatically.

Metrics reported:
- CER (character error rate) and 1 - CER (character accuracy) — the
  training-time ``val_accuracy`` from ketos is the same quantity.
- WER (word error rate) and 1 - WER (word accuracy) — matches
  ``val_word_accuracy``.
- Corpus-level aggregate: sum of edits / sum of reference units (a
  length-weighted average). This is the standard OCR headline number.
- Per-line median: less sensitive to a small number of catastrophically
  bad lines, useful as a sanity check next to the corpus number.

Ground-truth filenames end in ``.gt.txt``; prediction folders may be flat
or per-page (kraken/catmus output style) — ``find_pred`` handles both.
"""

import csv
import datetime
import json
import logging
import os
import statistics
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.distance import Levenshtein


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "evaluate_ocr", run_name: str | None = None
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


@dataclass
class LineEval:
    stem: str
    model: str
    n_chars_ref: int
    n_words_ref: int
    edit_chars: int
    edit_words: int
    cer: float
    wer: float


def normalise(text: str) -> str:
    """Strip surrounding whitespace, collapse internal whitespace, NFC."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def find_pred(pred_dir: Path, stem: str) -> Path | None:
    """Find the prediction file for a stem — flat first, then recursive.

    Prediction folders may be flat (``<dir>/<stem>.txt``) or nested by
    page (``<dir>/<page>/<stem>.txt`` — the ``run_transcribe_img.py``
    layout). We prefer the flat lookup for speed and fall back to
    ``rglob`` so both layouts work without configuration.
    """
    flat = pred_dir / f"{stem}.txt"
    if flat.is_file():
        return flat
    matches = list(pred_dir.rglob(f"{stem}.txt"))
    return matches[0] if matches else None


def char_error(ref: str, hyp: str) -> tuple[int, int]:
    """Return (edit_distance, len(ref)) at the character level."""
    return Levenshtein.distance(ref, hyp), len(ref)


def word_error(ref: str, hyp: str) -> tuple[int, int]:
    """Return (edit_distance, len(ref_words)) at the word level."""
    ref_w = ref.split()
    hyp_w = hyp.split()
    return Levenshtein.distance(ref_w, hyp_w), len(ref_w)


def eval_one_model(
    model_name: str, pred_dir: Path, gt_lines: dict[str, str], logger: logging.Logger
) -> tuple[list[LineEval], dict]:
    """Score one model against every ground-truth line; return per-line + aggregate."""
    per_line: list[LineEval] = []
    n_missing = n_empty_ref = 0
    total_chars_ref = total_words_ref = 0
    total_edit_chars = total_edit_words = 0

    for stem, ref in gt_lines.items():
        if not ref:
            n_empty_ref += 1
            continue
        pred_path = find_pred(pred_dir, stem)
        if pred_path is None:
            n_missing += 1
            continue
        hyp = normalise(pred_path.read_text(encoding="utf-8"))
        ec, nc = char_error(ref, hyp)
        ew, nw = word_error(ref, hyp)
        # Clip the edit distance at the reference length: a line can be at most 100% wrong, so
        # over-production (hyp longer than ref, edit > ref-len) can't push CER/WER above 1 nor
        # accuracy below 0. edit_chars/edit_words keep the raw distance for diagnostics.
        ec_c, ew_c = min(ec, nc), min(ew, nw)
        per_line.append(
            LineEval(
                stem=stem,
                model=model_name,
                n_chars_ref=nc,
                n_words_ref=nw,
                edit_chars=ec,
                edit_words=ew,
                cer=ec_c / nc if nc else 0.0,
                wer=ew_c / nw if nw else 0.0,
            )
        )
        total_chars_ref += nc
        total_words_ref += nw
        total_edit_chars += ec_c
        total_edit_words += ew_c

    agg = {
        "model": model_name,
        "n_lines_scored": len(per_line),
        "n_skipped_missing_pred": n_missing,
        "n_skipped_empty_ref": n_empty_ref,
        "cer_corpus": total_edit_chars / total_chars_ref if total_chars_ref else float("nan"),
        "wer_corpus": total_edit_words / total_words_ref if total_words_ref else float("nan"),
        "cer_median": statistics.median(e.cer for e in per_line) if per_line else float("nan"),
        "wer_median": statistics.median(e.wer for e in per_line) if per_line else float("nan"),
        "char_acc_corpus": 1 - (total_edit_chars / total_chars_ref)
        if total_chars_ref
        else float("nan"),
        "word_acc_corpus": 1 - (total_edit_words / total_words_ref)
        if total_words_ref
        else float("nan"),
    }
    logger.info(
        "Model %s: %d lines scored, %d missing pred, %d empty ref | " "char_acc=%.4f word_acc=%.4f",
        model_name,
        agg["n_lines_scored"],
        agg["n_skipped_missing_pred"],
        agg["n_skipped_empty_ref"],
        agg["char_acc_corpus"],
        agg["word_acc_corpus"],
    )
    return per_line, agg


def render_markdown(aggs: list[dict]) -> str:
    """Build the summary markdown table users paste into thesis text."""
    header = (
        "| model | lines | CER | char_acc | WER | word_acc | " "CER_median | WER_median | missing |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for a in aggs:
        rows.append(
            f"| {a['model']} "
            f"| {a['n_lines_scored']} "
            f"| {a['cer_corpus']:.4f} "
            f"| {a['char_acc_corpus']:.4f} "
            f"| {a['wer_corpus']:.4f} "
            f"| {a['word_acc_corpus']:.4f} "
            f"| {a['cer_median']:.4f} "
            f"| {a['wer_median']:.4f} "
            f"| {a['n_skipped_missing_pred']} |"
        )
    return "\n".join(rows)


def run_evaluate_ocr(
    gt_dir: str | Path,
    predictions: list[tuple[str, str | Path]],
    output_dir: str | Path | None = None,
    logs_dir: str | Path | None = None,
    task_name: str = "evaluate_ocr",
    run_name: str | None = None,
    log_config: bool = True,
) -> dict:
    """Score every ``predictions`` folder against the ``gt_dir`` reference.

    Args:
        gt_dir: folder of ``<stem>.gt.txt`` reference files.
        predictions: list of ``(display_name, pred_folder)`` pairs.
        output_dir: root for the per-run artefact folder. Default:
            ``tests/ocr/evaluations`` under PROJECT_ROOT.
        logs_dir: log directory. Default: ``logs/evaluate_ocr``.
        run_name: subdirectory name for artefacts + log basename.

    Returns:
        Dict with ``aggs`` (list of per-model aggregate dicts), and paths
        to the written ``csv`` / ``md`` / ``log`` artefacts.
    """
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    gt_dir = Path(gt_dir)
    output_dir = Path(output_dir) if output_dir else project_root / "tests/ocr/evaluations"
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name
    if run_name is None:
        run_name = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            "gt_dir": str(gt_dir),
            "predictions": [{"name": n, "path": str(p)} for n, p in predictions],
            "output_dir": str(output_dir),
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    assert gt_dir.is_dir(), f"Ground-truth folder not found: {gt_dir}"
    pred_entries: list[tuple[str, Path]] = []
    for name, path in predictions:
        p = Path(path)
        assert p.is_dir(), f"Prediction folder not found: {p}"
        pred_entries.append((name, p))
    assert pred_entries, "At least one prediction folder is required."

    gt_lines: dict[str, str] = {}
    for f in sorted(gt_dir.glob("*.gt.txt")):
        stem = f.name[: -len(".gt.txt")]
        gt_lines[stem] = normalise(f.read_text(encoding="utf-8"))
    assert gt_lines, f"No *.gt.txt files in {gt_dir}"
    logger.info("Loaded %d ground-truth lines from %s", len(gt_lines), gt_dir)

    all_per_line: list[LineEval] = []
    aggs: list[dict] = []
    for name, path in pred_entries:
        logger.info("Scoring %s from %s", name, path)
        per_line, agg = eval_one_model(name, path, gt_lines, logger)
        all_per_line.extend(per_line)
        aggs.append(agg)

    run_out = output_dir / run_name
    run_out.mkdir(parents=True, exist_ok=True)
    csv_path = run_out / f"{run_name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "stem",
                "model",
                "n_chars_ref",
                "n_words_ref",
                "edit_chars",
                "edit_words",
                "cer",
                "wer",
            ]
        )
        for e in all_per_line:
            w.writerow(
                [
                    e.stem,
                    e.model,
                    e.n_chars_ref,
                    e.n_words_ref,
                    e.edit_chars,
                    e.edit_words,
                    f"{e.cer:.6f}",
                    f"{e.wer:.6f}",
                ]
            )

    md = render_markdown(aggs)
    md_path = run_out / f"{run_name}.md"
    md_path.write_text(
        f"# OCR evaluation — {run_name}\n\n"
        f"Ground truth: `{gt_dir}` ({len(gt_lines)} lines)\n\n"
        "Models compared:\n"
        + "\n".join(f"- `{n}` from `{p}`" for n, p in pred_entries)
        + "\n\n"
        + md
        + "\n\n_CER = character error rate (corpus-level: sum of edits / "
        "sum of reference characters). char_acc = 1 - CER. Median = per-line "
        "median, less sensitive to one bad line. Missing = GT stems with no "
        "prediction file from that model._\n",
        encoding="utf-8",
    )

    # Also mirror the full aggregate table + per-model dicts into the log
    # file. That way the log alone is enough to recover the numbers if
    # the CSV / MD artefacts are lost or moved.
    logger.info("=== Aggregate results ===\n%s", md)
    logger.info("Per-model aggregates (JSON): %s", json.dumps(aggs, indent=2))
    logger.info("Per-line CSV: %s", csv_path)
    logger.info("Summary MD  : %s", md_path)
    print()
    print(md)
    print()
    print(f"Per-line CSV : {csv_path}")
    print(f"Summary MD   : {md_path}")

    return {
        "aggs": aggs,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "log_path": log_file,
    }
