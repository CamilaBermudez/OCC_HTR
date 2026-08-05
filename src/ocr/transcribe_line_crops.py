"""Transcribe a flat folder of pre-cropped line PNGs with a kraken model.

Unlike ``src.ocr.transcribe_img`` (page-based, requires segmentation
JSONs + a per-page folder structure + an inventory CSV), this module
runs directly on a flat folder of ``<stem>.png`` line crops. Intended
for evaluation flows where we want predictions on just a benchmark
subset (e.g. the 300 permanent-val lines) without regenerating the
full-corpus transcription.

Baseline handling: pre-cropped line images no longer carry the original
page-space baseline coordinates. For CTC-based kraken recognisers, the
baseline is only used as an orientation / rectification anchor, so we
synthesise a horizontal line at ~2/3 of the crop's height. Output is a
flat folder of ``<stem>.txt`` mirroring the layout ``run_evaluate_ocr``
expects when scored against a flat ``.gt.txt`` folder such as
``data/processed/annotated_samples/OCR/validation/``.

Follows the same ``scripts/`` <-> ``src/`` split as the other OCR
modules: [scripts/ocr/run_transcribe_line_crops.py](../../scripts/ocr/run_transcribe_line_crops.py)
is the argparse wrapper; this file hosts the logic + logging + config
dump.
"""

import datetime
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from kraken import rpred
from kraken.containers import BaselineLine, Segmentation
from kraken.lib import models
from PIL import Image
from tqdm import tqdm


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "transcribe_line_crops", run_name: str | None = None
):
    """File + console logger, same shape as the other ``src.ocr`` modules."""
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


def _synthesised_seg(im_path: Path) -> tuple[Image.Image, Segmentation]:
    """Return ``(PIL image, kraken Segmentation)`` for one pre-cropped line.

    The boundary spans the whole crop; the baseline is a horizontal line
    at ~2/3 height. That's where the ink of a typical CATMuS line sits
    relative to the crop, and it's what the CTC recogniser expects for
    normalisation.
    """
    im = Image.open(im_path).convert("L")
    w, h = im.size
    y = max(1, int(h * 0.7))
    line = BaselineLine(
        id="0",
        baseline=[(0, y), (w - 1, y)],
        boundary=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
    )
    seg = Segmentation(
        type="baselines",
        imagename=str(im_path),
        text_direction="horizontal-lr",
        script_detection=False,
        lines=[line],
    )
    return im, seg


def transcribe_line_crops(
    input_dir: str | Path,
    output_dir: str | Path,
    run_name: str,
    model_path: str | Path,
    *,
    device: str = "cpu",
    logs_dir: str | Path | None = None,
    task_name: str = "transcribe_line_crops",
    log_config: bool = True,
) -> dict:
    """Run kraken over every ``*.png`` in ``input_dir`` (flat folder).

    Args:
        input_dir: folder of ``<stem>.png`` line crops.
        output_dir: root under which ``<run_name>/`` is created (mirroring
            the layout used by ``run_transcribe_img``).
        run_name: subfolder name under ``output_dir`` + log basename.
        model_path: path to the ``.mlmodel`` to load.
        device: ``cpu`` or ``cuda:0``. Kraken is CPU-friendly.
        logs_dir: log directory. Default: ``logs/<task_name>``.
        log_config: dump JSON config to the log at start.

    Returns:
        Dict with ``n_written``, ``n_failed``, ``n_empty``, ``elapsed_s``,
        ``save_dir``, ``log_path``.
    """
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_path = Path(model_path)
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name

    logger, log_file = setup_simple_logging(
        logs_dir=str(logs_dir), task_name=task_name, run_name=run_name
    )

    assert input_dir.is_dir(), f"Input dir not found: {input_dir}"
    assert model_path.is_file(), f"Model not found: {model_path}"
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(input_dir.glob("*.png"))
    assert pngs, f"No *.png files in {input_dir}"

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
            "output_dir": str(output_dir),
            "save_dir": str(save_dir),
            "model_path": str(model_path),
            "device": device,
            "n_lines": len(pngs),
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        # Provenance travels with the predictions (logs often aren't pulled with them).
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "_provenance.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    logger.info("Loading kraken model: %s", model_path)
    model = models.load_any(str(model_path), device=device)

    n_written = 0
    n_failed = 0
    n_empty = 0
    t_start = time.time()
    for p in tqdm(pngs, desc="Transcribing", unit="line"):
        try:
            im, seg = _synthesised_seg(p)
            preds = list(rpred.rpred(model, im, seg))
        except Exception as exc:
            logger.error("Failed to transcribe %s: %s", p.name, exc)
            n_failed += 1
            continue
        text = ""
        if preds:
            text = getattr(preds[0], "prediction", "") or ""
        (save_dir / f"{p.stem}.txt").write_text(text + "\n", encoding="utf-8")
        n_written += 1
        if not text:
            n_empty += 1

    elapsed = time.time() - t_start
    rate = n_written / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done: %d written (%d empty), %d failed, %.1fs (%.2f lines/s)",
        n_written,
        n_empty,
        n_failed,
        elapsed,
        rate,
    )
    logger.info("Output: %s", save_dir)
    logger.info("Run log: %s", log_file)

    return {
        "n_written": n_written,
        "n_failed": n_failed,
        "n_empty": n_empty,
        "elapsed_s": elapsed,
        "save_dir": str(save_dir),
        "log_path": log_file,
    }
