"""Transcribe pre-segmented line images with a fine-tuned Swin+BERT TrOCR.

Loads a ``best_model/`` folder produced by ``src.ocr.trocr_finetune``
(``VisionEncoderDecoderModel`` + image processor + tokenizer) and emits one
``<stem>.txt`` per input image, mirroring the layout used by
``src.ocr.transcribe_img`` and ``src.ocr.medusa_transcribe`` so downstream
evaluation (``src.ocr.evaluate_ocr``) can score it identically to catmus /
kraken-finetune / medusa outputs.

Follows the same ``scripts/`` <-> ``src/`` split as the other OCR modules.
"""

import datetime
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "trocr_transcribe", run_name: str | None = None
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


def _detect_device(requested: str) -> str:
    """Resolve ``'auto'`` to mps > cuda > cpu, otherwise return ``requested`` as-is."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _collect_line_images(input_dir: Path) -> list[tuple[str, Path]]:
    """Walk ``input_dir`` and return ``(page_name, image_path)`` for every PNG.

    Supports both layouts:
      - flat folder of PNGs (returns ``("", path)`` for each)
      - per-page subdirectories holding PNGs (returns ``(subdir.name, path)``)
    """
    pairs: list[tuple[str, Path]] = []
    has_subdirs = any(p.is_dir() for p in input_dir.iterdir())
    if has_subdirs:
        for page_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
            for img in sorted(page_dir.glob("*.png")):
                pairs.append((page_dir.name, img))
    else:
        for img in sorted(input_dir.glob("*.png")):
            pairs.append(("", img))
    return pairs


def transcribe_trocr(
    model_dir: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    run_name: str,
    *,
    device: str = "auto",
    batch_size: int = 8,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    logs_dir: str | Path | None = None,
    task_name: str = "trocr_transcribe",
    log_config: bool = True,
) -> dict:
    """Transcribe every ``*.png`` under ``input_dir`` with a fine-tuned TrOCR.

    Args:
        model_dir: Folder holding the fine-tuned checkpoint. Must contain
            the ``VisionEncoderDecoderModel`` weights plus the image
            processor and tokenizer files — this is exactly the folder
            written by ``finetune_trocr`` at ``<run>/best_model/``.
        input_dir: Flat folder of ``*.png`` OR parent of per-page subdirs.
        output_dir: Output root; per-page subdirs land at
            ``<output_dir>/<run_name>/``.
        run_name: Run subdirectory name (also log basename).
        device: ``auto | mps | cuda | cpu``.
        batch_size: Images per forward pass.
        max_new_tokens: Generation cap per line.
        num_beams: Beam-search width. Overrides the checkpoint's default
            (which was set at training time).

    Returns:
        Dict with ``n_written``, ``n_skipped``, ``elapsed_s``, ``rate_lps``,
        ``save_dir``, ``log_path``.
    """
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    model_dir = Path(model_dir)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name

    logger, log_file = setup_simple_logging(logs_dir, task_name, run_name)

    assert model_dir.is_dir(), f"Model dir not found: {model_dir}"
    assert input_dir.is_dir(), f"Input dir not found: {input_dir}"

    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs = _collect_line_images(input_dir)
    assert pairs, f"No *.png files found under {input_dir}"

    resolved_device = _detect_device(device)

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
            "model_dir": str(model_dir),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "save_dir": str(save_dir),
            "n_lines": len(pairs),
            "device_requested": device,
            "device_resolved": resolved_device,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    logger.info("Loading TrOCR checkpoint: %s", model_dir)
    t0 = time.time()
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    model.to(resolved_device)
    model.eval()
    logger.info("Model + processor + tokenizer loaded in %.1fs", time.time() - t0)

    n_written = 0
    n_skipped = 0
    t_start = time.time()
    progress = tqdm(range(0, len(pairs), batch_size), desc="Transcribing", unit="batch")
    for batch_start in progress:
        batch_pairs = pairs[batch_start : batch_start + batch_size]
        try:
            images = [Image.open(p).convert("RGB") for _, p in batch_pairs]
        except Exception as exc:
            logger.error("Batch load failed at %s: %s", batch_pairs[0][1], exc)
            n_skipped += len(batch_pairs)
            continue
        pixel_values = image_processor(images=images, return_tensors="pt").pixel_values.to(
            resolved_device
        )
        try:
            with torch.no_grad():
                gen_ids = model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                )
        except Exception as exc:
            logger.error("Batch inference failed at %s: %s", batch_pairs[0][1], exc)
            n_skipped += len(batch_pairs)
            continue
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for (page, img_path), text in zip(batch_pairs, texts, strict=False):
            target_dir = save_dir / page if page else save_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"{img_path.stem}.txt").write_text(text.strip() + "\n", encoding="utf-8")
            n_written += 1
        progress.set_postfix(written=n_written, skipped=n_skipped)
    progress.close()

    elapsed = time.time() - t_start
    rate = n_written / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done: %d lines transcribed, %d skipped, %.0fs (%.2f lines/s)",
        n_written,
        n_skipped,
        elapsed,
        rate,
    )
    logger.info("Output: %s", save_dir)
    logger.info("Run log: %s", log_file)

    return {
        "n_written": n_written,
        "n_skipped": n_skipped,
        "elapsed_s": elapsed,
        "rate_lps": rate,
        "save_dir": str(save_dir),
        "log_path": log_file,
    }
