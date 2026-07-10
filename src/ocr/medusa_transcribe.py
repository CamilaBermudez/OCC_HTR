"""Transcribe pre-segmented line images with the ENC-PSL Medusa 0.2 VLM.

Medusa 0.2 (``ENC-PSL/Medusa0.2Line-9B``) is a 9B-parameter vision-language
model trained on 640k real + 500k synthetic medieval lines across the
Romance / Germanic / Celtic / Slavic families (9th-15th c.). It expects a
single pre-segmented line image and emits a CATMuS-compliant diplomatic
transcription. Reference: https://huggingface.co/ENC-PSL/Medusa0.2Line-9B

Output layout mirrors ``src.ocr.transcribe_img`` so the downstream
evaluation tooling (``src.ocr.evaluate_ocr``) can treat it like any other
transcription source: one ``<stem>.txt`` per input image, written under
``<output_dir>/<run_name>/<page>/`` (or the flat run folder when the
input is a flat folder of PNGs).

Follows the same ``scripts/`` <-> ``src/`` split as the other OCR modules:
the CLI wrapper in ``scripts/ocr/run_medusa_transcribe.py`` parses
argparse, and this module hosts the logic + logging + config dump.
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

# Compatibility shim — transformers 5.x's bitsandbytes integration calls
# model.set_submodule(...) which only exists in PyTorch >= 2.6. Our torch
# is pinned to 2.4.1 by kraken's dependency tree, so without this shim
# any --quantization run dies with
#   AttributeError: 'Qwen3_5ForConditionalGeneration' object has no
#   attribute 'set_submodule'
# The implementation matches the upstream method exactly: walk dotted
# path of module attributes, replace the leaf with the new module.
if not hasattr(torch.nn.Module, "set_submodule"):

    def _set_submodule(self: torch.nn.Module, target: str, module: torch.nn.Module) -> None:
        if target == "":
            raise ValueError("Cannot set the top-level module")
        atoms = target.split(".")
        parent = self
        for atom in atoms[:-1]:
            if not hasattr(parent, atom):
                raise AttributeError(f"{type(parent).__name__} has no attribute {atom!r}")
            parent = getattr(parent, atom)
        setattr(parent, atoms[-1], module)

    torch.nn.Module.set_submodule = _set_submodule

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

DEFAULT_MODEL_ID = "ENC-PSL/Medusa0.2Line-9B"
DEFAULT_PROMPT = (
    "Transcribe the handwritten text in this line image.\nOutput ONLY the transcription."
)


def setup_simple_logging(
    logs_dir: str | Path, task_name: str = "medusa_transcribe", run_name: str | None = None
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


def detect_device(requested: str) -> str:
    """Resolve 'auto' to mps > cuda > cpu, otherwise return ``requested`` as-is."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect_line_images(input_dir: Path) -> list[tuple[str, Path]]:
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


def transcribe_batch(
    images: list[Image.Image],
    *,
    model,
    processor,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Run a single forward pass on a list of PIL images, return raw transcriptions."""
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": img},
                ],
            }
        ]
        for img in images
    ]
    # `enable_thinking` only exists on transformers >= 4.50; fall back
    # gracefully so the module also runs on older versions where the
    # AutoModelForImageTextToText interface already works.
    chat_kwargs = dict(
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    try:
        inputs = processor.apply_chat_template(
            messages_batch, **chat_kwargs, enable_thinking=False
        ).to(device)
    except TypeError:
        inputs = processor.apply_chat_template(messages_batch, **chat_kwargs).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, input_len:]
    texts = processor.batch_decode(trimmed, skip_special_tokens=True)
    return [t.strip() for t in texts]


def _load_model_and_processor(
    model_id: str,
    device: str,
    quantization: str,
    logger: logging.Logger,
):
    """Load the Medusa processor + model, honouring the ``quantization`` flag.

    Returns ``(processor, model, dtype)``. Always BF16 — float32 would double
    the weight footprint (a 9B model in fp32 needs ~36 GB just for weights,
    which OOMs even on 32 GB instances).
    """
    dtype = torch.bfloat16
    # trust_remote_code=True lets the model's own processor/architecture
    # classes load when the installed transformers version doesn't ship
    # them natively. Medusa's processor uses a model-family-specific class
    # (Qwen3.5-VL style) that isn't in older transformers but is bundled
    # in the model repo itself.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    quant_config = None
    if quantization != "none":
        from transformers import BitsAndBytesConfig

        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8bit
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Quantization: %s (bitsandbytes)", quantization)

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        trust_remote_code=True,
    )
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        # bitsandbytes places layers itself and rejects .to() calls after load.
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(**load_kwargs)
    if quant_config is None:
        model = model.to(device)
    model.eval()
    return processor, model, dtype


def run_medusa_transcribe(
    input_dir: str | Path,
    output_dir: str | Path,
    run_name: str,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    batch_size: int = 4,
    max_new_tokens: int = 128,
    max_pages: int | None = None,
    prompt: str = DEFAULT_PROMPT,
    quantization: str = "none",
    logs_dir: str | Path | None = None,
    task_name: str = "medusa_transcribe",
    log_config: bool = True,
) -> dict:
    """Transcribe every ``*.png`` under ``input_dir`` with Medusa 0.2.

    Args:
        input_dir: flat folder of ``*.png`` OR parent of per-page subdirs.
        output_dir: root where the run subfolder is created.
        run_name: subfolder name under ``output_dir`` (and log basename).
        model_id: HuggingFace model id. Default: Medusa 0.2 Line 9B.
        device: ``auto | mps | cuda | cpu``. ``auto`` picks mps > cuda > cpu.
        batch_size: lines per forward pass.
        max_new_tokens: generation cap per line (CATMuS lines are <100 chars).
        max_pages: if set, only process the first N pages (subdirs).
        prompt: instruction passed to the VLM.
        quantization: ``none | 8bit | 4bit`` — bitsandbytes weight quant.
        logs_dir: log directory. Default: ``logs/medusa_transcribe``.
        run_name: run identifier + log basename.

    Returns:
        Dict with ``n_written``, ``n_skipped``, ``elapsed_s``, ``rate_lps``,
        ``save_dir``, ``log_path``.
    """
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir) if logs_dir else project_root / "logs" / task_name

    logger, log_file = setup_simple_logging(
        logs_dir=str(logs_dir), task_name=task_name, run_name=run_name
    )

    assert input_dir.is_dir(), f"Input dir not found: {input_dir}"
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_line_images(input_dir)
    assert pairs, f"No *.png files found under {input_dir}"

    if max_pages is not None:
        kept_pages: list[str] = []
        seen: set[str] = set()
        for page, _ in pairs:
            if page not in seen:
                if len(seen) >= max_pages:
                    break
                seen.add(page)
                kept_pages.append(page)
        pairs = [(p, i) for p, i in pairs if p in seen]
        logger.info("--max-pages=%s restricts to: %s", max_pages, kept_pages)

    resolved_device = detect_device(device)

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
            "n_lines": len(pairs),
            "model_id": model_id,
            "device_requested": device,
            "device_resolved": resolved_device,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "max_pages": max_pages,
            "quantization": quantization,
            "prompt": prompt,
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    if resolved_device == "cpu":
        logger.warning(
            "Running on CPU. 9B VLM inference will be very slow (~30-60s per line). "
            "Use --device mps on Apple Silicon, or cuda on NVIDIA, when possible."
        )

    logger.info("Loading model %s (this can take a minute on first run)...", model_id)
    t0 = time.time()
    processor, model, dtype = _load_model_and_processor(
        model_id=model_id,
        device=resolved_device,
        quantization=quantization,
        logger=logger,
    )
    logger.info(
        "Model loaded in %.1fs (dtype=%s, quantization=%s)",
        time.time() - t0,
        dtype,
        quantization,
    )

    n_written = 0
    n_skipped = 0
    t_start = time.time()
    progress = tqdm(range(0, len(pairs), batch_size), desc="Transcribing", unit="batch")
    for batch_start in progress:
        batch_pairs = pairs[batch_start : batch_start + batch_size]
        try:
            images = [Image.open(p).convert("RGB") for _, p in batch_pairs]
        except Exception as exc:
            logger.error("Failed to load batch starting at %s: %s", batch_pairs[0][1], exc)
            n_skipped += len(batch_pairs)
            continue
        try:
            texts = transcribe_batch(
                images,
                model=model,
                processor=processor,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=resolved_device,
            )
        except Exception as exc:
            logger.error("Batch inference failed at %s: %s", batch_pairs[0][1], exc)
            n_skipped += len(batch_pairs)
            continue
        for (page, img_path), text in zip(batch_pairs, texts, strict=False):
            target_dir = save_dir / page if page else save_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"{img_path.stem}.txt").write_text(text + "\n", encoding="utf-8")
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
