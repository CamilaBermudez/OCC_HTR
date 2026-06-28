"""Transcribe line-image crops with the ENC-PSL Medusa 0.2 VLM HTR model.

Medusa is a 9B-parameter vision-language model trained on 640k real +
500k synthetic medieval lines across Romance / Germanic / Celtic /
Slavic language families (9th-15th c.). It expects a single
pre-segmented line image and emits a CATMuS-compliant diplomatic
transcription. Reference: https://huggingface.co/ENC-PSL/Medusa0.2Line-9B

Output layout matches the existing ``run_transcribe_img.py``: one
``<stem>.txt`` per line image under ``--output-dir/<run_name>/<page>/``,
so the downstream benchmark / evaluation tooling can treat it like any
other transcription source.

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/run_medusa_transcribe.py \\
        --input-dir ./data/processed/extracted_lines/extraction_20260618_154440 \\
        --output-dir ./data/processed/transcription \\
        --device mps \\
        --max-pages 5
"""

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL_ID = "ENC-PSL/Medusa0.2Line-9B"
DEFAULT_PROMPT = (
    "Transcribe the handwritten text in this line image.\n" "Output ONLY the transcription."
)


def setup_logging(logs_dir: Path, run_name: str) -> tuple[logging.Logger, Path]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{run_name}_medusa.log"
    logger = logging.getLogger("medusa_transcribe")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for h in (logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger, log_file


def detect_device(requested: str) -> str:
    """Resolve 'auto' to mps > cuda > cpu, otherwise return requested as-is."""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def collect_line_images(input_dir: Path) -> list[tuple[str, Path]]:
    """Walk ``input_dir`` and return (page_name, image_path) for every PNG.

    Supports both layouts:
      - flat folder of PNGs (returns ("", path) for each)
      - per-page subdirectories holding PNGs (returns (subdir.name, path))
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
    # gracefully so the script also runs on older versions where the
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


def main():
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(
        description=(
            "Run the ENC-PSL Medusa 0.2 medieval HTR VLM on a folder of "
            "pre-segmented line images. Output layout mirrors the existing "
            "kraken/catmus transcription pipeline so the evaluation tooling "
            "can compare predictions across models on identical inputs."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder of line images. Either a flat folder of *.png OR a parent "
        "folder containing per-page subdirs of *.png (same layout as "
        "data/processed/extracted_lines/extraction_<ts>/).",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Output root. Per-page subdirs land at output-dir/<run-name>/. "
        "Default: data/processed/transcription",
    )
    parser.add_argument(
        "--run-name",
        required=False,
        help="Run subdirectory name. Default: medusa_<timestamp>.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id. Default: {DEFAULT_MODEL_ID}.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | mps | cuda | cpu. Default auto picks mps > cuda > cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Lines per forward pass. Lower this if out-of-memory on MPS "
        "(BF16 inference of a 9B model is memory-heavy). Default: 4.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Generation cap per line (CATMuS lines are typically <100 chars). " "Default: 128.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="If set, only process the first N pages (subdirs). Useful for a "
        "smoke test before committing to a full-corpus run.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="The instruction passed to the VLM. The model card explicitly "
        "warns that results degrade if the prompt is modified, so override "
        "at your own risk.",
    )
    parser.add_argument(
        "--logs-dir",
        required=False,
        help="Plain-text log directory. Default: logs/medusa_transcribe",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    assert input_dir.is_dir(), f"Input dir not found: {input_dir}"
    output_dir = (
        Path(args.output_dir) if args.output_dir else project_root / "data/processed/transcription"
    )
    logs_dir = Path(args.logs_dir) if args.logs_dir else project_root / "logs/medusa_transcribe"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"medusa_{stamp}"
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger, log_file = setup_logging(logs_dir, run_name)
    logger.info(f"=== Medusa transcription started | Run: {run_name} ===")

    pairs = collect_line_images(input_dir)
    assert pairs, f"No *.png files found under {input_dir}"

    if args.max_pages is not None:
        kept_pages = []
        seen = set()
        for page, _ in pairs:
            if page not in seen:
                if len(seen) >= args.max_pages:
                    break
                seen.add(page)
                kept_pages.append(page)
        pairs = [(p, i) for p, i in pairs if p in seen]
        logger.info(f"--max-pages={args.max_pages} restricts to: {kept_pages}")

    device = detect_device(args.device)
    logger.info(
        f"Config: input={input_dir} | n_lines={len(pairs)} | device={device} | "
        f"batch={args.batch_size} | model={args.model_id}"
    )
    if device == "cpu":
        logger.warning(
            "Running on CPU. 9B VLM inference will be very slow (~30-60s per line). "
            "Use --device mps on Apple Silicon, or cuda on NVIDIA, when possible."
        )

    logger.info(f"Loading model {args.model_id} (this can take a minute on first run)...")
    t0 = time.time()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    # trust_remote_code=True lets the model's own processor/architecture
    # classes load when the installed transformers version doesn't ship
    # them natively. Medusa's processor uses a model-family-specific class
    # (Qwen2.5-VL / Glm-V style) that isn't in transformers 4.46.x but
    # is bundled in the model repo itself.
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s (dtype={dtype})")

    n_written = 0
    n_skipped = 0
    t_start = time.time()
    progress = tqdm(range(0, len(pairs), args.batch_size), desc="Transcribing", unit="batch")
    for batch_start in progress:
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        try:
            images = [Image.open(p).convert("RGB") for _, p in batch_pairs]
        except Exception as exc:
            logger.error(f"Failed to load batch starting at {batch_pairs[0][1]}: {exc}")
            n_skipped += len(batch_pairs)
            continue
        try:
            texts = transcribe_batch(
                images,
                model=model,
                processor=processor,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
        except Exception as exc:
            logger.error(f"Batch inference failed at {batch_pairs[0][1]}: {exc}")
            n_skipped += len(batch_pairs)
            continue
        for (page, img_path), text in zip(batch_pairs, texts):
            target_dir = save_dir / page if page else save_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"{img_path.stem}.txt").write_text(text + "\n", encoding="utf-8")
            n_written += 1
        progress.set_postfix(written=n_written, skipped=n_skipped)
    progress.close()

    elapsed = time.time() - t_start
    rate = n_written / elapsed if elapsed > 0 else 0
    logger.info(
        f"Done: {n_written} lines transcribed, {n_skipped} skipped, "
        f"{elapsed:.0f}s ({rate:.2f} lines/s)"
    )
    logger.info(f"Output: {save_dir}")
    logger.info(f"Run log: {log_file}")


if __name__ == "__main__":
    sys.exit(main())
