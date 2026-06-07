"""Synthetic medieval-text image generation.
Apply probabilistic medieval orthographic substitutions:
   - long s (ſ, U+017F) replaces lowercase 's' at non-final positions;
     a separate (higher) probability is used for the start of a word vs.
     a mid-word position, so an occasional normal 's' still appears mid-word.
   - rotunda r (ꝛ, U+A75B) replaces lowercase 'r' immediately after a
     "round" letter (b/d/h/o/p/v/w/y).
"""

import datetime
import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Round letters that trigger rotunda r when followed by 'r'. Standard
# medieval set: letters whose right-side bowl curves outward.
ROUND_LETTERS: set[str] = set("bdhopvwy")

LONG_S = "ſ"  # U+017F
ROTUNDA_R = "ꝛ"  # U+A75B


def setup_medieval_text_logging(logs_dir: str | Path, run_name: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"{run_name}_medieval_text.log"

    logger = logging.getLogger("medieval_text")
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


def apply_medieval_substitutions(
    text: str,
    *,
    p_long_s_begin: float = 0.9,
    p_long_s_middle: float = 0.80,
    p_rotunda_r: float = 0.70,
    round_letters: set[str] = ROUND_LETTERS,
    rng: random.Random | None = None,
) -> str:
    rng = rng if rng is not None else random.Random()
    out_parts: list[str] = []
    # Split keeping whitespace separators so we can identify word boundaries.
    for tok in re.split(r"(\s+)", text):
        if not tok or tok.isspace():
            out_parts.append(tok)
            continue
        chars = list(tok)
        n = len(chars)
        for i, ch in enumerate(chars):
            if ch == "s" and i < n - 1:
                p = p_long_s_begin if i == 0 else p_long_s_middle
                if rng.random() < p:
                    chars[i] = LONG_S
            elif ch == "r" and i > 0 and chars[i - 1].lower() in round_letters:
                if rng.random() < p_rotunda_r:
                    chars[i] = ROTUNDA_R
        out_parts.append("".join(chars))
    return "".join(out_parts)


def render_text_to_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    margin: int = 20,
    fg: tuple[int, int, int] = (60, 40, 20),
    bg: tuple[int, int, int] = (240, 230, 200),
) -> Image.Image:
    """Render ``text`` as a sepia-on-cream image sized to fit (with margin).

    The canvas is sized from the text's bounding box so each line image is
    just large enough to contain its content plus the margin — no wasted
    whitespace, no clipping.
    """
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img_w = text_w + 2 * margin
    img_h = text_h + 2 * margin

    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)
    # Anchor so the text's actual bounding box starts at (margin, margin).
    draw.text((margin - bbox[0], margin - bbox[1]), text, fill=fg, font=font)
    return img


def _sanitize_for_filename(sample_id: str) -> str:
    """Convert a 'Filename.txt:lineno' sample_id to a filesystem-safe stem.

    Example: 'Additional_10323.txt:5' -> 'Additional_10323_l00005'.
    """
    file_part, lineno = sample_id.rsplit(":", 1)
    stem = Path(file_part).stem
    return f"{stem}_l{int(lineno):05d}"


def generate_medieval_text_dataset(
    input_json: str | Path,
    output_dir: str | Path,
    run_name: str,
    font_path: str | Path,
    *,
    font_size: int = 60,
    margin: int = 20,
    p_long_s_begin: float = 0.95,
    p_long_s_middle: float = 0.80,
    p_rotunda_r: float = 0.70,
    base_seed: int = 42,
    categories_filter: set[str] | None = None,
    max_samples: int | None = None,
    logs_dir: str | Path | None = None,
) -> Path:
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    font_path = Path(font_path)
    assert font_path.is_file(), f"Font not found: {font_path}"

    if logs_dir:
        logger, log_file = setup_medieval_text_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("medieval_text")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        log_file = None

    logger.info(f"=== Medieval-text generation started | Run: {run_name} ===")

    doc = json.loads(input_json.read_text(encoding="utf-8"))
    samples = doc.get("samples") or {}
    assert samples, f"No samples found under 'samples' key in {input_json}"

    config_summary = {
        "run": run_name,
        "git": _get_git_commit(),
        "input_json": str(input_json),
        "output_dir": str(save_dir),
        "font_path": str(font_path),
        "font_size": font_size,
        "margin": margin,
        "p_long_s_begin": p_long_s_begin,
        "p_long_s_middle": p_long_s_middle,
        "p_rotunda_r": p_rotunda_r,
        "base_seed": base_seed,
        "categories_filter": sorted(categories_filter) if categories_filter else None,
        "max_samples": max_samples,
        "n_samples_in_input": len(samples),
    }
    logger.info(f"Config: {json.dumps(config_summary)}")

    font = ImageFont.truetype(str(font_path), font_size)

    labels: dict[str, dict] = {}
    rendered = 0
    skipped = 0
    for idx, (sample_id, info) in enumerate(samples.items()):
        if max_samples is not None and rendered >= max_samples:
            break

        cats = info.get("categories", [])
        if categories_filter and not (set(cats) & categories_filter):
            continue

        text = info.get("text", "")
        if not text:
            skipped += 1
            continue

        rng = random.Random(base_seed + idx)
        medieval_text = apply_medieval_substitutions(
            text,
            p_long_s_begin=p_long_s_begin,
            p_long_s_middle=p_long_s_middle,
            p_rotunda_r=p_rotunda_r,
            rng=rng,
        )

        try:
            img = render_text_to_image(medieval_text, font=font, margin=margin)
        except Exception as exc:
            logger.error(f"Render failed for {sample_id}: {exc}")
            skipped += 1
            continue

        stem = _sanitize_for_filename(sample_id)
        out_name = f"{stem}.png"
        img.save(save_dir / out_name)

        labels[out_name] = {
            "sample_id": sample_id,
            "categories": cats,
            "original_text": text,
            "medieval_text": medieval_text,
            "seed": base_seed + idx,
        }
        rendered += 1

        if rendered % 500 == 0:
            logger.info(f"Progress: {rendered} rendered | {skipped} skipped")

    labels_path = save_dir / "labels.json"
    labels_payload = {
        "summary": {
            **config_summary,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "rendered": rendered,
            "skipped": skipped,
        },
        "labels": labels,
    }
    labels_path.write_text(json.dumps(labels_payload, indent=2, ensure_ascii=False))

    logger.info(f"Medieval-text generation complete: {rendered} rendered, " f"{skipped} skipped")
    logger.info(f"Output dir: {save_dir}")
    logger.info(f"Labels JSON: {labels_path}")
    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return save_dir
