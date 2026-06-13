"""Synthetic medieval-text image generation.
Apply probabilistic medieval orthographic substitutions:
   - long s (ſ, U+017F) replaces lowercase 's' at non-final positions;
     a separate (higher) probability is used for the start of a word vs.
     a mid-word position, so an occasional normal 's' still appears mid-word.
   - rotunda r (ꝛ, U+A75B) replaces lowercase 'r' immediately after a
     "round" letter (b/d/h/o/p/v/w/y).
   - tironian et (⁊, U+204A) appended once at the end of the line with a
     configurable probability — mirrors the reference manuscript where ⁊
     is a line-ending mark, never repeated mid-line. Because no standard
     font renders this glyph in the scribal hand of this manuscript, ⁊
     is composited into the rendered image from pre-cropped real-
     manuscript stamps (the label still stores the Unicode ⁊).
"""

import datetime
import json
import logging
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Round letters that trigger rotunda r when followed by 'r'. Standard
# medieval set: letters whose right-side bowl curves outward.
ROUND_LETTERS: set[str] = set("bdhopvwy")

LONG_S = "ſ"  # U+017F
ROTUNDA_R = "ꝛ"  # U+A75B
TIRONIAN_ET = "⁊"  # U+204A
# Lemma + colour for the capitulum rubric. When the literal word
# "Capitol" appears mid-text, the C is composited from a real-manuscript
# stamp (illuminated initial) and the rest of the word ("apitol") is
# rendered in rubric red. The label string keeps the plain "Capitol".
CAPITOL_WORD = "Capitol"
RUBRIC_RED: tuple[int, int, int] = (140, 30, 25)


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
    p_tironian_et: float = 0.0,
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
    result = "".join(out_parts)
    if p_tironian_et > 0.0:
        result = _insert_tironian_et(result, p_tironian_et, rng)
    return result


def _insert_tironian_et(text: str, p: float, rng: random.Random) -> str:
    """With probability ``p``, append ' ⁊' at the very end of ``text``.

    In the reference manuscript ⁊ appears as a line-ending mark — never
    more than once on a line, always after the last word. So this only
    appends a single trailing ⁊; mid-line sentence terminators are left
    alone. Idempotent: if ``text`` already ends in ⁊, it's returned
    unchanged.
    """
    stripped = text.rstrip()
    if not stripped or stripped.endswith(TIRONIAN_ET):
        return text
    if rng.random() < p:
        return f"{stripped} {TIRONIAN_ET}"
    return text


def load_glyph_stamps(
    stamp_dir: str | Path,
    target_height: int,
    fg: tuple[int, int, int] = (60, 40, 20),
    alpha_floor_gray: int = 210,
    alpha_full_gray: int = 80,
) -> list[Image.Image]:
    """Load all `*.png` / `*.jpg` crops from `stamp_dir`, derive a soft alpha
    from pixel darkness, recolour the ink to ``fg``, and resize so each
    stamp is `target_height` tall.

    Used for both Tironian-et (⁊) stamps and illuminated-C (Capitol)
    stamps — the only difference is the ``fg`` colour. The crops are
    expected to be small images of a glyph on parchment; the user crops
    these manually from manuscript scans. No transparency work needed
    beforehand: any pixel ≥ `alpha_floor_gray` becomes fully
    transparent, any pixel ≤ `alpha_full_gray` fully opaque, linear
    ramp in between.

    The stamp RGB is replaced with the rendering ``fg`` colour weighted
    by alpha. This is critical for downstream augmentation: the
    composite step reads the rendered image as a grayscale ink mask, and
    if each scribe's glyph kept its native (often faded) tone, the
    morphological erosion (p=0.85) was eating most of the stroke. With
    uniform fg-coloured ink, the stamp reads as densely as a regular
    rendered glyph, AND the colour signal (e.g. red for rubric) is
    preserved through composite_on_parchment's per-pixel-colour-aware
    ink multiplier.

    Returns an empty list if the directory is missing or empty — the
    caller can then fall back to skipping stamp insertion.
    """
    stamp_dir = Path(stamp_dir)
    if not stamp_dir.is_dir():
        return []
    paths = sorted(stamp_dir.glob("*.png")) + sorted(stamp_dir.glob("*.jpg"))
    stamps: list[Image.Image] = []
    span = max(1, alpha_floor_gray - alpha_full_gray)
    fg_rgb = np.array(fg, dtype=np.float32)
    for p in paths:
        try:
            raw = Image.open(p).convert("RGB")
        except Exception:
            continue
        arr = np.asarray(raw, dtype=np.float32)
        gray = arr.mean(axis=2)
        alpha = np.clip((alpha_floor_gray - gray) / span, 0.0, 1.0)
        # Recolour the stamp to the rendering ink colour. Background pixels
        # have alpha~0 so their RGB doesn't matter; the ink area gets a
        # uniform dark fg that survives erosion in the augmentation step.
        rgb_ink = np.broadcast_to(fg_rgb, arr.shape).copy()
        rgba = np.concatenate([rgb_ink, (alpha * 255.0)[..., None]], axis=2).astype(np.uint8)
        stamp = Image.fromarray(rgba, mode="RGBA")
        # Resize preserving aspect ratio.
        w, h = stamp.size
        new_w = max(1, int(round(w * target_height / max(h, 1))))
        stamp = stamp.resize((new_w, target_height), Image.LANCZOS)
        stamps.append(stamp)
    return stamps


def _tokenize_for_render(
    text: str,
    et_stamps: list[Image.Image] | None,
    c_stamps: list[Image.Image] | None,
    fg: tuple[int, int, int],
    fg_red: tuple[int, int, int],
    rng: random.Random,
) -> list[tuple]:
    """Convert ``text`` into a flat list of render tokens. Each token is:

      ("text", string, colour, pad_left, pad_right)
      ("stamp", PIL.Image, pad_left, pad_right)

    Matches:
      - the literal word ``Capitol`` (\\b-delimited): becomes a C stamp
        followed by ``apitol`` in ``fg_red``. If no c_stamps are
        provided, falls back to rendering the whole word in ``fg_red``.
      - the character ⁊: becomes a ⁊ stamp surrounded by small pads. If
        no et_stamps are provided, the ⁊ passes through to the font
        (which usually shows a missing-glyph box).
    """
    tokens: list[tuple] = []
    pattern = re.compile(rf"\b{re.escape(CAPITOL_WORD)}\b|{re.escape(TIRONIAN_ET)}")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            tokens.append(("text", text[pos : m.start()], fg, 0, 0))
        match = m.group(0)
        if match == TIRONIAN_ET:
            if et_stamps:
                tokens.append(("stamp", rng.choice(et_stamps), 1, 1))
            else:
                tokens.append(("text", TIRONIAN_ET, fg, 0, 0))
        else:  # "Capitol"
            if c_stamps:
                # C stamp absorbs the leading "C"; no right-pad so "apitol"
                # joins it as a single word visually.
                tokens.append(("stamp", rng.choice(c_stamps), 1, 0))
                tokens.append(("text", CAPITOL_WORD[1:], fg_red, 0, 0))
            else:
                # Fallback: just colour the word red.
                tokens.append(("text", CAPITOL_WORD, fg_red, 0, 0))
        pos = m.end()
    if pos < len(text):
        tokens.append(("text", text[pos:], fg, 0, 0))
    return tokens


def render_text_to_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    margin: int = 20,
    fg: tuple[int, int, int] = (60, 40, 20),
    bg: tuple[int, int, int] = (240, 230, 200),
    et_stamps: list[Image.Image] | None = None,
    c_stamps: list[Image.Image] | None = None,
    fg_red: tuple[int, int, int] = RUBRIC_RED,
    rng: random.Random | None = None,
) -> Image.Image:
    """Render ``text`` as a sepia-on-cream image sized to fit (with margin).

    Sized from the text's bounding box so each line image is just large
    enough to contain its content plus the margin — no wasted whitespace,
    no clipping.

    Two stamp paths are supported, both via ``_tokenize_for_render``:

      - ``⁊`` (Tironian et) is composited from one of ``et_stamps``.
      - The literal word ``Capitol`` becomes ``[C-stamp]`` + ``apitol``
        rendered in ``fg_red`` — matches the manuscript's rubric pattern
        where the illuminated C heads the chapter and the word's tail is
        in red rubric ink.

    When neither stamp path applies (no markers and no stamps), the
    function takes the original single-draw fast path.
    """
    rng = rng if rng is not None else random.Random()
    tokens = _tokenize_for_render(text, et_stamps, c_stamps, fg, fg_red, rng)

    # Fast path: a single plain text token in fg only — no stamps, no red.
    if len(tokens) == 1 and tokens[0][0] == "text" and tokens[0][2] == fg:
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        img_w = text_w + 2 * margin
        img_h = text_h + 2 * margin
        img = Image.new("RGB", (img_w, img_h), bg)
        draw = ImageDraw.Draw(img)
        draw.text((margin - bbox[0], margin - bbox[1]), text, fill=fg, font=font)
        return img

    # Mixed-token path: measure, build canvas, walk tokens.
    dummy = Image.new("RGB", (1, 1))
    measure = ImageDraw.Draw(dummy)
    max_top = 10**9
    max_bottom = -(10**9)
    metrics: list[tuple[int, int]] = []  # (advance_width, draw_x_offset)
    for t in tokens:
        if t[0] == "text":
            s = t[1]
            if not s:
                metrics.append((0, 0))
                continue
            b = measure.textbbox((0, 0), s, font=font)
            metrics.append((b[2] - b[0], -b[0]))
            max_top = min(max_top, b[1])
            max_bottom = max(max_bottom, b[3])
        else:  # stamp
            stamp = t[1]
            metrics.append((stamp.width, 0))
            max_top = min(max_top, 0)
            max_bottom = max(max_bottom, stamp.height)

    if max_top >= max_bottom:
        max_top, max_bottom = 0, font.size

    text_h = max_bottom - max_top
    pad_unit = max(4, text_h // 12)

    # Total advance width = sum(token advances) + sum(left+right pads * pad_unit).
    total_pad = sum((t[2] + t[3]) if t[0] == "stamp" else (t[3] + t[4]) for t in tokens) * pad_unit
    total_w = sum(m[0] for m in metrics) + total_pad + 2 * margin
    total_h = text_h + 2 * margin

    img = Image.new("RGBA", (total_w, total_h), bg + (255,))
    draw = ImageDraw.Draw(img)
    cursor_x = margin
    baseline_y = margin - max_top
    for i, t in enumerate(tokens):
        adv, dx = metrics[i]
        if t[0] == "text":
            _, s, colour, pad_l, pad_r = t
            cursor_x += pad_l * pad_unit
            if s:
                draw.text((cursor_x + dx, baseline_y), s, fill=colour, font=font)
                cursor_x += adv
            cursor_x += pad_r * pad_unit
        else:  # stamp
            _, stamp, pad_l, pad_r = t
            cursor_x += pad_l * pad_unit
            stamp_y = margin + max(0, (text_h - stamp.height) // 2)
            img.paste(stamp, (cursor_x, stamp_y), stamp)
            cursor_x += stamp.width + pad_r * pad_unit

    return img.convert("RGB")


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
    p_tironian_et: float = 0.0,
    et_stamp_dir: str | Path | None = None,
    c_stamp_dir: str | Path | None = None,
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
        "p_tironian_et": p_tironian_et,
        "et_stamp_dir": str(et_stamp_dir) if et_stamp_dir else None,
        "c_stamp_dir": str(c_stamp_dir) if c_stamp_dir else None,
        "base_seed": base_seed,
        "categories_filter": sorted(categories_filter) if categories_filter else None,
        "max_samples": max_samples,
        "n_samples_in_input": len(samples),
    }
    logger.info(f"Config: {json.dumps(config_summary)}")

    font = ImageFont.truetype(str(font_path), font_size)

    # Load ⁊ stamps once if both a directory and a non-zero probability
    # are provided. Stamp height is matched to font ascender so the glyph
    # sits naturally on the baseline. If the directory is missing or
    # empty, ⁊ insertion is silently disabled and the rendered text omits
    # the marker — better than crashing or shipping missing-glyph boxes.
    et_stamps: list[Image.Image] = []
    if p_tironian_et > 0.0 and et_stamp_dir:
        # 1.2× font size — the ⁊ in this manuscript ascends above x-height,
        # and the taller stamp gives the thin strokes enough budget to
        # survive the augmentation pipeline's erosion step.
        et_stamps = load_glyph_stamps(et_stamp_dir, target_height=int(font_size * 1.2))
        if not et_stamps:
            logger.warning(
                f"p_tironian_et={p_tironian_et} but no stamps found in "
                f"{et_stamp_dir}; ⁊ insertion disabled for this run."
            )
    effective_p_et = p_tironian_et if et_stamps else 0.0

    # Illuminated-C stamps for the "Capitol" rubric pattern. Recoloured to
    # RUBRIC_RED so downstream composite_on_parchment paints them in red
    # ink. Slightly taller than regular glyphs (1.3×) — initials in the
    # reference manuscript visibly ascend above the line.
    c_stamps: list[Image.Image] = []
    if c_stamp_dir:
        c_stamps = load_glyph_stamps(
            c_stamp_dir,
            target_height=int(font_size * 1.3),
            fg=RUBRIC_RED,
        )
        if not c_stamps:
            logger.warning(
                f"c_stamp_dir={c_stamp_dir} but no stamps found there; "
                "Capitol rubric stamps disabled for this run."
            )

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
            p_tironian_et=effective_p_et,
            rng=rng,
        )

        try:
            img = render_text_to_image(
                medieval_text,
                font=font,
                margin=margin,
                et_stamps=et_stamps or None,
                c_stamps=c_stamps or None,
                rng=rng,
            )
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
