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
from tqdm import tqdm

# Round letters that trigger rotunda r when followed by 'r'. Standard
# medieval set: letters whose right-side bowl curves outward.
ROUND_LETTERS: set[str] = set("bdhopvwy")

LONG_S = "ſ"  # U+017F
ROTUNDA_R = "ꝛ"  # U+A75B
TIRONIAN_ET = "⁊"  # U+204A
# Scribal abbreviation marks. Each entry maps a lowercase trigger letter
# to one or more (stamp_folder, label, has_descender) variants. When a
# trigger letter fires (probability p_abbreviation), one variant is
# chosen at random: its `label` string replaces the trigger char in the
# medieval_text, and its `stamp_folder` is composited at that position
# during rendering. The `has_descender` flag tells the renderer to shift
# the stamp down by the font's descent so the base letter sits on the
# baseline (otherwise a p/q stamp's descender forces the whole glyph to
# float above where it should be).
#
# Why stamps and not combining-mark fonts? Most of these glyphs use
# combining diacritics (U+0303, U+0363, U+0365, etc.) that the current
# fonts in fonts/ don't carry — they render as missing-glyph boxes.
# Using real-manuscript crops lets the visual match the scribe's hand
# exactly.
ABBREV_MAP: dict[str, list[tuple[str, str, bool]]] = {
    "e": [("e_tilde", "ẽ", False)],
    "l": [("l_tilde", "l'", False)],
    "m": [("m_tilde", "m̃", False)],
    "n": [("n_tilde", "ñ", False)],
    "o": [("o_tilde", "õ", False)],
    "p": [("p_tilde", "p̃", True)],
    "q": [
        ("q_circle", "q°", True),
        ("q_i", "qͥ", True),
        ("q_tilde", "q̃", True),
    ],
    "r": [("r_tilde", "r̃", False)],
}
# Per-folder height multiplier applied on top of `font_size` when loading
# abbreviation stamps. Defaults to 1.0 (the original target). Override
# for folders whose source crops put the tilde unusually close to the
# base letter — in those crops the trimmed bbox is mostly base-letter,
# so resizing to the same target height makes the base letter visibly
# taller than the surrounding x-height. e_tilde and o_tilde have h/w
# bbox ratios around 1.35 (vs ~1.05 for n_tilde / m_tilde), which is
# exactly that situation; 0.85x brings them back in line.
ABBREV_HEIGHT_SCALE: dict[str, float] = {
    "e_tilde": 0.55,
    "o_tilde": 0.55,
    "m_tilde": 0.55,
    "r_tilde": 0.55,
}
# Lemma + colour for the capitulum rubric. When the literal word
# "Capitol" appears mid-text, the C is composited from a real-manuscript
# stamp (illuminated initial) and the rest of the word ("apitol") is
# rendered in rubric red. The label string keeps the plain "Capitol".
CAPITOL_WORD = "Capitol"
RUBRIC_RED: tuple[int, int, int] = (140, 30, 25)

# Pattern-substring stamps: scribal-abbreviation ligatures and special
# letters that cover multiple base characters (syllables) or whole-word
# matches. Each entry: (regex, stamp_folder, probability, has_descender).
# When a pattern matches in the rendered text the substring is replaced
# IN THE IMAGE by a stamp from the folder, with the given probability.
# The label keeps the matched substring as-is so the model learns the
# mapping from the visual stamp to the original characters.
#
# Sorted longest-first so multi-char patterns win over their substrings
# (e.g. "cum" beats "um", "an" beats "n" — though "n" isn't in this
# table; it's handled by ABBREV_MAP which uses a separate code path).
PATTERN_STAMPS_CFG: list[tuple[str, str, float, bool]] = [
    (r"cum", "cum", 0.80, False),
    (r"\bo\b", "O_", 1.00, False),  # standalone "o" (space-o-space)
    (r"am", "am", 0.80, False),
    (r"an", "an", 0.80, False),
    (r"au", "au", 0.80, False),
    (r"em", "em", 0.80, False),
    (r"ma", "ma", 0.80, False),
    (r"me", "me", 0.80, False),
    (r"mi", "mi", 0.80, False),
    (r"mu", "mu", 0.80, False),
    (r"nu", "nu", 0.80, False),
    (r"um", "um", 0.80, False),
    (r"un", "un", 0.80, False),
    (r"x", "x", 1.00, False),
]

# Purely-decorative line-end glyph. When p_end_decor fires, a stamp from
# this folder is composited at the very end of the rendered line. The
# label is NOT modified — this is a visual mark only.
END_DECOR_FOLDER = "end_decor"


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
    p_abbreviation: float = 0.0,
    max_abbreviation_per_line: int = 3,
    max_abbreviation_per_word: int = 1,
    round_letters: set[str] = ROUND_LETTERS,
    rng: random.Random | None = None,
) -> str:
    rng = rng if rng is not None else random.Random()
    out_parts: list[str] = []
    # Cap the total number of scribal-abbreviation substitutions per line.
    # Without this, a long line with many trigger letters (e/n/m/o/q/r…)
    # ends up with 5–8 abbreviations even at modest per-char probability,
    # which crowds the result well past real-manuscript density.
    abbrev_count = 0
    # Split keeping whitespace separators so we can identify word boundaries.
    for tok in re.split(r"(\s+)", text):
        if not tok or tok.isspace():
            out_parts.append(tok)
            continue
        chars = list(tok)
        n = len(chars)
        # Per-word cap. Even at the line level we'd often see two
        # abbreviations land in the same word (e.g. "autr̃ẽiat" — both r̃
        # and ẽ firing), which crowds the glyphs visually and is not how
        # the reference manuscript distributes them. Resetting this
        # counter per token keeps a clean one-mark-per-word density.
        abbrev_in_word = 0
        for i, ch in enumerate(chars):
            if ch == "s" and i < n - 1:
                p = p_long_s_begin if i == 0 else p_long_s_middle
                if rng.random() < p:
                    chars[i] = LONG_S
                    continue
            if ch == "r" and i > 0 and chars[i - 1].lower() in round_letters:
                # Rotunda-r wins precedence for round-letter+r; if it
                # doesn't fire the r stays plain (no abbreviation fallback,
                # because rͣ in that position would be visually odd).
                if rng.random() < p_rotunda_r:
                    chars[i] = ROTUNDA_R
                    continue
            if (
                p_abbreviation > 0.0
                and ch in ABBREV_MAP
                and abbrev_count < max_abbreviation_per_line
                and abbrev_in_word < max_abbreviation_per_word
            ):
                if rng.random() < p_abbreviation:
                    # Pick a variant; only the label matters for the
                    # string. The stamp folder is consumed at render time.
                    chars[i] = rng.choice(ABBREV_MAP[ch])[1]
                    abbrev_count += 1
                    abbrev_in_word += 1
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
    trim_whitespace: bool = False,
    source_alpha_threshold: int = 200,
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
            raw = Image.open(p).convert("RGBA")
        except Exception:
            continue
        arr_rgba = np.asarray(raw, dtype=np.float32)
        arr = arr_rgba[..., :3]
        gray = arr.mean(axis=2)
        # Darkness-derived alpha: parchment greys → transparent, dark ink
        # → opaque. Works for the older rectangular crops where the whole
        # PNG is opaque against a light parchment background.
        darkness_alpha = np.clip((alpha_floor_gray - gray) / span, 0.0, 1.0)
        # Source alpha (from a non-rectangular crop or an editor that
        # already exported transparency). Where the user explicitly cut
        # away background, this is 0 — wins over darkness keying, so
        # transparent borders aren't accidentally darkened back in by
        # the recolour step.
        #
        # A non-rectangular crop done in an image editor produces an
        # effectively binary mask (alpha=255 inside, 0 outside) with a
        # 1–2 pixel anti-aliased band at the cut edge. Those boundary
        # pixels carry mid-range alpha *and* mid-range grey, so the
        # darkness keying treats them as faint ink — recoloured to dark
        # brown, they show up as a halo around the glyph. Snapping any
        # source alpha below `source_alpha_threshold` to 0 hardens the
        # cut edge. Fully-opaque inputs (alpha=255 everywhere) are
        # unaffected: they still fall through to pure darkness keying.
        source_alpha_raw = arr_rgba[..., 3]
        source_alpha = np.where(
            source_alpha_raw >= source_alpha_threshold,
            source_alpha_raw / 255.0,
            0.0,
        )
        alpha = np.minimum(darkness_alpha, source_alpha)
        if trim_whitespace:
            # Crop to the bounding box of the ink area so the resized stamp
            # fills target_height with ink rather than border whitespace.
            # Critical for inline abbreviation stamps: with whitespace
            # padding, bottom-aligning to baseline lifts the base letter
            # above where the surrounding glyphs sit.
            # Threshold 0.20 (not 0.05) so faint anti-aliasing greys at
            # the crop edges don't keep border padding alive — that was
            # making inconsistently-cropped source stamps render at
            # different effective sizes after resize.
            mask = alpha > 0.20
            if mask.any():
                row_any = np.any(mask, axis=1)
                col_any = np.any(mask, axis=0)
                y0, y1 = int(np.argmax(row_any)), int(len(row_any) - np.argmax(row_any[::-1]))
                x0, x1 = int(np.argmax(col_any)), int(len(col_any) - np.argmax(col_any[::-1]))
                arr = arr[y0:y1, x0:x1]
                alpha = alpha[y0:y1, x0:x1]
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
    e_stamps: list[Image.Image] | None,
    abbrev_stamps: dict[str, list[Image.Image]] | None,
    abbrev_descenders: set[str] | None,
    pattern_stamps: list[tuple[str, list[Image.Image], float, bool]] | None,
    end_decor_stamps: list[Image.Image] | None,
    p_end_decor: float,
    fg: tuple[int, int, int],
    fg_red: tuple[int, int, int],
    rng: random.Random,
    p_capital_e: float = 0.0,
) -> list[tuple]:
    """Convert ``text`` into a flat list of render tokens. Each token is:

      ("text", string, colour, pad_left, pad_right)
      ("stamp", PIL.Image, pad_left, pad_right)

    Matches (in pattern priority order):
      - the literal word ``Capitol`` (\\b-delimited): becomes a C stamp
        followed by ``apitol`` in ``fg_red``. Always fires when c_stamps
        are provided. Falls back to the word in ``fg_red`` otherwise.
      - the character ⁊: becomes a ⁊ stamp surrounded by small pads. If
        no et_stamps are provided, the ⁊ passes through to the font
        (which usually shows a missing-glyph box).
      - any multi-letter word starting with E or e: with probability
        ``p_capital_e`` becomes an E stamp followed by the word's tail
        in regular ``fg`` (the manuscript's enlarged E is not a rubric
        — it's the same iron-gall ink as the body, just larger).
        Single-letter "e" (the conjunction "and") is excluded by the
        ``[A-Za-z]+`` tail.
      - any scribal-abbreviation label present in ``abbrev_stamps`` keys
        (e.g. ñ, q̃, õ, …): becomes a stamp from the matching pool.
        The presence of an abbreviation label in ``text`` is taken as
        signal that the substitution step already fired; the renderer
        always stamps it. Labels are matched longest-first so that
        e.g. ``q°`` wins over the leading ``q`` alone.
    """
    tokens: list[tuple] = []
    parts = [
        rf"\b{re.escape(CAPITOL_WORD)}\b",
        re.escape(TIRONIAN_ET),
        r"\b[Ee][A-Za-z]+",
    ]
    if abbrev_stamps:
        # Longest first so multi-char sequences win over partial matches.
        for label in sorted(abbrev_stamps.keys(), key=len, reverse=True):
            parts.append(re.escape(label))
    # Pattern stamps (syllable ligatures, standalone-o, x). The regex
    # engine scans left-to-right and picks the FIRST alternative that
    # matches at each starting position — so a pattern like "am" at
    # position 0 of "am̃i" would greedily consume "a"+"m" before the
    # engine ever gets to position 1, where the m̃ abbreviation lives,
    # leaving the combining tilde (U+0303) orphaned and rendered as a
    # missing-glyph box. The negative lookahead below prevents pattern
    # stamps from matching when their final character is followed by a
    # combining diacritic, so abbreviation labels always win in that
    # case. The lookahead is appended in code (not baked into
    # PATTERN_STAMPS_CFG) so the config stays readable.
    NO_COMBINING_NEXT = r"(?![̀-ͯ])"
    if pattern_stamps:
        for ps in pattern_stamps:
            parts.append(ps[0] + NO_COMBINING_NEXT)
    pattern = re.compile("|".join(parts))

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
        elif match == CAPITOL_WORD:
            if c_stamps:
                # C stamp absorbs the leading "C"; no right-pad so "apitol"
                # joins it as a single word visually.
                tokens.append(("stamp", rng.choice(c_stamps), 1, 0))
                tokens.append(("text", CAPITOL_WORD[1:], fg_red, 0, 0))
            else:
                tokens.append(("text", CAPITOL_WORD, fg_red, 0, 0))
        elif abbrev_stamps and match in abbrev_stamps:
            # 5th element selects vertical alignment.
            #   "baseline-descender" for p̃, q̃, q°, qͥ — letters whose
            #     base glyph has a descender. The stamp must shift DOWN
            #     by the font's descent or the base letter floats above
            #     the baseline (because the descender portion of the
            #     stamp would otherwise sit at the baseline instead).
            #   "baseline" for non-descender letters (n, m, e, o, l, r):
            #     stamp BOTTOM aligned to baseline.
            if abbrev_descenders and match in abbrev_descenders:
                align = "baseline-descender"
            else:
                align = "baseline"
            tokens.append(("stamp", rng.choice(abbrev_stamps[match]), 0, 0, align))
        elif re.fullmatch(r"[Ee][A-Za-z]+", match):
            if e_stamps and rng.random() < p_capital_e:
                # Brown stamp + brown tail — illuminated E is a size feature
                # in this manuscript, not a colour rubric.
                tokens.append(("stamp", rng.choice(e_stamps), 1, 0))
                tokens.append(("text", match[1:], fg, 0, 0))
            else:
                tokens.append(("text", match, fg, 0, 0))
        else:
            # Try pattern stamps. Iterate by regex (in declaration order
            # so longer entries listed first in PATTERN_STAMPS_CFG win)
            # and use the first one whose full regex matches the match.
            handled = False
            if pattern_stamps:
                for ps in pattern_stamps:
                    pat_regex, ps_stamps, ps_prob, ps_has_desc = ps
                    if re.fullmatch(pat_regex, match):
                        if rng.random() < ps_prob:
                            align = "baseline-descender" if ps_has_desc else "baseline"
                            tokens.append(("stamp", rng.choice(ps_stamps), 0, 0, align))
                        else:
                            # Probability didn't fire: render as plain text.
                            tokens.append(("text", match, fg, 0, 0))
                        handled = True
                        break
            if not handled:
                # Defensive: shouldn't reach here. Fall back to plain text.
                tokens.append(("text", match, fg, 0, 0))
        pos = m.end()
    if pos < len(text):
        tokens.append(("text", text[pos:], fg, 0, 0))

    # End-of-line decoration: a purely-visual stamp pasted at the very
    # end of the rendered line with probability p_end_decor. The label
    # is NOT modified — the model is expected to learn to ignore this
    # mark since it has no corresponding character in the transcription.
    # Skip when the line already ends with ⁊ (Tironian et): the et mark
    # IS itself a line terminator in this manuscript, so doubling it up
    # with another end-of-line decoration looks redundant.
    line_ends_with_et = text.rstrip().endswith(TIRONIAN_ET)
    if end_decor_stamps and not line_ends_with_et and rng.random() < p_end_decor:
        tokens.append(("stamp", rng.choice(end_decor_stamps), 1, 0))

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
    e_stamps: list[Image.Image] | None = None,
    abbrev_stamps: dict[str, list[Image.Image]] | None = None,
    abbrev_descenders: set[str] | None = None,
    pattern_stamps: list[tuple[str, list[Image.Image], float, bool]] | None = None,
    end_decor_stamps: list[Image.Image] | None = None,
    p_end_decor: float = 0.0,
    p_capital_e: float = 0.0,
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
    tokens = _tokenize_for_render(
        text,
        et_stamps,
        c_stamps,
        e_stamps,
        abbrev_stamps,
        abbrev_descenders,
        pattern_stamps,
        end_decor_stamps,
        p_end_decor,
        fg,
        fg_red,
        rng,
        p_capital_e,
    )

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
    def _stamp_pad(t):
        return t[2] + t[3]

    def _text_pad(t):
        return t[3] + t[4]

    total_pad = sum(_stamp_pad(t) if t[0] == "stamp" else _text_pad(t) for t in tokens) * pad_unit
    total_w = sum(m[0] for m in metrics) + total_pad + 2 * margin
    total_h = text_h + 2 * margin

    img = Image.new("RGBA", (total_w, total_h), bg + (255,))
    draw = ImageDraw.Draw(img)
    cursor_x = margin
    # PIL's draw.text((x, y), ...) anchors the bbox TOP at y. To put each
    # token's bbox top at y=margin (in the canvas), we draw at:
    #     y_draw = margin - max_top
    # The actual text BASELINE (where the bottom of 'n' sits) is one
    # ascent below the top of the bbox of an ascender letter; computed
    # from the font's metrics so it doesn't depend on which letters
    # happen to be in the line.
    y_draw = margin - max_top
    text_baseline = y_draw + font.getmetrics()[0]
    for i, t in enumerate(tokens):
        adv, dx = metrics[i]
        if t[0] == "text":
            _, s, colour, pad_l, pad_r = t
            cursor_x += pad_l * pad_unit
            if s:
                draw.text((cursor_x + dx, y_draw), s, fill=colour, font=font)
                cursor_x += adv
            cursor_x += pad_r * pad_unit
        else:  # stamp
            # Optional 5th element selects vertical alignment:
            #   "center"   (default) — vertically centred in the text band,
            #               used for decorative initials (⁊, C, E) where
            #               the stamp is itself the whole glyph.
            #   "baseline" — stamp BOTTOM sits on the text baseline so the
            #               base letter (n in ñ, etc.) lines up with
            #               surrounding lowercase letters; the diacritic
            #               sticks up above x-height naturally.
            _, stamp, pad_l, pad_r = t[:4]
            alignment = t[4] if len(t) >= 5 else "center"
            cursor_x += pad_l * pad_unit
            if alignment in ("baseline", "baseline-descender"):
                if alignment == "baseline-descender":
                    # Descender letters (p, q): shift the stamp DOWN by
                    # the font's descent so the base letter rests on the
                    # baseline and the descender hangs below it, like a
                    # native lowercase p/q would.
                    descent_shift = font.getmetrics()[1]
                    stamp_y = text_baseline + descent_shift - stamp.height
                else:
                    stamp_y = text_baseline - stamp.height
                # Small visual nudge left to compensate for the side-
                # bearing that was lost when we trimmed whitespace from
                # the source crop. Keep it modest — a larger overlap
                # crowds adjacent stamps (rẽteñ-style clusters).
                bearing = max(1, text_h // 40)
                img.paste(stamp, (cursor_x - bearing, stamp_y), stamp)
                cursor_x += stamp.width - bearing + pad_r * pad_unit
            else:
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
    p_capital_e: float = 0.0,
    e_stamp_dir: str | Path | None = None,
    p_abbreviation: float = 0.0,
    abbrev_base_dir: str | Path | None = None,
    max_abbreviation_per_line: int = 3,
    max_abbreviation_per_word: int = 1,
    enable_pattern_stamps: bool = False,
    p_end_decor: float = 0.0,
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
        "p_capital_e": p_capital_e,
        "e_stamp_dir": str(e_stamp_dir) if e_stamp_dir else None,
        "p_abbreviation": p_abbreviation,
        "abbrev_base_dir": str(abbrev_base_dir) if abbrev_base_dir else None,
        "max_abbreviation_per_line": max_abbreviation_per_line,
        "max_abbreviation_per_word": max_abbreviation_per_word,
        "enable_pattern_stamps": enable_pattern_stamps,
        "p_end_decor": p_end_decor,
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

    # Illuminated-E stamps for sentence-starting words. Loaded in the
    # regular iron-gall brown fg — unlike the C-Capitol rubric, the
    # decorative E in this manuscript is just an enlarged version of the
    # normal body ink, so the rest of the E-word also reads as normal
    # text (no red tail). Fires only when both a stamp pool and a non-
    # zero probability are configured. 1.3× font height matches the
    # ascender of the manuscript's larger initial.
    e_stamps: list[Image.Image] = []
    if p_capital_e > 0.0 and e_stamp_dir:
        e_stamps = load_glyph_stamps(
            e_stamp_dir,
            target_height=int(font_size * 1.3),
            # Default fg=(60, 40, 20) — brown ink, same as the body text.
        )
        if not e_stamps:
            logger.warning(
                f"p_capital_e={p_capital_e} but no stamps found in "
                f"{e_stamp_dir}; E-word initial disabled for this run."
            )
    effective_p_e = p_capital_e if e_stamps else 0.0

    # Scribal-abbreviation stamps. Each variant in ABBREV_MAP points at a
    # subfolder under `abbrev_base_dir`; load whichever folders exist.
    # Sized to 1.15× font height — base char sits on baseline and the
    # diacritic ascends a little above the regular ascender line.
    abbrev_stamps: dict[str, list[Image.Image]] = {}
    abbrev_descenders: set[str] = set()
    if p_abbreviation > 0.0 and abbrev_base_dir:
        abbrev_base = Path(abbrev_base_dir)
        for trigger, variants in ABBREV_MAP.items():
            for folder, label, has_descender in variants:
                stamp_dir = abbrev_base / folder
                if has_descender:
                    abbrev_descenders.add(label)
                # 1.0× font height after a tight whitespace trim (default),
                # with per-folder overrides in ABBREV_HEIGHT_SCALE for crops
                # where the diacritic-to-letter spacing inflates the base
                # letter (e_tilde, o_tilde). Renderer handles descender
                # letters (p, q) separately by shifting them down by font
                # descent so the base letter sits on the baseline.
                scale = ABBREV_HEIGHT_SCALE.get(folder, 1.0)
                stamps = load_glyph_stamps(
                    stamp_dir,
                    target_height=int(font_size * scale),
                    trim_whitespace=True,
                )
                if stamps:
                    abbrev_stamps[label] = stamps
                else:
                    logger.warning(
                        f"abbreviation '{label}' (trigger '{trigger}'): "
                        f"no stamps in {stamp_dir}, skipping"
                    )
    effective_p_abbrev = p_abbreviation if abbrev_stamps else 0.0

    # Pattern stamps (syllable ligatures + standalone-o + x). Loaded from
    # PATTERN_STAMPS_CFG. 0.6× font height (vs 1.0× for single-letter
    # abbreviations) because these are pure-body ligatures: the crops
    # contain x-height letters with at most a small flourish above, no
    # full ascender/descender like p̃ / q̃ have. At 1.0× the stamp body
    # was ~1.7× the surrounding 'n' x-height and visually dominated the
    # line; 0.6× brings the base letters in line with body text while
    # still leaving room for the diacritic flourish above x-height.
    pattern_stamps: list[tuple[str, list[Image.Image], float, bool]] = []
    if enable_pattern_stamps and abbrev_base_dir:
        abbrev_base = Path(abbrev_base_dir)
        for pat_regex, folder, prob, has_desc in PATTERN_STAMPS_CFG:
            stamps = load_glyph_stamps(
                abbrev_base / folder,
                target_height=int(font_size * 0.6),
                trim_whitespace=True,
            )
            if stamps:
                pattern_stamps.append((pat_regex, stamps, prob, has_desc))
            else:
                logger.warning(
                    f"pattern stamp /{pat_regex}/ ({folder}): "
                    f"no stamps in {abbrev_base / folder}, skipping"
                )

    # End-of-line decoration: optional purely-visual mark with no label
    # contribution. Slightly taller than body text — these decorations
    # often extend a bit above x-height in the manuscript.
    end_decor_stamps: list[Image.Image] = []
    if p_end_decor > 0.0 and abbrev_base_dir:
        end_decor_stamps = load_glyph_stamps(
            Path(abbrev_base_dir) / END_DECOR_FOLDER,
            target_height=int(font_size * 1.10),
            trim_whitespace=True,
        )
        if not end_decor_stamps:
            logger.warning(
                f"p_end_decor={p_end_decor} but no stamps in "
                f"{Path(abbrev_base_dir) / END_DECOR_FOLDER}; "
                "end-of-line decoration disabled."
            )
    effective_p_end_decor = p_end_decor if end_decor_stamps else 0.0

    labels: dict[str, dict] = {}
    rendered = 0
    skipped = 0
    total = min(len(samples), max_samples) if max_samples is not None else len(samples)
    progress = tqdm(samples.items(), total=total, desc="Rendering", unit="img")
    for idx, (sample_id, info) in enumerate(progress):
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
            p_abbreviation=effective_p_abbrev,
            max_abbreviation_per_line=max_abbreviation_per_line,
            max_abbreviation_per_word=max_abbreviation_per_word,
            rng=rng,
        )

        try:
            img = render_text_to_image(
                medieval_text,
                font=font,
                margin=margin,
                et_stamps=et_stamps or None,
                c_stamps=c_stamps or None,
                e_stamps=e_stamps or None,
                abbrev_stamps=abbrev_stamps or None,
                abbrev_descenders=abbrev_descenders or None,
                pattern_stamps=pattern_stamps or None,
                end_decor_stamps=end_decor_stamps or None,
                p_end_decor=effective_p_end_decor,
                p_capital_e=effective_p_e,
                rng=rng,
            )
        except Exception as exc:
            logger.error(f"Render failed for {sample_id}: {exc}")
            skipped += 1
            progress.set_postfix(rendered=rendered, skipped=skipped)
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
        progress.set_postfix(rendered=rendered, skipped=skipped)

    progress.close()
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
