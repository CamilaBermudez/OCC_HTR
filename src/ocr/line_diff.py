"""Classify how a diplomatic OCR transcription differs from the scholarly edition.

Base = the **scholarly** edition (fully expanded, normalized, pure ASCII). We
describe what the **OCR** (diplomatic: keeps abbreviations ⁊/ꝑ/tildes, u/v,
manuscript lineation) does relative to it, per difference, in six categories:

  abbreviation  OCR uses a scribal abbreviation the edition expands (⁊→et,
                cõ→com, ꝑ→per, qͥ→qui). Signal: the OCR span carries a non-ASCII
                letter/combining mark (the edition is pure ASCII).
  orthographic  same word, different spelling — u/v, i/j, long-s, spacing, and
                line-break word splits (folded letters are equal).
  punctuation   editorial punctuation (, . : ; ¶) present on one side only.
  addition      material in the OCR not in the edition (over-generation / gloss).
  deletion      material in the edition the OCR omits (dropped letters/words).
  substitution  genuine divergence — an OCR misread or a real variant.

Diffing is **page-level**: the page's lines are concatenated into one stream on
each side and diffed continuously, so a word broken across manuscript lines
(``cau`` | ``sa`` vs edition ``causa``) resolves to *orthographic*, not a false
substitution. Each difference is then attributed back to the OCR line(s) it
falls in, so the viewer can highlight per line.

Editorial TEI is emitted per type (``<choice><abbr>/<expan>``,
``<choice><orig>/<reg>``, ``<add>``, ``<del>``, ``<sic>/<corr>``). Reusable:
:func:`diff_page` takes two plain line lists, independent of any particular
model or of the line-alignment step (spec §6.7).
"""

from __future__ import annotations

import difflib
import html
import unicodedata
from dataclasses import dataclass

_FOLD = str.maketrans({"v": "u", "j": "i", "ſ": "s", "ꝛ": "r"})


def _fold(text: str) -> str:
    """Reduce a span to comparable letters: NFKD, drop marks/punct/space, fold u/v i/j."""
    text = unicodedata.normalize("NFKD", text).lower().translate(_FOLD)
    return "".join(c for c in text if c.isascii() and c.isalnum())


def _despace(text: str) -> str:
    """The span with all whitespace removed (case preserved)."""
    return "".join(text.split())


def _is_subseq(short: str, long: str) -> bool:
    """True if ``short`` is a subsequence of ``long`` (letters dropped in order).

    Signals a scribal *contraction*: ``del`` ⊂ ``delo`` (``de lo``); a random OCR
    error like ``marors`` vs ``majors`` is NOT a subsequence.
    """
    it = iter(long)
    return all(ch in it for ch in short)


def tokenize(text: str) -> list[str]:
    """Split into word tokens (combining marks kept attached) + standalone punctuation.

    Combining marks (Unicode category ``M*``) join their base letter, so ``qͥt``
    is one token, not three — this is what the naive ``\\w+`` regex got wrong.
    """
    text = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    cur = ""
    for ch in text:
        if ch.isspace():
            if cur:
                tokens.append(cur)
                cur = ""
        elif unicodedata.category(ch)[0] == "P":  # punctuation -> standalone token
            if cur:
                tokens.append(cur)
                cur = ""
            tokens.append(ch)
        else:  # letter / digit / combining mark -> part of the current word
            cur += ch
    if cur:
        tokens.append(cur)
    return tokens


def _has_abbrev_mark(text: str) -> bool:
    """True if any char is a non-ASCII letter or combining mark (a brevigraph).

    Excludes punctuation like ``¶`` (category P) so structural marks don't count
    as abbreviations.
    """
    for c in text:
        if ord(c) > 127 and unicodedata.category(c)[0] in ("L", "M"):
            return True
    return False


def _is_punct(text: str) -> bool:
    """True if the span is all punctuation / whitespace (no letters or digits)."""
    stripped = [c for c in text if not c.isspace()]
    return bool(stripped) and all(unicodedata.category(c)[0] == "P" for c in stripped)


DiffType = str  # one of the six category names


def classify_region(base: str, ocr: str) -> DiffType:
    """Classify one aligned difference region. ``base`` = scholarly, ``ocr`` = OCR.

    Returns one of the six category names, or ``"spacing"`` for a pure
    whitespace/word-break difference (identical modulo whitespace) — the caller
    drops those, since a line-wrap is not an edit.
    """
    if not ocr:
        return "punctuation" if _is_punct(base) else "deletion"
    if not base:
        return "punctuation" if _is_punct(ocr) else "addition"
    if _despace(base) == _despace(ocr):
        return "spacing"  # only whitespace differs -> not a real difference
    if _has_abbrev_mark(ocr):
        return "abbreviation"
    fb, fo = _fold(base), _fold(ocr)
    # scribal contraction: OCR word(s) are a subsequence of the scholarly form,
    # keeping >=60% of the letters (del=de lo). Distinguishes a contraction from
    # a random OCR error (which is not a subsequence).
    if (
        fo
        and fb
        and fo != fb
        and len(fo) < len(fb)
        and 0.6 * len(fb) <= len(fo)
        and _is_subseq(fo, fb)
    ):
        return "abbreviation"
    if fb and fb == fo:
        return "orthographic"  # u/v, i/j, long-s
    if _is_punct(base) and _is_punct(ocr):
        return "punctuation"
    return "substitution"


def _tei(diff_type: DiffType, base: str, ocr: str) -> str:
    b, o = html.escape(base), html.escape(ocr)
    if diff_type == "abbreviation":
        return f"<choice><abbr>{o}</abbr><expan>{b}</expan></choice>"
    if diff_type == "orthographic":
        return f"<choice><orig>{o}</orig><reg>{b}</reg></choice>"
    if diff_type == "addition":
        return f"<add>{o}</add>"
    if diff_type == "deletion":
        return f"<del>{b}</del>"
    if diff_type == "punctuation":
        return f"<add>{o}</add>" if o and not b else f"<del>{b}</del>"
    return f"<sic>{o}</sic><corr>{b}</corr>"  # substitution


@dataclass(frozen=True)
class Diff:
    """One classified difference, attributed to an OCR line."""

    type: DiffType
    base_text: str  # scholarly span (the base)
    ocr_text: str  # OCR span
    ocr_line: int | None  # segmentation-line index this falls in (None if pure deletion at edge)
    tei: str

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "base_text": self.base_text,
            "ocr_text": self.ocr_text,
            "ocr_line": self.ocr_line,
            "tei": self.tei,
        }


def _wordchar(c: str) -> bool:
    """True for letters/digits/combining marks — i.e. not whitespace or punctuation."""
    return not c.isspace() and unicodedata.category(c)[0] != "P"


def _word_start(s: str, p: int) -> int:
    while p > 0 and _wordchar(s[p - 1]):
        p -= 1
    return p


def _word_end(s: str, p: int) -> int:
    while p < len(s) and _wordchar(s[p]):
        p += 1
    return p


def _expand(s: str, a: int, b: int) -> tuple[int, int]:
    """Grow ``[a, b)`` to whole-word bounds, but only if it contains a word char."""
    if a < b and any(_wordchar(c) for c in s[a:b]):
        return _word_start(s, a), _word_end(s, b)
    return a, b


def diff_page(scholarly_lines: list[str], ocr_lines: list[tuple[int, str]]) -> list[Diff]:
    """Diff a page: ``scholarly_lines`` (ordered) vs ``ocr_lines`` (``(seg_idx, text)``).

    **Character-level** alignment (``difflib`` on the concatenated page text on
    each side) — robust to spacing, contractions and repeated words where a
    token-level diff mis-anchors (``de ambulacio`` vs ``deambulacio``,
    ``en lu`` vs ``eulu``). Each non-equal char range is grown to whole-word
    boundaries and overlapping ranges merged into one region, then classified.
    Page-level so a word broken across manuscript lines resolves cleanly; each
    diff is attributed to the OCR line its span starts in.
    """
    base = " ".join(scholarly_lines)
    # OCR page string + per-char owner (segmentation line index)
    ocr_parts: list[str] = []
    owner: list[int] = []
    for seg_idx, text in ocr_lines:
        if ocr_parts:
            ocr_parts.append(" ")
            owner.append(seg_idx)
        ocr_parts.append(text)
        owner.extend([seg_idx] * len(text))
    ocr = "".join(ocr_parts)

    def owner_at(pos: int) -> int | None:
        if not owner:
            return None
        return owner[min(pos, len(owner) - 1)]

    sm = difflib.SequenceMatcher(a=base, b=ocr, autojunk=False)
    regions: list[list[int]] = []  # [base_start, base_end, ocr_start, ocr_end]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # grow to whole-word boundaries ONLY when the changed span actually
        # touches word characters; a pure whitespace/punctuation change must NOT
        # swallow its neighbouring words (else "en tot"->"entot" reads as a
        # deletion instead of a harmless space merge).
        bs, be = _expand(base, i1, i2)
        os_, oe = _expand(ocr, j1, j2)
        regions.append([bs, be, os_, oe])
    regions.sort()
    merged: list[list[int]] = []
    for r in regions:
        # merge only when two char-diffs land in the SAME word (their
        # word-expanded ranges genuinely overlap on BOTH sides) — never across a
        # gap, so scattered diffs stay separate and punctuation isn't absorbed.
        if merged and r[0] < merged[-1][1] and r[2] < merged[-1][3]:
            merged[-1][1] = max(merged[-1][1], r[1])
            merged[-1][3] = max(merged[-1][3], r[3])
        else:
            merged.append(r)

    diffs: list[Diff] = []
    for bs, be, os_, oe in merged:
        b, o = base[bs:be].strip(), ocr[os_:oe].strip()
        if not b and not o:
            continue  # nothing left after stripping (a pure whitespace change)
        dtype = classify_region(b, o)
        if dtype == "spacing":
            continue  # a line-wrap / pure spacing change is not an edit
        diffs.append(Diff(dtype, b, o, owner_at(os_), _tei(dtype, b, o)))
    return diffs
