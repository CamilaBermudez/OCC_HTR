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
    """Classify one non-equal region. ``base`` = scholarly span, ``ocr`` = OCR span."""
    if _fold(base) and _fold(base) == _fold(ocr):
        # same letters after folding: u/v, i/j, spacing, or a line-break word split
        return "orthographic"
    if ocr and not base:
        return "punctuation" if _is_punct(ocr) else "addition"
    if base and not ocr:
        return "punctuation" if _is_punct(base) else "deletion"
    # both sides non-empty (a true replace)
    if _has_abbrev_mark(ocr):
        return "abbreviation"
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


def _refine_replace(
    base_toks: list[str], ocr_toks: list[str], j0: int
) -> list[tuple[DiffType, str, str, int]]:
    """Word-level refine of one replace region -> ``(type, base_span, ocr_span, j)``.

    ``j`` is the global OCR-token index used for line attribution. Greedy: pairs
    tokens by fold-equality, grows the OCR side to absorb line-break word splits
    (``dispo`` + ``sicios`` == ``disposicios``), and only leaves genuinely
    divergent tokens as abbreviation / substitution.
    """
    out: list[tuple[DiffType, str, str, int]] = []
    i = j = 0
    while i < len(base_toks) or j < len(ocr_toks):
        if i < len(base_toks) and j < len(ocr_toks):
            b, o = base_toks[i], ocr_toks[j]
            if _fold(b) == _fold(o):
                if b != o:  # same letters, cosmetic diff (u/v, case): orthographic
                    out.append(("orthographic", b, o, j0 + j))
                i, j = i + 1, j + 1
                continue
            # spacing/word-break: one word on one side == several on the other
            # (split: base == concat of OCR tokens; merge: OCR == concat of base)
            grew = False
            for k in range(j + 1, min(j + 4, len(ocr_toks)) + 1):  # split
                if _fold(b) and _fold(b) == _fold("".join(ocr_toks[j:k])):
                    out.append(("orthographic", b, " ".join(ocr_toks[j:k]), j0 + j))
                    i, j, grew = i + 1, k, True
                    break
            if grew:
                continue
            for k in range(i + 1, min(i + 4, len(base_toks)) + 1):  # merge
                if _fold(o) and _fold(o) == _fold("".join(base_toks[i:k])):
                    out.append(("orthographic", " ".join(base_toks[i:k]), o, j0 + j))
                    i, j, grew = k, j + 1, True
                    break
            if grew:
                continue
            out.append((classify_region(b, o), b, o, j0 + j))  # single divergent pair
            i, j = i + 1, j + 1
        elif i < len(base_toks):  # base leftover -> OCR omitted it
            b = base_toks[i]
            out.append(("punctuation" if _is_punct(b) else "deletion", b, "", j0 + j))
            i += 1
        else:  # OCR leftover -> not in the edition
            o = ocr_toks[j]
            out.append(("punctuation" if _is_punct(o) else "addition", "", o, j0 + j))
            j += 1
    return out


def diff_page(scholarly_lines: list[str], ocr_lines: list[tuple[int, str]]) -> list[Diff]:
    """Diff a page: ``scholarly_lines`` (ordered text) vs ``ocr_lines`` (``(seg_idx, text)``).

    Returns every classified :class:`Diff`, each tagged with the OCR line it
    belongs to. Page-level so word-wrap resolves; attribution via per-token line
    provenance.
    """
    base_tokens: list[str] = []
    for line in scholarly_lines:
        base_tokens.extend(tokenize(line))

    ocr_tokens: list[str] = []
    ocr_owner: list[int] = []  # ocr_tokens[k] came from segmentation line ocr_owner[k]
    for seg_idx, text in ocr_lines:
        toks = tokenize(text)
        ocr_tokens.extend(toks)
        ocr_owner.extend([seg_idx] * len(toks))

    def owner(jg: int) -> int | None:
        if not ocr_owner:
            return None
        return ocr_owner[min(jg, len(ocr_owner) - 1)]

    sm = difflib.SequenceMatcher(a=base_tokens, b=ocr_tokens, autojunk=False)
    diffs: list[Diff] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for dtype, base_span, ocr_span, jg in _refine_replace(
            base_tokens[i1:i2], ocr_tokens[j1:j2], j1
        ):
            diffs.append(
                Diff(dtype, base_span, ocr_span, owner(jg), _tei(dtype, base_span, ocr_span))
            )
    return diffs
