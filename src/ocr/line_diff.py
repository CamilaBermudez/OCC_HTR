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
    # scribal contraction: a SHORT OCR function word is a subsequence of the
    # scholarly form (del=de lo, dels=de los, al=a lo). Capped at 4 letters —
    # without a brevigraph mark, a *content* word that merely drops letters
    # (inscio<-inscisio) is a misread, not an abbreviation, so it must fall
    # through to `substitution` and stay visible.
    if (
        fo
        and fb
        and fo != fb
        and len(fo) <= 4
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
    if diff_type in ("orthographic", "spacing"):
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


def _diff_core(base: str, ocr: str, owner: list[int]) -> list[Diff]:
    """Character-level diff of one ``base`` string vs one ``ocr`` string.

    ``owner[k]`` = segmentation-line index of ``ocr[k]`` (for per-line attribution).
    Non-equal char ranges are grown to whole words (only if they touch a word
    char), merged within a word, a pure spacing-shift absorbed, then classified;
    spacing + orthographic are suppressed.
    """

    def owner_at(pos: int) -> int | None:
        if not owner:
            return None
        return owner[min(pos, len(owner) - 1)]

    sm = difflib.SequenceMatcher(a=base, b=ocr, autojunk=False)
    regions: list[list[int]] = []  # [base_start, base_end, ocr_start, ocr_end]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        bs, be = _expand(base, i1, i2)
        os_, oe = _expand(ocr, j1, j2)
        regions.append([bs, be, os_, oe])
    regions.sort()
    merged: list[list[int]] = []
    for r in regions:
        if merged and r[0] < merged[-1][1] and r[2] < merged[-1][3]:
            merged[-1][1] = max(merged[-1][1], r[1])
            merged[-1][3] = max(merged[-1][3], r[3])
        else:
            merged.append(r)
    # absorb an adjacent region only when the combined span is a pure spacing
    # shift ("un apostema" vs "una postema") — never merges into a visible diff
    absorbed: list[list[int]] = []
    for r in merged:
        if absorbed:
            cb = base[absorbed[-1][0] : r[1]]
            co = ocr[absorbed[-1][2] : r[3]]
            if len(cb) <= 40 and _despace(cb) == _despace(co):
                absorbed[-1][1], absorbed[-1][3] = r[1], r[3]
                continue
        absorbed.append(list(r))

    diffs: list[Diff] = []
    for bs, be, os_, oe in absorbed:
        b, o = base[bs:be].strip(), ocr[os_:oe].strip()
        if not b and not o:
            continue
        dtype = classify_region(b, o)
        if dtype in ("spacing", "orthographic"):
            # spacing = wrap/segmentation; orthographic (u/v, i/j) suppressed for
            # now (user decision 2026-07-31) — re-enable by dropping it here.
            continue
        diffs.append(Diff(dtype, b, o, owner_at(os_), _tei(dtype, b, o)))
    return diffs


def _concat_ocr(ocr_lines: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """Join OCR lines with single spaces; return (text, per-char owner seg idx)."""
    parts: list[str] = []
    owner: list[int] = []
    for seg_idx, text in ocr_lines:
        if parts:
            parts.append(" ")
            owner.append(seg_idx)
        parts.append(text)
        owner.extend([seg_idx] * len(text))
    return "".join(parts), owner


def diff_page(scholarly_lines: list[str], ocr_lines: list[tuple[int, str]]) -> list[Diff]:
    """Free page-level char diff (both sides concatenated). Reusable, alignment-free."""
    ocr, owner = _concat_ocr(ocr_lines)
    return _diff_core(" ".join(scholarly_lines), ocr, owner)


# ---------------------------------------------------------------------------
# Substantive / editorial split + scramble guard (spec §6.7.2, 2026-08-02).
# Non-destructive layer over diff_page: separates genuine OCR differences from
# editorial normalization and drops alignment-scramble artifacts, so a caller
# (assessment or viewer) can show a clean substantive-error view.
# ---------------------------------------------------------------------------

# Bare Occitan articles that appear as a false add/del when the scribe joins
# ``de``/``a`` + article (``delo``, ``als``) and the editor spaces them.
_ARTICLE_FOLDS = {"lo", "la", "los", "las", "lu", "le", "l", "els", "al", "als"}

# A single real edit is never this long; an add/del span above it is the
# free page-level diff mis-associating distant text (a scramble), not an edit.
_SCRAMBLE_LEN = 50


def is_editorial(d: Diff) -> bool:
    """True if the difference is editorial normalization, not a transcription diff.

    Editorial (hidden by default, shown via the viewer toggle) = punctuation the
    editor adds, and pure **orthographic** variation (u/v, i/j, long-s). A bare
    article added/dropped (``de``/``a`` + article) is also editorial spacing.
    Everything else — **substitution, addition, deletion, abbreviation (marked
    brevigraphs AND `del`=`de lo` contractions), and word-boundary `spacing`
    (`Esi`/`E si`, `la gremas`/`lagremas`)** — is a real transcription
    difference and stays visible.
    """
    if d.type in ("punctuation", "orthographic"):
        return True
    if d.type in ("addition", "deletion"):
        span = d.ocr_text if d.type == "addition" else d.base_text
        return _fold(span) in _ARTICLE_FOLDS
    return False


def diff_group(d: Diff) -> str:
    """Which shown-group a single diff belongs to: substantive / editorial / scramble."""
    span_len = max(len(d.base_text), len(d.ocr_text))
    if d.type in ("addition", "deletion") and span_len > _SCRAMBLE_LEN:
        return "scramble"
    return "editorial" if is_editorial(d) else "substantive"


def split_diffs(diffs: list[Diff]) -> tuple[list[Diff], list[Diff], list[Diff]]:
    """Partition diffs into (substantive, editorial, scramble).

    ``substantive`` = genuine OCR differences (misread / dropped / added word);
    ``editorial`` = normalization (punctuation, expansion, article spacing);
    ``scramble`` = over-long add/del spans that signal an alignment failure on
    that stretch, not a real edit — surfaced so the caller can flag the region
    rather than show fabricated diffs.
    """
    groups: dict[str, list[Diff]] = {"substantive": [], "editorial": [], "scramble": []}
    for d in diffs:
        groups[diff_group(d)].append(d)
    return groups["substantive"], groups["editorial"], groups["scramble"]


def diff_aligned(
    scholarly_lines: list[tuple[int, str]],
    ocr_lines: list[tuple[int, str]],
    align: dict[int, int],
) -> list[Diff]:
    """Alignment-constrained diff: each scholarly line vs the OCR line(s) aligned to it.

    ``scholarly_lines`` = ``(scholarly_no, text)``; ``ocr_lines`` =
    ``(seg_idx, text)``; ``align`` = ``{seg_idx: scholarly_no}`` (from
    ``line_alignment.json``). Diffing per aligned group — not free over the whole
    page — avoids the global mis-alignment on 2-column pages, and makes an
    unmatched (e.g. merged) scholarly line show up as a clean **deletion**
    against its one aligned OCR line ("all this text is empty in the OCR"). OCR
    lines with no alignment are skipped (already flagged in the viewer).
    """
    groups: dict[int, list[tuple[int, str]]] = {}
    for seg_idx, text in ocr_lines:
        no = align.get(seg_idx)
        if no is not None:
            groups.setdefault(no, []).append((seg_idx, text))
    diffs: list[Diff] = []
    for no, text in scholarly_lines:
        group = groups.get(no)
        if not group:
            continue  # scholarly line with no aligned OCR line -> nothing to attach
        ocr, owner = _concat_ocr(group)
        diffs.extend(_diff_core(text, ocr, owner))
    return diffs
