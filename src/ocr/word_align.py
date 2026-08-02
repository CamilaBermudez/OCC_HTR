"""Anchored banded word-level Needleman–Wunsch diff (spec §6.7.3, 2026-08-02).

The free char-level ``diff_page`` (§6.7 #2) mis-associates distant text on hard
stretches (~25 % of substantive substitutions are such "loose" garbage). This
module replaces it with a **word-level** alignment that is:

* **fuzzy** — cells are scored by folded word similarity (rapidfuzz), so a
  misread word (``superftuitatz`` vs ``superfluitatz``) is a cheap *match*, not a
  delete+insert;
* **merge/split aware** — the DP allows 1↔1, 2↔1 and 1↔2 word steps, so the
  scribe's ``delo`` ↔ editor's ``de lo`` (and the reverse) align in one step;
* **anchored + banded** — the line-alignment matches (``{seg_idx: scholarly_no}``)
  are diagonal anchors; the DP is restricted to a band around the interpolated
  diagonal, so distant-but-similar words (repeated ``de``/``que``) can't
  mis-match (kills the scramble) while words still flow **across line
  boundaries** inside the band (kills the edge-spill that sank the per-line
  approach, §6.7 #3).

Output is the same ``Diff`` list as ``line_diff.diff_page``, so it drops into the
existing classifier + ``split_diffs`` + viewer unchanged.
"""

from __future__ import annotations

from src.ocr.line_diff import Diff, _despace, _fold, _is_punct, _tei, classify_region, tokenize

try:
    from rapidfuzz.distance import Levenshtein

    def _word_sim(a: str, b: str) -> float:
        pa, pb = _is_punct(a), _is_punct(b)
        if pa and pb:  # two punctuation tokens
            return 1.0 if a == b else 0.4
        if pa != pb:  # a word vs a punctuation mark never align -> force a gap
            return -1.5
        fa, fb = _fold(a), _fold(b)
        if not fa or not fb:
            return 0.0
        return 1.0 - Levenshtein.distance(fa, fb) / max(len(fa), len(fb))
except ImportError:  # pragma: no cover - rapidfuzz is a hard dep in practice

    def _word_sim(a: str, b: str) -> float:
        return 1.0 if _fold(a) == _fold(b) else 0.0


_GAP = -0.55  # cost of leaving a word unaligned (tuned: < a weak 1-1 match)
_MERGE_EPS = 0.02  # tiny bias so a clean 1-1 wins over an equal-scoring merge


def _emit(base: str, ocr: str, owner: int | None, out: list[Diff]) -> None:
    base, ocr = base.strip(), ocr.strip()
    if base == ocr:  # identical (or both empty) -> a match, not a difference
        return
    # Everything that actually differs is emitted; the group (is_editorial) +
    # the viewer decide what to show. Word-boundary `spacing` (Esi/E si) and
    # abbreviations are kept; only punctuation/orthographic are hidden by default.
    dtype = classify_region(base, ocr)
    out.append(Diff(dtype, base, ocr, owner, _tei(dtype, base, ocr)))


def _expected_diagonal(n: int, anchors: list[tuple[int, int]]) -> list[int]:
    """For each scholarly-token index i in [0, n], the expected OCR index (piecewise
    linear through the sorted, monotonic ``anchors`` = (sch_idx, ocr_idx))."""
    exp = [0] * (n + 1)
    a = 0
    for i in range(n + 1):
        while a + 1 < len(anchors) and anchors[a + 1][0] <= i:
            a += 1
        si, oj = anchors[a]
        if a + 1 < len(anchors):
            si2, oj2 = anchors[a + 1]
            frac = (i - si) / (si2 - si) if si2 > si else 0.0
            exp[i] = round(oj + frac * (oj2 - oj))
        else:
            exp[i] = oj
    return exp


def _align(
    sch: list[str],
    ocr: list[str],
    owners: list[int | None],
    anchors: list[tuple[int, int]],
    band: int,
) -> list[Diff]:
    n, m = len(sch), len(ocr)
    exp = _expected_diagonal(n, anchors)
    neg = float("-inf")

    def owner_at(pos: int) -> int | None:
        # nearest OCR line for a span (esp. a deletion, which consumes no OCR
        # token) — clamped so we never emit a None owner, matching diff_page.
        return owners[min(max(pos, 0), m - 1)] if m else None

    def in_band(i: int, j: int) -> bool:
        return abs(j - exp[i]) <= band

    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if not in_band(i, j) and not (i == n and j == m):
                continue
            best, op = neg, None
            if i > 0 and dp[i - 1][j] > neg:  # delete scholarly word
                cand = dp[i - 1][j] + _GAP
                if cand > best:
                    best, op = cand, ("del", i - 1, j)
            if j > 0 and dp[i][j - 1] > neg:  # insert OCR word
                cand = dp[i][j - 1] + _GAP
                if cand > best:
                    best, op = cand, ("ins", i, j - 1)
            if i > 0 and j > 0 and dp[i - 1][j - 1] > neg:  # 1-1 sub/match
                cand = dp[i - 1][j - 1] + _word_sim(sch[i - 1], ocr[j - 1])
                if cand > best:
                    best, op = cand, ("sub", i - 1, j - 1)
            # merge/split never swallow a punctuation token — a mark must align
            # on its own (else `agudas .`->`agudas` is folded to a false match).
            if (
                i > 1
                and j > 0
                and dp[i - 2][j - 1] > neg
                and not _is_punct(sch[i - 2])
                and not _is_punct(sch[i - 1])
            ):  # 2 sch -> 1 ocr
                cand = (
                    dp[i - 2][j - 1] + _word_sim(sch[i - 2] + sch[i - 1], ocr[j - 1]) - _MERGE_EPS
                )
                if cand > best:
                    best, op = cand, ("merge", i - 2, j - 1)
            if (
                i > 0
                and j > 1
                and dp[i - 1][j - 2] > neg
                and not _is_punct(ocr[j - 2])
                and not _is_punct(ocr[j - 1])
            ):  # 1 sch -> 2 ocr
                cand = (
                    dp[i - 1][j - 2]
                    + _word_sim(sch[i - 1], ocr[j - 2] + " " + ocr[j - 1])
                    - _MERGE_EPS
                )
                if cand > best:
                    best, op = cand, ("split", i - 1, j - 2)
            dp[i][j], bt[i][j] = best, op

    # backtrace
    out: list[Diff] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bt[i][j]
        if op is None:  # fell out of band at an edge — drain remaining as gaps
            if i > 0:
                _emit(sch[i - 1], "", owner_at(j - 1), out)
                i -= 1
            else:
                _emit("", ocr[j - 1], owner_at(j - 1), out)
                j -= 1
            continue
        kind, pi, pj = op
        if kind == "del":
            _emit(sch[pi], "", owner_at(pj - 1), out)
        elif kind == "ins":
            _emit("", ocr[pj], owner_at(pj), out)
        elif kind == "sub":
            _emit(sch[pi], ocr[pj], owner_at(pj), out)
        elif kind == "merge":
            _emit(sch[pi] + " " + sch[pi + 1], ocr[pj], owner_at(pj), out)
        elif kind == "split":
            b, o = sch[pi], ocr[pj] + " " + ocr[pj + 1]
            # a word split ONLY by manuscript line-wrap (the two OCR tokens sit on
            # different model lines and are identical modulo spacing) is not a
            # transcription difference — suppress it. A same-line split (la gremas
            # / lagremas) is a real word-boundary diff and is emitted.
            if owner_at(pj) != owner_at(pj + 1) and _despace(b) == _despace(o):
                pass
            else:
                _emit(b, o, owner_at(pj), out)
        i, j = pi, pj
    out.reverse()
    return out


def diff_page_banded(
    scholarly_lines: list[str],
    ocr_lines: list[tuple[int, str]],
    align: dict[int, int],
    band: int = 6,
) -> list[Diff]:
    """Anchored banded word-NW diff of a page. ``align`` = ``{seg_idx: scholarly_no}``.

    Falls back to a single wide band when there are no anchors (an unaligned page).
    """
    # OCR token stream (reading order) + per-token owner seg_idx + line-start index.
    ocr_tokens: list[str] = []
    owners: list[int | None] = []
    ocr_line_start: dict[int, int] = {}
    for seg_idx, text in ocr_lines:
        ocr_line_start[seg_idx] = len(ocr_tokens)
        for tok in tokenize(text):
            ocr_tokens.append(tok)
            owners.append(seg_idx)

    # Scholarly token stream (by scholarly_no) + per-line-start index.
    sch_tokens: list[str] = []
    sch_line_start: dict[int, int] = {}
    for no, text in enumerate(scholarly_lines, start=1):
        sch_line_start[no] = len(sch_tokens)
        sch_tokens.extend(tokenize(text))

    # Anchors from the line matches, in scholarly-token order, made monotonic.
    raw = sorted(
        (sch_line_start[no], ocr_line_start[seg])
        for seg, no in align.items()
        if no in sch_line_start and seg in ocr_line_start
    )
    anchors: list[tuple[int, int]] = [(0, 0)]
    last_o = 0
    for si, oj in raw:
        if si > anchors[-1][0] and oj >= last_o:
            anchors.append((si, oj))
            last_o = oj
    anchors.append((len(sch_tokens), len(ocr_tokens)))

    # No real anchors -> widen the band so the DP can still find the diagonal.
    eff_band = band if len(anchors) > 2 else max(band, len(ocr_tokens))
    return _align(sch_tokens, ocr_tokens, owners, anchors, eff_band)
