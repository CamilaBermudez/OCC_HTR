"""Content-based line alignment between two transcriptions of the same text.

Reusable + transcription-agnostic: give it two ordered lists of text lines
(e.g. the per-segmentation-line **model** OCR of a page and the **scholarly**
edition lines of that page) and it returns a monotonic 1-to-1-with-gaps
alignment based on text similarity — tolerant of the line-break, orthography
and count discrepancies that make naive positional pairing (line i <-> line i)
drift.

Why this is needed (see spec §6.6): the manuscript viewer keys everything by
segmentation-line index and pairs ``scholarly[i]`` with model line ``i``. But
the scholarly edition breaks lines where the editor chose, not where the page
was segmented, so a single split/merge/omitted line shifts every pairing after
it. This module realigns by *what the lines say*, not their ordinal position.

Algorithm: Needleman-Wunsch global alignment (the classic monotonic DP), scoring
a pair by normalized string similarity and charging a flat gap penalty for an
unmatched line on either side. Monotonic = reading order is preserved and pairs
never cross, which is exactly right for two transcriptions of one page.

Design is independent of the earlier scholarly<->manuscript page alignment
(that one is authoritative); this only aligns *within* a page.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

# Medieval orthography folded away for *matching only* (never for display): the
# two transcriptions spell the same word differently (uos/vos, marors/majors,
# long-s, abbreviation tildes, punctuation as separate tokens). Folding these
# lets a true pair score high without changing what the viewer shows.
_COMBINING = dict.fromkeys(range(0x300, 0x370))  # combining diacritics block
_FOLD = str.maketrans({"v": "u", "j": "i", "ſ": "s", "ꝛ": "r"})


def normalize_for_match(text: str) -> str:
    """Aggressively normalize a line for *similarity scoring only*.

    Lowercase; drop combining marks; fold u/v, i/j, long-s, rotunda-r; strip
    everything that is not a latin letter or digit (punctuation, whitespace,
    ⁊, ¶, °, ...). The result is never shown to the user.
    """
    text = unicodedata.normalize("NFKD", text).lower().translate(_FOLD)
    text = text.translate(_COMBINING)
    return re.sub(r"[^a-z0-9]", "", text)


def line_similarity(a: str, b: str) -> float:
    """Similarity of two lines in ``[0, 1]`` (1.0 == identical after folding).

    Empty-vs-empty is 1.0; empty-vs-nonempty is 0.0.
    """
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return fuzz.ratio(na, nb) / 100.0


@dataclass(frozen=True)
class AlignPair:
    """One step of the alignment.

    ``source_idx`` / ``target_idx`` are indices into the input lists, or
    ``None`` for a gap (an unmatched line on that side). ``score`` is the
    similarity of a matched pair (0.0 for gaps).
    """

    source_idx: int | None
    target_idx: int | None
    score: float

    @property
    def is_match(self) -> bool:
        return self.source_idx is not None and self.target_idx is not None


def align_lines(
    source: list[str],
    target: list[str],
    *,
    gap_penalty: float = 0.4,
    min_match_score: float = 0.34,
    sim_fn=line_similarity,
) -> list[AlignPair]:
    """Monotonically align ``source`` lines to ``target`` lines by content.

    Returns the full alignment path as a list of :class:`AlignPair` in reading
    order (matches interleaved with source-only / target-only gaps).

    - ``gap_penalty``: cost of leaving a line unmatched. Higher => the aligner
      prefers to pair lines even when similarity is mediocre; lower => it more
      readily inserts gaps. 0.4 works well for model-vs-scholarly medieval text.
    - ``min_match_score``: a paired step whose similarity is below this is
      *demoted* to two gaps in the returned path (a weak pairing is treated as
      "no correspondence" rather than a wrong highlight). Set to 0 to keep all
      pairs the DP produced.
    - ``sim_fn``: swap in a different ``(a, b) -> [0,1]`` scorer if desired.
    """
    n, m = len(source), len(target)
    # dp[i][j] = best score aligning source[:i] with target[:j]
    NEG = float("-inf")
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    # back[i][j] = 0 diagonal(match) | 1 up(source gap) | 2 left(target gap)
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_penalty
        back[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_penalty
        back[0][j] = 2

    for i in range(1, n + 1):
        si = source[i - 1]
        row, prev = dp[i], dp[i - 1]
        brow = back[i]
        for j in range(1, m + 1):
            diag = prev[j - 1] + sim_fn(si, target[j - 1])
            up = prev[j] - gap_penalty
            left = row[j - 1] - gap_penalty
            best = diag
            b = 0
            if up > best:
                best, b = up, 1
            if left > best:
                best, b = left, 2
            row[j], brow[j] = best, b

    # backtrace
    path: list[AlignPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b == 0:
            s = sim_fn(source[i - 1], target[j - 1])
            if s < min_match_score:
                # demote a weak pair to two gaps (a weak pairing is "no
                # correspondence", better than a confidently-wrong highlight)
                path.append(AlignPair(i - 1, None, 0.0))
                path.append(AlignPair(None, j - 1, 0.0))
            else:
                path.append(AlignPair(i - 1, j - 1, s))
            i, j = i - 1, j - 1
        elif b == 1:
            path.append(AlignPair(i - 1, None, 0.0))
            i -= 1
        else:
            path.append(AlignPair(None, j - 1, 0.0))
            j -= 1
    path.reverse()
    return path


def source_to_target_map(pairs: list[AlignPair]) -> dict[int, int]:
    """Convenience: ``{source_idx: target_idx}`` for matched pairs only."""
    return {p.source_idx: p.target_idx for p in pairs if p.is_match}
