"""Character n-gram language model (stupid-backoff) for OCR rescoring.

A deliberately small, dependency-free LM for the reranking plan (spec §6.13). It
estimates P(char | preceding chars) from character n-gram counts and is used to add a
**language prior** on top of a recogniser's visual scores — turning "best pixel match"
into "best pixel match that is also plausible Occitan".

Why n-gram / stupid-backoff and not a neural LM:
  * The clean *diplomatic* corpus is tiny (~600 annotated lines, §6.13) — n-gram with
    backoff degrades gracefully where a neural model would overfit.
  * Stupid backoff (Brants et al. 2007) needs no normalisation and is excellent for
    *ranking* hypotheses, which is all a rescorer does.
  * Zero heavy deps (no KenLM/C++ toolchain) so we can measure whether the LM helps at
    all before investing in the full KenLM + pyctcdecode CTC-lattice pipeline.

Diplomatic style matters: the recogniser output + 300-val GT keep abbreviation marks
(⁊ ¶ ꝑ tildes); train this LM ONLY on diplomatic text (annotated GT / catmus output),
never on the normalized DOM dictionary or the normalized scholarly edition — that
mismatch is exactly what sank the blind lexicon swap (§6.10).
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path

_BOS = "\x02"  # sentence-start pad
_ALPHA = 0.4  # stupid-backoff discount


class CharNGramLM:
    """Character n-gram LM with stupid backoff. Scores are log-probabilities.

    Not a normalised distribution (stupid backoff isn't), but monotone in
    plausibility, which is all a rescorer needs. Use :meth:`logscore` to rank
    candidate strings; higher = more Occitan-plausible.
    """

    def __init__(self, order: int = 6) -> None:
        self.order = order
        # counts[k] maps a k-length char string -> frequency (k = 1..order)
        self.counts: list[dict[str, int]] = [defaultdict(int) for _ in range(order + 1)]
        self._total = 0
        self._vocab: set[str] = set()

    def train(self, lines: list[str]) -> CharNGramLM:
        for line in lines:
            s = _BOS * (self.order - 1) + line
            self._vocab.update(line)
            for i in range(self.order - 1, len(s)):
                self._total += 1
                for k in range(1, self.order + 1):
                    self.counts[k][s[i - k + 1 : i + 1]] += 1
        return self

    def _cond(self, ctx: str, ch: str) -> float:
        """Stupid-backoff P(ch | ctx): count(ctx+ch)/count(ctx), else 0.4·backoff."""
        for k in range(len(ctx), 0, -1):
            c = ctx[len(ctx) - k :]
            num = self.counts[k + 1].get(c + ch, 0)
            den = self.counts[k].get(c, 0)
            if den > 0 and num > 0:
                return (_ALPHA ** (len(ctx) - k)) * num / den
        # unigram floor with add-1 smoothing over the observed alphabet
        v = max(1, len(self._vocab))
        return (_ALPHA ** len(ctx)) * (self.counts[1].get(ch, 0) + 1) / (self._total + v)

    def logscore(self, text: str, *, per_char: bool = False) -> float:
        """Sum of log P(char | preceding order-1 chars). per_char=True length-normalises
        (use for comparing hypotheses of different lengths)."""
        s = _BOS * (self.order - 1) + text
        total = 0.0
        n = 0
        for i in range(self.order - 1, len(s)):
            ctx = s[i - (self.order - 1) : i]
            total += math.log(self._cond(ctx, s[i]))
            n += 1
        return total / n if (per_char and n) else total

    def logcond(self, text: str, ch: str) -> float:
        """log P(ch | the last order-1 chars of text). Incremental scoring for a
        left-to-right beam search — cheaper than re-scoring the whole prefix."""
        s = _BOS * (self.order - 1) + text
        ctx = s[len(s) - (self.order - 1) :]
        return math.log(self._cond(ctx, ch))

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(
            pickle.dumps(
                {
                    "order": self.order,
                    "counts": [dict(c) for c in self.counts],
                    "total": self._total,
                    "vocab": self._vocab,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> CharNGramLM:
        d = pickle.loads(Path(path).read_bytes())
        lm = cls(order=d["order"])
        lm.counts = [defaultdict(int, c) for c in d["counts"]]
        lm._total = d["total"]
        lm._vocab = d["vocab"]
        return lm


def train_from_files(paths: list[str | Path], order: int = 6) -> CharNGramLM:
    lines: list[str] = []
    for p in paths:
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                lines.append(ln)
    return CharNGramLM(order=order).train(lines)
