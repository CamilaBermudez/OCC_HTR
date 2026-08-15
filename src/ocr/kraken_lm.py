"""Shared kraken/CTC + char-LM rescoring primitives (spec §6.13).

Kept in ``src`` (the importable package) so both the rescore driver and the
lambda-tuning driver can reuse them. See ``scripts/ocr/kraken_lm_rescore.py`` and
``scripts/ocr/kraken_lm_tune.py``.
"""

from __future__ import annotations

import math

import numpy as np

from src.ocr.char_lm import CharNGramLM


def label_to_char(codec, n_labels: int) -> list[str]:
    """label index -> character ('' marks the CTC blank / no glyph)."""
    out = []
    for i in range(n_labels):
        try:
            dec = codec.decode([(i, 0, 0, 1.0)])
            out.append(dec[0][0] if dec else "")
        except Exception:  # noqa: BLE001
            out.append("")
    return out


def line_candidates(net, ts, l2c, im, seg, topk: int):
    """Per predicted char, the top-k (char, logprob) at its peak output frame."""
    from kraken.lib.segmentation import extract_polygons

    box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
    preds = net.predict(ts(box).unsqueeze(0))[0]  # [(char, start, end, conf)]
    om = np.asarray(net.outputs)
    om = om[0] if om.ndim == 3 else om  # [labels, frames]
    cands = []
    for _ch, s, e, _ in preds:
        e = max(e, s)
        span = om[:, s : e + 1]
        peak = s + int(span.max(axis=0).argmax())
        col = om[:, peak]
        idxs = col.argsort()[::-1][:topk]
        opts = [(l2c[int(i)], float(np.log(max(col[int(i)], 1e-9)))) for i in idxs if l2c[int(i)]]
        if opts:
            cands.append(opts)
    return cands


_NEG_INF = float("-inf")


def _logsumexp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _accum(store: dict, pfx: str, pb: float = _NEG_INF, pnb: float = _NEG_INF) -> None:
    """Log-add (pb, pnb) into store[pfx]'s (logPblank, logPnonblank)."""
    ob, onb = store.get(pfx, (_NEG_INF, _NEG_INF))
    store[pfx] = (_logsumexp(ob, pb), _logsumexp(onb, pnb))


def ctc_beam_search(
    logp, l2c, blank_set, lm: CharNGramLM | None, alpha: float, beam_width: int, prune: int = 16
) -> str:
    """CTC prefix beam search with char-LM shallow fusion (spec §6.13 B5).

    Unlike the per-position rescorer (`rescore`), this decodes over the raw per-FRAME
    posteriors, so a path can emit a char where greedy read blank (insertion) or blank
    where greedy read a char (deletion) — it reaches all three error types. LM is applied
    (weight ``alpha``) each time a NEW character is appended.

    logp: [T, V] log-probs. blank_set: set of label indices that are the CTC blank / no
    glyph. Returns the best decoded string.
    """
    beams: dict[str, tuple[float, float]] = {
        "": (0.0, _NEG_INF)
    }  # prefix -> (logPblank, logPnonblank)
    T = len(logp)
    for t in range(T):
        row = logp[t]
        cand = sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:prune]
        nxt: dict[str, tuple[float, float]] = {}

        for prefix, (pb, pnb) in beams.items():
            last = prefix[-1] if prefix else None
            for c in cand:
                p = row[c]
                if c in blank_set:
                    _accum(nxt, prefix, pb=_logsumexp(pb + p, pnb + p))
                    continue
                ch = l2c[c]
                if not ch:
                    continue
                if ch == last:
                    _accum(nxt, prefix, pnb=pnb + p)  # repeat: stays same string
                    newp = prefix + ch  # new char only via a preceding blank
                    lm_s = alpha * lm.logcond(prefix, ch) if (lm and alpha) else 0.0
                    _accum(nxt, newp, pnb=pb + p + lm_s)
                else:
                    newp = prefix + ch
                    lm_s = alpha * lm.logcond(prefix, ch) if (lm and alpha) else 0.0
                    _accum(nxt, newp, pnb=_logsumexp(pb, pnb) + p + lm_s)
        beams = dict(
            sorted(nxt.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)[:beam_width]
        )
    best = max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))
    return best[0]


def rescore(cands, lm: CharNGramLM | None, lam: float, beam: int, topk: int) -> str:
    """Left-to-right beam search: score = visual_logprob + lam * LM_logcond(char|prefix)."""
    beams = [("", 0.0)]
    for pos in cands:
        nxt = []
        for text, sc in beams:
            for ch, vlp in pos[:topk]:
                lm_s = lam * lm.logcond(text, ch) if (lm and lam) else 0.0
                nxt.append((text + ch, sc + vlp + lm_s))
        nxt.sort(key=lambda x: -x[1])
        beams = nxt[:beam]
    return beams[0][0] if beams else ""
