"""Shared kraken/CTC + char-LM rescoring primitives (spec §6.13).

Kept in ``src`` (the importable package) so both the rescore driver and the
lambda-tuning driver can reuse them. See ``scripts/ocr/kraken_lm_rescore.py`` and
``scripts/ocr/kraken_lm_tune.py``.
"""

from __future__ import annotations

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
