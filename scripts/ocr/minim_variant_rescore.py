"""Minim-variant rescoring: a targeted char-LM pass over the ambiguous strokes.

Motivation (spec §6.8 / §6.8.1): the recognizers' #1 error is minim confusion —
the vertical-stroke letters i/n/u/m are read as the wrong partition of the same
strokes (``cum``→``cuin``, ``m``→``ni``, ``n``→``u`` …). N-best rescoring can only
reorder hypotheses the recognizer emitted; here we instead *generate* the minim
variants explicitly and let the FULL char-LM pick the most plausible reading.

For each maximal run of minim letters in a word, we compute its total stroke
count (i=1, n=2, u=2, m=3), enumerate every way to repartition those strokes
into i/n/u/m, and greedily keep the variant that maximises the full-line char-LM
score. The LM is the SAME full-corpus model as the reranker (§6.13) — we only
change the *candidate set*, not the LM's training (see the note in spec on why
restricting the LM's training to minim words would hurt).

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/minim_variant_rescore.py \
        --pred-dir data/processed/transcription/kraken_600real_8020_val300 \
        --gt-dir data/processed/annotated_samples/OCR/validation \
        --lm-train-dir data/processed/annotated_samples/OCR/full_annotated \
        --out-dir tests/ocr/evaluations/minim_variant_rescore
"""

from __future__ import annotations

import argparse
import re
from itertools import product
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from src.ocr.char_lm import CharNGramLM

# Minim-stroke counts. i=1, n=2, u=2, m=3 (diplomatic text: v->u, j->i already).
_STROKES = {"i": 1, "n": 2, "u": 2, "m": 3}
_PART_LETTERS = {1: ("i",), 2: ("n", "u"), 3: ("m",)}
_RUN = re.compile(r"[inum]+")
_MAX_STROKES = 8  # cap enumeration on long runs (rare); leave longer runs as-is


def _compositions(k: int):
    """Yield tuples of parts from {1,2,3} summing to k."""
    if k == 0:
        yield ()
        return
    for p in (1, 2, 3):
        if p <= k:
            for rest in _compositions(k - p):
                yield (p, *rest)


def run_variants(run: str) -> list[str]:
    """All minim readings of a run with the same total stroke count."""
    k = sum(_STROKES[c] for c in run)
    if not (2 <= k <= _MAX_STROKES):
        return [run]
    seen: set[str] = set()
    out: list[str] = []
    for comp in _compositions(k):
        for combo in product(*[_PART_LETTERS[p] for p in comp]):
            v = "".join(combo)
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def rescore_line(line: str, lm: CharNGramLM, margin: float = 0.0) -> str:
    """Greedily repartition each minim run to the full-line-LM-best reading.

    ``margin``: only override the recogniser's original run when the LM prefers
    the variant by more than ``margin`` log-prob (the recogniser's reading is the
    acoustic argmax; a text-only LM must *strongly* disagree to overrule it).
    """
    result = line
    pos = 0
    while True:
        m = _RUN.search(result, pos)
        if not m:
            break
        orig = m.group()
        variants = run_variants(orig)
        if len(variants) <= 1:
            pos = m.end()
            continue
        orig_s = lm.logscore(result)
        best_v, best_gain = orig, 0.0
        for v in variants:
            if v == orig:
                continue
            gain = lm.logscore(result[: m.start()] + v + result[m.end() :]) - orig_s
            if gain > best_gain:
                best_gain, best_v = gain, v
        if best_gain > margin:  # override only when the LM strongly disagrees
            result = result[: m.start()] + best_v + result[m.end() :]
            pos = m.start() + len(best_v)
        else:
            pos = m.end()
    return result


def _texts(folder: Path, suffix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(folder.glob(f"*{suffix}")):
        stem = f.name[: -len(suffix)]
        out[stem] = f.read_text(encoding="utf-8").strip()
    return out


def _corpus_cer(pred: dict[str, str], gt: dict[str, str]) -> tuple[float, float, int]:
    ce = cn = we = wn = 0
    for stem, g in gt.items():
        p = pred.get(stem, "")
        ce += Levenshtein.distance(p, g)
        cn += len(g)
        we += Levenshtein.distance(p.split(), g.split())
        wn += len(g.split())
    return ce / cn, we / wn, cn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True, help="CTC 1-best <stem>.txt dir")
    ap.add_argument("--gt-dir", type=Path, required=True, help="<stem>.gt.txt (+ png) dir")
    ap.add_argument("--lm-train-dir", type=Path, required=True, help="600 <stem>.gt.txt for the LM")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--margins", type=float, nargs="+", default=[0, 2, 4, 6, 8, 12, 20])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    lm = CharNGramLM(order=args.order).train(list(_texts(args.lm_train_dir, ".gt.txt").values()))
    preds = _texts(args.pred_dir, ".txt")
    gt = _texts(args.gt_dir, ".gt.txt")

    b_cer, b_wer, n = _corpus_cer(preds, gt)
    print(f"lines={len(gt)}  ref_chars={n}")
    print(f"{'margin':>7}{'changed':>9}{'char-acc':>10}{'CER':>9}{'WER':>9}{'Δchar':>8}")
    print(f"{'CTC':>7}{'-':>9}{1 - b_cer:>10.4f}{b_cer:>9.4f}{b_wer:>9.4f}{'—':>8}")
    best = None
    for margin in args.margins:
        rescored = {s: rescore_line(t, lm, margin) for s, t in preds.items()}
        changed = sum(1 for s in preds if rescored[s] != preds[s])
        r_cer, r_wer, _ = _corpus_cer(rescored, gt)
        d = 100 * (b_cer - r_cer)
        print(f"{margin:>7.1f}{changed:>9}{1 - r_cer:>10.4f}{r_cer:>9.4f}{r_wer:>9.4f}{d:>+7.2f}pp")
        if best is None or r_cer < best[1]:
            best = (margin, r_cer, rescored)
    print(
        "\nNOTE: margins swept on the 300-val itself — illustrative, NOT honestly tuned "
        "(a real result tunes margin on a dev split, like the α/λ rerankers)."
    )

    if args.out_dir and best:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for s, t in best[2].items():
            (args.out_dir / f"{s}.txt").write_text(t, encoding="utf-8")
        print(f"wrote best-margin ({best[0]}) rescored predictions -> {args.out_dir}")


if __name__ == "__main__":
    main()
