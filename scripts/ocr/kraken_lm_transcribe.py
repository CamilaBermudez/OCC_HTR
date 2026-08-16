"""Transcribe line crops with the FULL kraken pipeline = CTC + per-position char-LM
rescoring (spec §6.13, P1): kraken's top-k candidates at each greedy position are
re-ranked by a char n-gram LM, score = visual_logprob + lam*LM_logcond. This is the
deployed kraken pipeline (300-val 0.9743 / 0.8367) vs raw kraken (0.9710 / 0.8201).

Input may be a flat folder of crops (e.g. the 300-val) or a parent of per-page subdirs
(the filtered corpus); output mirrors the transcribe layout — per-line ``<stem>.txt``,
plus a reading-order ``<page>_full.txt`` for each page when nested.

    PROJECT_ROOT=. uv run python scripts/ocr/kraken_lm_transcribe.py \
        --input-dir data/processed/annotated_samples/OCR/validation \
        --model-path models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --lm-dir data/processed/annotated_samples/OCR/full_annotated \
        --run-name krakenLM_val300 --lam 0.2
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from kraken.lib import models
from kraken.lib.dataset import ImageInputTransforms

from src.ocr.char_lm import CharNGramLM
from src.ocr.kraken_lm import label_to_char, line_candidates, rescore
from src.ocr.transcribe_line_crops import _synthesised_seg

_LINE_RE = re.compile(r"_line_(\d+)\.png$")


def _line_no(p: Path) -> int:
    m = _LINE_RE.search(p.name)
    return int(m.group(1)) if m else 0


def _train_lm(lm_dir: Path, order: int = 6) -> CharNGramLM:
    txts = []
    for c in sorted(lm_dir.glob("*.png")):
        g = c.with_name(c.stem + ".gt.txt")
        if g.is_file():
            t = g.read_text(encoding="utf-8").strip()
            if t:
                txts.append(t)
    if not txts:
        raise SystemExit(f"No .gt.txt found in {lm_dir} to train the char-LM")
    return CharNGramLM(order=order).train(txts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument(
        "--lm-dir", type=Path, required=True, help="real crops w/ .gt.txt for the char-LM"
    )
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/transcription"))
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    save = a.output_dir / a.run_name
    save.mkdir(parents=True, exist_ok=True)
    net = models.load_any(str(a.model_path), device=a.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)
    lm = _train_lm(a.lm_dir)

    # Build the label->char map: one warm predict populates net.outputs so we can read
    # the label count (dummy l2c on the warm-up call — its candidates are discarded).
    warm = next(a.input_dir.rglob("*.png"))
    _ = line_candidates(net, ts, [""] * 512, *_synthesised_seg(warm), a.topk)
    om0 = np.asarray(net.outputs)
    l2c = label_to_char(net.codec, int(om0.shape[1] if om0.ndim == 3 else om0.shape[0]))

    def predict(crop: Path) -> str:
        try:
            cands = line_candidates(net, ts, l2c, *_synthesised_seg(crop), a.topk)
            return rescore(cands, lm, a.lam, a.beam, a.topk)
        except Exception:  # noqa: BLE001
            return ""

    flat = sorted(a.input_dir.glob("*.png"))
    t0 = time.time()
    if flat:  # flat folder of crops (e.g. the 300-val)
        for cr in flat:
            (save / f"{cr.stem}.txt").write_text(predict(cr) + "\n", encoding="utf-8")
        print(f"DONE {len(flat)} lines in {time.time() - t0:.0f}s -> {save}", flush=True)
        return

    pages = sorted(d for d in a.input_dir.iterdir() if d.is_dir())  # nested per-page
    n = 0
    for pd in pages:
        crops = sorted(pd.glob("*.png"), key=_line_no)
        if not crops:
            continue
        pdir = save / pd.name
        pdir.mkdir(exist_ok=True)
        full = []
        for cr in crops:
            text = predict(cr)
            (pdir / f"{cr.stem}.txt").write_text(text + "\n", encoding="utf-8")
            full.append(text)
            n += 1
        (save / f"{pd.name}_full.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
        print(f"  {pd.name}: {len(crops)} lines ({n} total)", flush=True)
    print(f"DONE {n} lines in {time.time() - t0:.0f}s -> {save}", flush=True)


if __name__ == "__main__":
    main()
