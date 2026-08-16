"""Transcribe the full filtered-line corpus with a kraken model, per-page, mirroring
the catmus `finetune_400_full_corpus` layout so the viewer/eval tooling reads it the
same way: for each page dir of pre-cut line crops, write
``<out>/<run>/<page>/<line>.txt`` + a reading-order ``<out>/<run>/<page>_full.txt``.

The stock `run_transcribe_line_crops.py` is flat-only; this loads the model ONCE and
walks the per-page subdirs of the filtered-kept tree.

    PROJECT_ROOT=. uv run python scripts/ocr/kraken_full_corpus.py \
        --input-dir data/processed/filtered_images/20260618_160948/original/kept \
        --model-path models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --run-name krakenbest_full_corpus
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from kraken import rpred
from kraken.lib import models

from src.ocr.transcribe_line_crops import _synthesised_seg

_LINE_RE = re.compile(r"_line_(\d+)\.png$")


def _line_no(p: Path) -> int:
    m = _LINE_RE.search(p.name)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir", type=Path, required=True, help="parent of per-page line-crop dirs"
    )
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/transcription"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    save = a.output_dir / a.run_name
    save.mkdir(parents=True, exist_ok=True)
    model = models.load_any(str(a.model_path), device=a.device)
    pages = sorted(d for d in a.input_dir.iterdir() if d.is_dir())
    print(f"pages={len(pages)} -> {save}", flush=True)

    t0 = time.time()
    n_lines = n_empty = n_failed = 0
    for pd in pages:
        crops = sorted(pd.glob("*.png"), key=_line_no)
        if not crops:
            continue
        pdir = save / pd.name
        pdir.mkdir(exist_ok=True)
        full: list[str] = []
        for c in crops:
            try:
                im, seg = _synthesised_seg(c)
                preds = list(rpred.rpred(model, im, seg))
                text = (getattr(preds[0], "prediction", "") or "") if preds else ""
            except Exception:  # noqa: BLE001
                text = ""
                n_failed += 1
            (pdir / f"{c.stem}.txt").write_text(text + "\n", encoding="utf-8")
            full.append(text)
            n_lines += 1
            if not text:
                n_empty += 1
        (save / f"{pd.name}_full.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
        print(f"  {pd.name}: {len(crops)} lines ({n_lines} total)", flush=True)

    print(
        f"DONE {n_lines} lines ({n_empty} empty, {n_failed} failed) in "
        f"{time.time() - t0:.0f}s -> {save}",
        flush=True,
    )


if __name__ == "__main__":
    main()
