"""catmus (kraken/CTC) transcription WITH per-character confidence over line crops.

Mirror of ``vit_transcribe_conf.py`` for the CTC side. For each ``<page>/<stem>.png``
crop, run ``kraken.rpred`` (the official path — correct line normalisation) and record
the text plus each character's confidence (``record.confidences``, aligned 1:1 with
``record.prediction``). Writes one JSON per page:

    {"page": "<page>", "lines": {"<stem>": {"text": "...",
                                             "chars": [["y", 0.998], ["s", 1.0], ...]}}}

Feeds the viewer's model-comparison tab (spec §7.4.1). Run over the SAME kept
filtered crops as ViT so the two models compare on identical inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kraken import rpred
from kraken.lib import models

sys.path.insert(0, "src")
from ocr.transcribe_line_crops import _synthesised_seg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir", required=True, type=Path, help="kept-crops root (per-page subdirs)"
    )
    ap.add_argument("--model", type=Path, default=Path("models/ocr/catmus-medieval.mlmodel"))
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit-pages", type=int, default=0)
    args = ap.parse_args()

    net = models.load_any(str(args.model), device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "_provenance.json").write_text(
        json.dumps({"model": str(args.model), "input_dir": str(args.input_dir)}), encoding="utf-8"
    )

    pages = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    if args.limit_pages:
        pages = pages[: args.limit_pages]

    t0 = time.time()
    n_lines = 0
    for pg in pages:
        out_json = args.out_dir / f"{pg.name}.json"
        if out_json.exists():  # resumable
            n_lines += len(json.loads(out_json.read_text()).get("lines", {}))
            continue
        lines: dict[str, dict] = {}
        for crop in sorted(pg.glob("*.png")):
            try:
                im, seg = _synthesised_seg(crop)
                recs = list(rpred.rpred(net, im, seg))
            except Exception:  # noqa: BLE001
                lines[crop.stem] = {"text": "", "chars": []}
                continue
            rec = recs[0] if recs else None
            text = (rec.prediction if rec else "") or ""
            confs = list(rec.confidences) if rec else []
            chars = [
                [text[i], round(float(confs[i]), 4)] for i in range(min(len(text), len(confs)))
            ]
            lines[crop.stem] = {"text": text, "chars": chars}
        out_json.write_text(
            json.dumps({"page": pg.name, "lines": lines}, ensure_ascii=False), encoding="utf-8"
        )
        n_lines += len(lines)
        print(
            f"{pg.name}: {len(lines)} lines  (total {n_lines}, {time.time() - t0:.0f}s)", flush=True
        )

    print(f"DONE {n_lines} lines across {len(pages)} pages in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
