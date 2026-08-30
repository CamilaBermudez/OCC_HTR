"""Whole-manuscript confidence pass — the per-line confidence distribution over EVERY predicted
line (~13.8k), not just the 899 annotated (spec §6.13, human-in-the-loop triage).

For the HITL review queue we rank unseen folio lines by kraken's raw-CTC confidence. This runs the
deployed CTC recogniser over all extracted line crops (no GT needed) and dumps per-line mean/min
peak-frame posterior confidence, so we can see the real distribution the triage queue will sort and
compare it to the annotated 899 (train+val). Streams to CSV so a crash loses nothing.

    PROJECT_ROOT=. uv run python scripts/ocr/manuscript_confidence.py \
        --lines-dir data/processed/extracted_lines/extraction_20260618_154440 \
        --model models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel \
        --out tests/ocr/evaluations/manuscript_confidence
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lines-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("tests/ocr/evaluations/manuscript_confidence"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from kraken.lib import models
    from kraken.lib.dataset import ImageInputTransforms
    from kraken.lib.segmentation import extract_polygons

    from src.ocr.transcribe_line_crops import _synthesised_seg

    net = models.load_any(str(a.model), device=a.device)
    b, c, h, w = net.nn.input
    ts = ImageInputTransforms(b, h, w, c, (16, 0), valid_norm=False)

    crops = sorted(a.lines_dir.rglob("*.png"))
    print(f"{len(crops)} line crops under {a.lines_dir}")
    csv_path = a.out / "manuscript_line_confidence.csv"
    means, mins, nfail = [], [], 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["folio", "stem", "n_chars", "mean_conf", "min_conf"])
        for i, crop in enumerate(crops):
            try:
                im, seg = _synthesised_seg(crop)
                box, _ = next(extract_polygons(im, seg, legacy=net.nn.use_legacy_polygons))
                preds = net.predict(ts(box).unsqueeze(0))[0]  # [(char, s, e, conf)]
                om = np.asarray(net.outputs)
                om = om[0] if om.ndim == 3 else om  # [labels, frames]
                confs = []
                for _ch, s, e, _ in preds:
                    e = max(e, s)
                    peak = s + int(om[:, s : e + 1].max(axis=0).argmax())
                    confs.append(float(om[:, peak].max()))
                if not confs:
                    nfail += 1
                    continue
                mc, mn = float(np.mean(confs)), float(np.min(confs))
                means.append(mc)
                mins.append(mn)
                wr.writerow([crop.parent.name, crop.stem, len(confs), f"{mc:.4f}", f"{mn:.4f}"])
            except Exception:  # noqa: BLE001
                nfail += 1
            if (i + 1) % 1000 == 0:
                f.flush()
                print(f"  {i + 1}/{len(crops)} done ({nfail} failed)", flush=True)

    means, mins = np.array(means), np.array(mins)
    print(f"\nscored {len(means)} lines ({nfail} failed) -> {csv_path}")
    qs = [5, 10, 25, 50, 75, 90, 95]
    print("mean-conf percentiles: " + "  ".join(f"p{q}={np.percentile(means, q):.3f}" for q in qs))
    print(" min-conf percentiles: " + "  ".join(f"p{q}={np.percentile(mins, q):.3f}" for q in qs))
    for thr in (0.90, 0.95, 0.97, 0.99):
        print(
            f"  lines with mean-conf < {thr}: {100 * np.mean(means < thr):.1f}%  "
            f"| min-conf < {thr}: {100 * np.mean(mins < thr):.1f}%"
        )


if __name__ == "__main__":
    main()
