"""End-to-end single-page transcription: segment -> reading-order -> recognise.

Wires the pipeline the thesis uses (spec §6) into one call for the viewer's
"upload a page" tab: kraken baseline **segmentation** (`kraken segment -bl`), our
2-column **reading-order** reorder, then line-by-line **recognition** with a
kraken/CATMuS ``.mlmodel`` via ``kraken.rpred``. Default recogniser = the CATMuS
baseline; other models (kraken-ft, TrOCR) can be added later behind ``model``.

``transcribe_page`` returns everything the frontend needs to draw the result:
image size, and per line (in reading order) the boundary polygon + predicted text.
"""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image

from src.data_preprocessing.image_segmentation import (
    apply_reading_order_to_json,
    segment_image,
)

CATMUS_MODEL = "models/ocr/catmus-medieval.mlmodel"
# Registry the frontend can grow later; value = path to a kraken .mlmodel.
KRAKEN_MODELS = {
    "catmus": CATMUS_MODEL,
}


@lru_cache(maxsize=4)
def _load_rec_model(model_path: str, device: str = "cpu"):
    from kraken.lib import models  # lazy: pulls torch/scipy

    return models.load_any(model_path, device=device)


def transcribe_page(
    image_path: str | Path,
    model: str = "catmus",
    device: str = "cpu",
) -> dict:
    """Segment + reading-order + recognise one page image.

    Returns ``{width, height, model, n_lines, lines: [{order, polygon,
    baseline, text}]}`` with lines already in reading order.
    """
    from kraken import rpred
    from kraken.containers import BaselineLine, Segmentation

    image_path = Path(image_path)
    model_path = KRAKEN_MODELS.get(model, CATMUS_MODEL)

    # 1. segmentation (kraken baseline) -> JSON of lines (baseline + boundary)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        seg_json = Path(tf.name)
    if not segment_image(image_path, seg_json):
        raise RuntimeError("kraken segmentation failed")

    # 2. reading order (handles 2-column medieval pages)
    try:
        apply_reading_order_to_json(seg_json, image_path)
    except Exception:
        pass  # fall back to raw segmentation order if the reorder can't run

    data = json.loads(seg_json.read_text(encoding="utf-8"))
    seg_json.unlink(missing_ok=True)
    json_lines = data.get("lines", [])

    im = Image.open(image_path)
    width, height = im.size

    # 3. recognise all lines over the full page in one rpred pass
    kraken_lines = [
        BaselineLine(
            id=str(i),
            baseline=[tuple(p) for p in ln["baseline"]],
            boundary=[tuple(p) for p in ln["boundary"]],
        )
        for i, ln in enumerate(json_lines)
    ]
    out_lines: list[dict] = []
    if kraken_lines:
        seg = Segmentation(
            type="baselines",
            imagename=str(image_path),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=kraken_lines,
        )
        rec_model = _load_rec_model(model_path, device)
        preds = list(rpred.rpred(rec_model, im, seg))
        for order, (ln, pred) in enumerate(zip(json_lines, preds, strict=False)):
            out_lines.append(
                {
                    "order": order,
                    "polygon": ln["boundary"],
                    "baseline": ln["baseline"],
                    "text": getattr(pred, "prediction", "") or "",
                }
            )

    return {
        "width": width,
        "height": height,
        "model": model,
        "n_lines": len(out_lines),
        "lines": out_lines,
    }
