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
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image

from src.data_preprocessing.image_segmentation import (
    apply_reading_order_to_json,
    segment_image,
)

CATMUS_MODEL = "models/ocr/catmus-medieval.mlmodel"
# CTC recognisers — a kraken .mlmodel run over the whole page in one rpred pass.
KRAKEN_MODELS = {
    "kraken_leader": "models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel",
    "catmus": CATMUS_MODEL,
}
# seq2seq recognisers — segment with kraken, then run TrOCR per line crop (value = HF
# checkpoint dir with resize_mode.txt + tokenizer). Handled by ``_recognise_trocr``.
TROCR_MODELS = {
    "trocr_leader": "models/vit_lightreal_med4k/trocr_20260823_073535/best_model",
}


@lru_cache(maxsize=4)
def _load_rec_model(model_path: str, device: str = "cpu"):
    from kraken.lib import models  # lazy: pulls torch/scipy

    return models.load_any(model_path, device=device)


def _minimal_alto(image_name: str, width: int, height: int, out_lines: list[dict]) -> str:
    """Text+layout ALTO for the TrOCR path (no per-glyph cuts — seq2seq has none)."""
    import html

    tl = []
    for i, ln in enumerate(out_lines):
        poly = ln["polygon"] or [[0, 0]]
        xs, ys = [int(p[0]) for p in poly], [int(p[1]) for p in poly]
        x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        pts = " ".join(f"{int(p[0])} {int(p[1])}" for p in poly)
        box = f'HPOS="{x0}" VPOS="{y0}" WIDTH="{w}" HEIGHT="{h}"'
        tl.append(
            f'<TextLine ID="line_{i}" {box}><Shape><Polygon POINTS="{pts}"/></Shape>'
            f'<String CONTENT="{html.escape(ln["text"])}" {box}/></TextLine>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"><Description>'
        f"<sourceImageInformation><fileName>{html.escape(image_name)}</fileName>"
        "</sourceImageInformation></Description><Layout>"
        f'<Page WIDTH="{width}" HEIGHT="{height}" PHYSICAL_IMG_NR="1" ID="page_1"><PrintSpace>'
        f'<TextBlock ID="block_1">{"".join(tl)}</TextBlock></PrintSpace></Page></Layout></alto>'
    )


def _recognise_trocr(im, seg, json_lines: list[dict], model_dir: str, device: str) -> list[dict]:
    """Run TrOCR on each segmented line crop; returns out_lines aligned to json_lines."""
    import torch
    from kraken.lib.segmentation import extract_polygons
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    from src.ocr.image_prep import prepare_image

    mdir = Path(model_dir)
    resize = "stretch"
    if (mdir / "resize_mode.txt").is_file():
        resize = (mdir / "resize_mode.txt").read_text(encoding="utf-8").strip() or "stretch"
    tmodel = VisionEncoderDecoderModel.from_pretrained(mdir).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(mdir)
    tok = AutoTokenizer.from_pretrained(mdir)

    crops = [box for box, _ in extract_polygons(im, seg, legacy=False)]
    out_lines: list[dict] = []
    for order, ln in enumerate(json_lines):
        text = ""
        if order < len(crops):
            img = prepare_image(crops[order].convert("RGB"), proc, resize)
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                g = tmodel.generate(pixel_values=pv, num_beams=1, max_length=128)
            text = tok.batch_decode(g, skip_special_tokens=True)[0].strip()
        out_lines.append(
            {"order": order, "polygon": ln["boundary"], "baseline": ln["baseline"], "text": text}
        )
    return out_lines


def transcribe_page(
    image_path: str | Path,
    model: str = "catmus",
    device: str = "cpu",
    image_name: str | None = None,
) -> dict:
    """Segment + reading-order + recognise one page image.

    ``image_name`` overrides the name written into the ALTO ``<fileName>`` (the
    server hands us a temp file, so we pass the user's original upload name).

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
    alto = ""
    if kraken_lines and model in TROCR_MODELS:
        # seq2seq path: kraken segmentation above, TrOCR recognition per line crop.
        seg = Segmentation(
            type="baselines",
            imagename=str(image_path),
            text_direction="horizontal-lr",
            script_detection=False,
            lines=kraken_lines,
        )
        out_lines = _recognise_trocr(im, seg, json_lines, TROCR_MODELS[model], device)
        alto = _minimal_alto(image_name or image_path.name, width, height, out_lines)
    elif kraken_lines:
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
        # ALTO XML complementing the segmentation with the recognised text.
        # Serialise the rpred RECORDS directly — they carry prediction + character
        # cuts, so ALTO <String CONTENT> is populated (a plain BaselineLine.text
        # is not). Standard OCR interchange format: layout + text in one file.
        try:
            from kraken import serialization

            alto = serialization.serialize(
                Segmentation(
                    type="baselines",
                    imagename=image_name or image_path.name,
                    text_direction="horizontal-lr",
                    script_detection=False,
                    lines=preds,
                ),
                image_size=(width, height),
                template="alto",
                sub_line_segmentation=False,  # word-level <String>, no per-glyph clutter
            )
        except Exception:
            logging.exception("ALTO serialization failed")

    return {
        "width": width,
        "height": height,
        "model": model,
        "n_lines": len(out_lines),
        "lines": out_lines,
        "alto": alto,
    }
