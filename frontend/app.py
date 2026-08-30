"""FastAPI backend for the AlbucE manuscript viewer.

Two responsibilities:
- JSON API under ``/api/*`` for page listing, page payloads, and image
  bytes (see ``manuscript_data.py`` for what a page payload contains).
- Static-file mount at ``/`` serving the SPA (``index.html`` + JS/CSS).

Launch with ``make frontend`` (see makefile) or directly:

    PROJECT_ROOT=. uv run uvicorn frontend.app:app --reload --port 8000

Environment variables (all optional; see :class:`frontend.config.Config`):
- ``VIEWER_RAW_PAGES``, ``VIEWER_SEGMENTATION``,
  ``VIEWER_MODEL_TRANSCRIPTION``, ``VIEWER_SCHOLARLY_TXT``,
  ``VIEWER_FILTERED_KEPT``.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from frontend.manuscript_data import get_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="AlbucE manuscript viewer")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Tell the browser to revalidate every asset AND data response each load.

    The static JS/CSS is edited during development; without this the browser
    serves a stale ``app.js``/``style.css`` after a change (the chips/legend
    silently don't appear). The ``/api/*`` JSON is *regenerated* too — e.g.
    ``/api/compare/<page>`` gains kraken/TrOCR rows when the line-compare files
    are rebuilt — so a browser holding the old JSON shows blank model rows in the
    Model-compare tab even though the current code + data are correct. ``no-cache``
    forces revalidation on everything (cheap: ETag 304s when unchanged).
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.on_event("startup")
def _warm_repo() -> None:
    """Build the in-memory index at startup so the first request is fast."""
    stats = get_repo().stats()
    logging.info("Manuscript repo warmed: %s", stats)


@app.get("/api/stats")
def api_stats() -> dict:
    """Repo summary — useful for the frontend to show which model/scholarly
    txt is currently loaded, and to diagnose 'why is my page missing'."""
    return get_repo().stats()


@app.get("/api/pages")
def api_list_pages() -> dict:
    """Every ``page_key`` we can render (has both a raw JPG and a segmentation
    JSON). The frontend uses this to populate the page dropdown."""
    return {"pages": get_repo().list_pages()}


@app.get("/api/models")
def api_list_models() -> dict:
    """Transcription models the viewer can switch between (dropdowns in tabs 1/2)."""
    return {"models": get_repo().list_models()}


@app.get("/api/pages/{page_key}")
def api_get_page(page_key: str, model: str | None = None) -> dict:
    """Full page payload for the given ``model`` (default = first registry entry):
    image dimensions + per-line polygons + that model's per-line transcription + diffs
    + per-line scholarly transcription. One shot so tabs 1 and 2 render from one fetch.
    """
    try:
        page = get_repo().get_page(page_key, model_key=model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    # asdict handles the ``PageMeta`` dataclass but the Path field needs
    # to be stringified for JSON serialisation.
    payload = asdict(page)
    payload["raw_image_path"] = str(page.raw_image_path)
    return payload


@app.get("/api/pages/{page_key}/image")
def api_get_page_image(page_key: str) -> FileResponse:
    """Serve the raw JPG for a page. Cached by the browser via the JPG's
    own ETag (FileResponse sets ``Last-Modified``)."""
    try:
        path = get_repo().get_image_path(page_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/transcribe")
def api_transcribe(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "catmus",
) -> dict:
    """Upload a page image → segment + reading-order + line-by-line recognise.

    Sync endpoint (FastAPI runs it in a threadpool) since kraken is blocking and
    takes ~30-60 s/page. Returns image size + per-line polygon + predicted text.
    """
    import shutil
    import tempfile

    from src.ocr.page_pipeline import transcribe_page

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        shutil.copyfileobj(file.file, tf)
        tmp = Path(tf.name)
    try:
        return transcribe_page(tmp, model=model, image_name=file.filename)
    except Exception as exc:  # surface pipeline errors to the client
        logging.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


# ---- Tab 4: model-comparison + confidence (spec §7.4.1) ----
@app.get("/api/compare/pages")
def api_compare_pages() -> dict:
    """Pages that have a precomputed line-comparison JSON."""
    d = get_repo().config.line_compare_dir
    pages = sorted(p.stem for p in d.glob("[0-9]*.json")) if d.is_dir() else []
    return {"pages": pages}


@app.get("/api/compare/{page_key}")
def api_compare_page(page_key: str) -> FileResponse:
    """The per-line comparison JSON for one page (scholarly/catmus/ViT + conf + mismatch)."""
    if "/" in page_key or ".." in page_key:
        raise HTTPException(status_code=400, detail="bad page key")
    path = get_repo().config.line_compare_dir / f"{page_key}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no comparison for page")
    return FileResponse(path, media_type="application/json")


@app.get("/api/compare/{page_key}/image/{stem}")
def api_compare_image(page_key: str, stem: str) -> FileResponse:
    """The kept line-crop image for one physical line (the carousel photo)."""
    if any(bad in (page_key + stem) for bad in ("/", "..", "\\")):
        raise HTTPException(status_code=400, detail="bad path")
    path = get_repo().config.filtered_kept_dir / page_key / f"{stem}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no crop")
    return FileResponse(path, media_type="image/png")


@app.get("/api/diffs.json")
def api_diffs_json(model: str | None = None) -> FileResponse:
    """The full classified-discrepancy file for the SELECTED ``model`` (all pages),
    for download / further analysis — the same ``line_diff.json`` the viewer renders as
    chips: ``{page -> {seg_line_idx -> [ {type, ocr_text, base_text, tei, group} ]}}``,
    each diff carrying its TEI encoding (see ``src/ocr/line_diff.py``).
    """
    path = get_repo().model_diff_path(model)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no line_diff.json for this model")
    key = model or "model"
    return FileResponse(
        path, media_type="application/json", filename=f"AlbucE_line_diff_{key}.json"
    )


# ---- Tab 6: human-in-the-loop review & correct (spec §6.13) ----
@app.get("/api/review/stats")
def api_review_stats() -> dict:
    """How many lines are in the review queue and how many corrected so far."""
    from frontend import review

    cfg = get_repo().config
    q = review.get_queue(cfg)
    done = review.load_done(review.corrections_path())
    return {"total": len(q), "done": len(done), "pending": len(q) - len(done)}


@app.get("/api/review/queue")
def api_review_queue(limit: int = 300, skip_done: int = 1) -> dict:
    """The confidence-ranked review queue (worst-first): the next ``limit`` pending lines
    (or all, with ``skip_done=0``), each with the model transcriptions + min-conf + disagreement."""
    from frontend import review

    return review.queue_payload(get_repo().config, limit=limit, skip_done=bool(skip_done))


@app.post("/api/review/save")
def api_review_save(
    line_id: Annotated[str, Form()],
    corrected_text: Annotated[str, Form()] = "",
    confidence: Annotated[str, Form()] = "certain",
) -> dict:
    """Append one human correction (+ the annotator's self-rated confidence) and return done-count."""
    from frontend import review

    cfg = get_repo().config
    rec = review.append_correction(
        review.corrections_path(), line_id, corrected_text, cfg, confidence=confidence
    )
    done = review.load_done(review.corrections_path())
    return {"ok": True, "record": rec, "done": len(done)}


# Static mount is LAST so ``/api/*`` routes take precedence. The SPA at
# ``static/index.html`` is served for any other path. ``html=True`` makes
# ``/`` return ``index.html`` implicitly.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
