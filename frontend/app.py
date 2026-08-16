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
    """Tell the browser to revalidate the SPA assets every load.

    The static JS/CSS is edited during development; without this the browser
    serves a stale ``app.js``/``style.css`` after a change (the chips/legend
    silently don't appear). ``no-cache`` forces revalidation (cheap: 304s).
    """
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
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


@app.get("/api/pages/{page_key}")
def api_get_page(page_key: str) -> dict:
    """Full page payload: image dimensions + per-line polygons +
    per-line our-transcription + per-line scholarly-transcription.

    Returned in one shot so the frontend can render tab 1 and tab 2 from
    the same fetch — reduces round-trips vs. one request per line.
    """
    try:
        page = get_repo().get_page(page_key)
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
def api_diffs_json() -> FileResponse:
    """The full classified-discrepancy file (all pages) for download / further analysis.

    This is the same ``line_diff.json`` the viewer renders as chips: keyed
    ``{page -> {seg_line_idx -> [ {type, ocr_text, base_text, tei, group, ocr_line} ]}}``,
    each diff carrying its TEI encoding (see ``src/ocr/line_diff.py``).
    """
    path = get_repo().config.line_diff_json
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no line_diff.json")
    return FileResponse(path, media_type="application/json", filename="AlbucE_line_diff.json")


# Static mount is LAST so ``/api/*`` routes take precedence. The SPA at
# ``static/index.html`` is served for any other path. ``html=True`` makes
# ``/`` return ``index.html`` implicitly.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
