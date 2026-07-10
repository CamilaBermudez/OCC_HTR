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

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from frontend.manuscript_data import get_repo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="AlbucE manuscript viewer")


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


# Static mount is LAST so ``/api/*`` routes take precedence. The SPA at
# ``static/index.html`` is served for any other path. ``html=True`` makes
# ``/`` return ``index.html`` implicitly.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
