"""In-memory index of the manuscript for the viewer backend.

Responsibilities:
- Discover every page across four sources (raw JPGs, segmentation JSONs,
  per-line model transcriptions, per-page scholarly transcription).
- Normalise raw JPG filenames (``5 - garde - 001.jpg``) into the same
  ``page_key`` format used by segmentation / transcription
  (``05_garde_001``).
- Serve a single ``get_page(page_key)`` payload that the frontend can
  render directly: image dimensions + per-line polygon + our transcription
  + scholarly transcription.

Line indexing convention:
- Segmentation JSON's ``lines[i]`` is 0-based.
- Per-line model files use ``<page>_line_<i>.txt`` with the same 0-based
  index.
- The scholarly aligned txt uses 1-based line indices (``1:`` is the
  first line); we subtract 1 on parse so everything the API returns is
  0-based.

Missing data is expected and handled gracefully:
- Some segmentation lines have no transcription (line filter dropped
  the crop): ``our_text = None`` in the payload.
- Some pages have no scholarly transcription block: every line's
  ``scholarly_text`` is ``None``.
- Some pages have no raw JPG: those pages are skipped entirely and
  won't appear in ``list_pages()``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from frontend.config import Config

logger = logging.getLogger("manuscript_data")


_SCHOLARLY_HEADER_RE = re.compile(
    r"^==========\s*IMAGE:\s*(?P<page>\S+?)(?:_full)?\s*==========\s*$"
)
_SCHOLARLY_LINE_RE = re.compile(r"^(?P<idx>\d+)\s*:\s*(?P<text>.*)$")


def _normalise_raw_filename(name: str) -> str | None:
    """Translate a raw JPG filename into the shared ``page_key``.

    Raw filenames use human-readable spacing/punctuation
    (``5 - garde - 001.jpg``, ``10 - f. 005v - 006.jpg``); the
    segmentation JSONs and per-line transcription folders use a
    machine-normal form (``05_garde_001``, ``10_f_005v_006``) with a
    zero-padded 2-digit leading number.

    Returns ``None`` for unparseable filenames — they get skipped
    entirely rather than half-loaded.
    """
    stem = Path(name).stem
    parts = [p.strip() for p in stem.split(" - ")]
    if len(parts) < 2:
        return None
    # First token is the sequential page number; zero-pad to 2 digits so
    # sort order matches segmentation/transcription folder names.
    try:
        parts[0] = f"{int(parts[0]):02d}"
    except ValueError:
        return None
    # Inside a token, dots + spaces become underscores
    # ("f. 005v" -> "f_005v"), and remaining spaces become underscores
    # too (defensive for future filenames we haven't seen).
    parts = [re.sub(r"[.\s]+", "_", p) for p in parts]
    return "_".join(parts)


def _parse_scholarly(txt_path: Path) -> dict[str, dict[int, str]]:
    """Parse the aligned scholarly txt into ``{page_key: {line_idx: text}}``.

    Line indices in the file are 1-based; we store them 0-based so the
    frontend can look them up with the same index it uses for polygons
    and per-line txts.
    """
    if not txt_path.is_file():
        logger.warning(
            "Scholarly txt not found at %s — tab 2 will show empty middle column", txt_path
        )
        return {}

    per_page: dict[str, dict[int, str]] = {}
    current_page: str | None = None
    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        header_match = _SCHOLARLY_HEADER_RE.match(raw_line)
        if header_match:
            current_page = header_match.group("page")
            per_page.setdefault(current_page, {})
            continue
        if current_page is None:
            continue
        line_match = _SCHOLARLY_LINE_RE.match(raw_line)
        if line_match:
            idx_1based = int(line_match.group("idx"))
            text = line_match.group("text")
            per_page[current_page][idx_1based - 1] = text
    logger.info(
        "Scholarly txn: %d pages, %d total lines",
        len(per_page),
        sum(len(v) for v in per_page.values()),
    )
    return per_page


def _load_our_transcription(model_dir: Path, page_key: str, line_idx: int) -> str | None:
    """Return the per-line prediction text or None if absent."""
    p = model_dir / page_key / f"{page_key}_line_{line_idx}.txt"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8").strip()
    return text or None


def _load_line_alignment(path: Path) -> dict[str, dict[int, str]]:
    """Parse ``line_alignment.json`` into ``{page_key: {seg_idx: scholarly_text}}``.

    Built by ``scripts/ocr/align_transcriptions.py`` — pairs each model
    (segmentation) line with the *content-matched* scholarly line, so the viewer
    highlights the right one instead of the positional guess (spec §6.6). A
    missing/unreadable file is not an error: returns ``{}`` and the caller falls
    back to positional pairing.
    """
    if not path.is_file():
        logger.info("No line-alignment file at %s — using positional scholarly pairing", path)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read line alignment %s (%s) — positional fallback", path, exc)
        return {}
    out: dict[str, dict[int, str]] = {}
    for page_key, page in raw.items():
        mapping: dict[int, str] = {}
        for pair in page.get("pairs", []):
            mi, text = pair.get("model_idx"), pair.get("scholarly_text")
            if mi is not None and text is not None:
                mapping[int(mi)] = text
        if mapping:
            out[page_key] = mapping
    return out


@dataclass(frozen=True)
class PageMeta:
    """Everything about one page that the frontend needs at render time."""

    page_key: str
    raw_image_path: Path
    image_width: int
    image_height: int
    lines: list[dict]  # each: {idx, polygon:[[x,y],...], our_text, scholarly_text}


class ManuscriptRepo:
    """In-memory index built once at server start.

    Cost is a few seconds — 71 JSON reads + 1 txt parse + a PIL open for
    each page image (used only to grab natural width/height for the SVG
    viewBox). Data itself is not held in RAM per-line until requested
    via ``get_page``; transcription txt reads are cached with LRU.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._page_keys: list[str] = []
        self._raw_map: dict[str, Path] = {}
        self._image_size: dict[str, tuple[int, int]] = {}
        self._scholarly: dict[str, dict[int, str]] = {}
        # {page_key: {segmentation_line_idx: aligned scholarly text}} from the
        # content-based aligner (spec §6.6). Empty => positional fallback.
        self._alignment: dict[str, dict[int, str]] = {}
        self._segmentation_pages: set[str] = set()
        self._load()

    def _load(self) -> None:
        # 1. Map raw JPGs -> page_key
        for jpg in sorted(self.config.raw_pages_dir.glob("*.jpg")):
            key = _normalise_raw_filename(jpg.name)
            if key is None:
                logger.debug("Skipping raw file (unparseable): %s", jpg.name)
                continue
            self._raw_map[key] = jpg

        # 2. Which pages have segmentation JSONs?
        for j in self.config.segmentation_dir.glob("*.json"):
            self._segmentation_pages.add(j.stem)

        # 3. A page is "usable" only if it has both a raw image AND a
        #    segmentation JSON. Pages missing either are skipped so the
        #    dropdown can't offer broken options.
        self._page_keys = sorted(self._raw_map.keys() & self._segmentation_pages)
        skipped = self._raw_map.keys() ^ self._segmentation_pages
        if skipped:
            logger.warning(
                "%d page keys skipped (raw or segmentation missing): %s",
                len(skipped),
                sorted(skipped)[:10],
            )

        # 4. Scholarly transcription (optional).
        self._scholarly = _parse_scholarly(self.config.scholarly_txt)

        # 5. Content-based line alignment (optional; positional fallback if absent).
        self._alignment = _load_line_alignment(self.config.line_alignment_json)

        logger.info(
            "ManuscriptRepo loaded: %d usable pages, %d with scholarly txn, %d with line alignment",
            len(self._page_keys),
            sum(1 for k in self._page_keys if k in self._scholarly),
            sum(1 for k in self._page_keys if k in self._alignment),
        )

    def list_pages(self) -> list[str]:
        return self._page_keys

    def get_image_path(self, page_key: str) -> Path:
        try:
            return self._raw_map[page_key]
        except KeyError as exc:
            raise KeyError(f"Unknown page_key: {page_key}") from exc

    def _get_image_size(self, page_key: str) -> tuple[int, int]:
        """Cached (width, height) of the raw JPG for viewBox scaling."""
        if page_key not in self._image_size:
            with Image.open(self._raw_map[page_key]) as im:
                self._image_size[page_key] = (im.width, im.height)
        return self._image_size[page_key]

    def get_page(self, page_key: str) -> PageMeta:
        if page_key not in self._page_keys:
            raise KeyError(f"Unknown page_key: {page_key}")

        seg = json.loads(
            (self.config.segmentation_dir / f"{page_key}.json").read_text(encoding="utf-8")
        )
        width, height = self._get_image_size(page_key)
        scholarly_lines = self._scholarly.get(page_key, {})
        # Content-based alignment maps each segmentation line to its true
        # scholarly counterpart (spec §6.6). Fall back to positional pairing
        # per-page only when this page has no alignment entry.
        aligned = self._alignment.get(page_key)

        lines: list[dict] = []
        for idx, seg_line in enumerate(seg.get("lines", [])):
            if aligned is not None:
                scholarly_text = aligned.get(idx)
            else:
                scholarly_text = scholarly_lines.get(idx)
            lines.append(
                {
                    "idx": idx,
                    "polygon": seg_line.get("boundary") or [],
                    "baseline": seg_line.get("baseline") or [],
                    "our_text": _load_our_transcription(
                        self.config.model_transcription_dir, page_key, idx
                    ),
                    "scholarly_text": scholarly_text,
                }
            )

        return PageMeta(
            page_key=page_key,
            raw_image_path=self._raw_map[page_key],
            image_width=width,
            image_height=height,
            lines=lines,
        )

    def stats(self) -> dict:
        """Startup summary for the ``/api/stats`` endpoint."""
        n_scholarly = sum(1 for k in self._page_keys if k in self._scholarly)
        return {
            "n_pages": len(self._page_keys),
            "n_pages_with_scholarly": n_scholarly,
            "raw_pages_dir": str(self.config.raw_pages_dir),
            "segmentation_dir": str(self.config.segmentation_dir),
            "model_transcription_dir": str(self.config.model_transcription_dir),
            "scholarly_txt": str(self.config.scholarly_txt),
        }


@lru_cache(maxsize=1)
def get_repo() -> ManuscriptRepo:
    """Module-level singleton so FastAPI dependency injection doesn't
    rebuild the index on every request."""
    return ManuscriptRepo(Config.from_env())
