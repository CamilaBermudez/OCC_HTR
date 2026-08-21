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


def _load_line_alignment(path: Path) -> dict[str, dict[int, int]]:
    """Parse ``line_alignment.json`` into ``{page_key: {seg_idx: scholarly_no}}``.

    Built by ``scripts/ocr/align_transcriptions.py`` — maps each model
    (segmentation) line to the *content-matched* scholarly line NUMBER, so the
    viewer highlights the right scholarly line across the two full columns
    (spec §6.6). Both transcriptions are shown in full; this only drives the
    cross-highlight. Missing/unreadable file => ``{}`` (no cross-highlight).
    """
    if not path.is_file():
        logger.info("No line-alignment file at %s — no cross-highlight", path)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read line alignment %s (%s)", path, exc)
        return {}
    out: dict[str, dict[int, int]] = {}
    for page_key, page in raw.items():
        mapping: dict[int, int] = {}
        for pair in page.get("pairs", []):
            mi, no = pair.get("model_idx"), pair.get("scholarly_no")
            if mi is not None and no is not None:
                mapping[int(mi)] = int(no)
        if mapping:
            out[page_key] = mapping
    return out


def _load_line_diff(path: Path) -> dict[str, dict[int, list[dict]]]:
    """Parse ``line_diff.json`` into ``{page_key: {seg_idx: [diff dicts]}}``.

    Built by ``scripts/ocr/diff_transcriptions.py`` (spec §6.7). Missing /
    unreadable file returns ``{}`` (no chips shown).
    """
    if not path.is_file():
        logger.info("No line-diff file at %s — no per-line diff chips", path)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read line diff %s (%s) — skipping", path, exc)
        return {}
    out: dict[str, dict[int, list[dict]]] = {}
    for page_key, page in raw.items():
        by_line = {int(k): v for k, v in page.get("by_line", {}).items()}
        if by_line:
            out[page_key] = by_line
    return out


# Transcription models the viewer can switch between (spec §6.5.28). ``dir`` is a folder
# name under data/processed/transcription/ (per-page <page>/<page>_line_<n>.txt + an optional
# line_diff.json). Stats are 300-val (spec §6.5.25/§6.13). Only entries whose dir exists on
# disk are offered — so ``catmus`` appears once its full corpus finishes generating. Order =
# dropdown order; the first is the default.
_MODEL_REGISTRY: list[dict] = [
    {
        "key": "kraken_leader",
        "label": "kraken 0.9743 (CTC + char-LM)",
        "dir": "krakenLM_full_corpus",
        "arch": "CTC (VGSL CRNN) + char n-gram LM rescore",
        "size": "4.08M params / 16 MB",
        "cer": 0.0256,
        "char_acc": 0.9743,
        "wer": 0.1629,
        "word_acc": 0.8371,
        "desc": "Best overall pipeline: fine-tuned CATMuS + per-position char-LM rescoring.",
    },
    {
        "key": "trocr_leader",
        "label": "TrOCR 0.9549 (ViT+RoBERTa)",
        "dir": "mixedmed4k_full_corpus",
        "arch": "ViT encoder + RoBERTa decoder (seq2seq)",
        "size": "282.6M params / 1130 MB",
        "cer": 0.0451,
        "char_acc": 0.9549,
        "wer": 0.2280,
        "word_acc": 0.7720,
        "desc": "Best seq2seq.",
        #: 600 real + 3000 anno + 4000 medical (gentle), stretch + BPE-150
    },
    {
        "key": "medusa",
        "label": "Medusa 0.9510 (9B VLM)",
        "dir": "medusa_full_corpus_l4_20260713_095002_clean",
        "arch": "9B vision-language model (autoregressive)",
        "size": "9B params / ~18 GB BF16",
        "cer": 0.0490,
        "char_acc": 0.9510,
        "wer": 0.3106,
        "word_acc": 0.6894,
        "desc": "Off-the-shelf multilingual medieval VLM (ENC-PSL) (Moins et al., 2026).",
    },
    {
        "key": "catmus",
        "label": "catmus 0.9603 (CTC baseline)",
        "dir": "catmus_full_corpus",
        "arch": "CTC (CATMuS-medieval, frozen)",
        "size": "4.08M params / 16 MB",
        "cer": 0.0397,
        "char_acc": 0.9603,
        "wer": 0.1488,
        "word_acc": 0.8512,
        "desc": "Frozen off-the-shelf CATMuS-medieval — strong zero-fine-tune baseline "
        "(Pinche et al., 2024).",
    },
]


@dataclass(frozen=True)
class PageMeta:
    """Everything about one page that the frontend needs at render time."""

    page_key: str
    raw_image_path: Path
    image_width: int
    image_height: int
    lines: list[dict]  # model, per segmentation line: {idx, polygon, our_text, diffs}
    scholarly_lines: list[dict]  # full scholarly edition: {no, text} in order
    align: dict[str, int]  # {segmentation_idx (str): scholarly_no} for cross-highlight


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
        # {page_key: {segmentation_line_idx: scholarly line NO}} from the
        # content-based aligner (spec §6.6) — drives the cross-highlight only.
        self._alignment: dict[str, dict[int, int]] = {}
        # Per-MODEL line diffs: {model_key: {page_key: {seg_idx: [diff dicts]}}}.
        self._diffs_by_model: dict[str, dict[str, dict[int, list[dict]]]] = {}
        # Available transcription models (registry entries whose dir exists on disk).
        self._models: list[dict] = []
        self._default_model: str = ""
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

        # 6. Transcription model registry — every source the viewer can switch
        #    between (spec §6.5.28). Only models whose transcription dir exists are
        #    offered; each carries its own line_diff.json (per-model diff chips).
        tx_base = self.config.model_transcription_dir.parent
        for m in _MODEL_REGISTRY:
            mdir = tx_base / m["dir"]
            if not mdir.is_dir():
                continue
            entry = {**m, "dir": mdir}
            self._models.append(entry)
            self._diffs_by_model[m["key"]] = _load_line_diff(mdir / "line_diff.json")
        if not self._models:
            # Fallback to the single configured dir so the viewer still works.
            self._models = [
                {"key": "model", "label": "model", "dir": self.config.model_transcription_dir}
            ]
            self._diffs_by_model["model"] = _load_line_diff(self.config.line_diff_json)
        self._default_model = self._models[0]["key"]

        logger.info(
            "ManuscriptRepo loaded: %d usable pages, %d scholarly, %d aligned; models=%s",
            len(self._page_keys),
            sum(1 for k in self._page_keys if k in self._scholarly),
            sum(1 for k in self._page_keys if k in self._alignment),
            [m["key"] for m in self._models],
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

    def _resolve_model(self, model_key: str | None) -> dict:
        """Registry entry for ``model_key`` (falls back to the default)."""
        for m in self._models:
            if m["key"] == model_key:
                return m
        return self._models[0]

    def list_models(self) -> list[dict]:
        """Public registry (no filesystem Paths) for the ``/api/models`` dropdown."""
        return [
            {k: v for k, v in m.items() if k != "dir"}
            | {"default": m["key"] == self._default_model}
            for m in self._models
        ]

    def model_diff_path(self, model_key: str | None = None) -> Path:
        """Path to the selected model's line_diff.json (for the per-model download)."""
        return self._resolve_model(model_key)["dir"] / "line_diff.json"

    def get_page(self, page_key: str, model_key: str | None = None) -> PageMeta:
        if page_key not in self._page_keys:
            raise KeyError(f"Unknown page_key: {page_key}")

        model = self._resolve_model(model_key)
        seg = json.loads(
            (self.config.segmentation_dir / f"{page_key}.json").read_text(encoding="utf-8")
        )
        width, height = self._get_image_size(page_key)
        page_diffs = self._diffs_by_model.get(model["key"], {}).get(page_key, {})

        # Model column: one entry per segmentation line (selected model's transcription).
        lines: list[dict] = []
        for idx, seg_line in enumerate(seg.get("lines", [])):
            lines.append(
                {
                    "idx": idx,
                    "polygon": seg_line.get("boundary") or [],
                    "baseline": seg_line.get("baseline") or [],
                    "our_text": _load_our_transcription(model["dir"], page_key, idx),
                    "diffs": page_diffs.get(idx, []),
                }
            )

        # Scholarly column: the FULL scholarly edition for this page, its own
        # numbering (0-based key -> 1-based display no). Never hidden — the
        # alignment below only decides which line highlights across.
        scholarly = self._scholarly.get(page_key, {})
        scholarly_lines = [{"no": k + 1, "text": scholarly[k]} for k in sorted(scholarly)]
        align = {str(seg): no for seg, no in self._alignment.get(page_key, {}).items()}

        return PageMeta(
            page_key=page_key,
            raw_image_path=self._raw_map[page_key],
            image_width=width,
            image_height=height,
            lines=lines,
            scholarly_lines=scholarly_lines,
            align=align,
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
