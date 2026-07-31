"""Align a per-line model transcription to the scholarly edition, per page.

Independent of the earlier scholarly<->manuscript page alignment (that one is
authoritative). This aligns *within* each page, by content, so the viewer can
highlight the right scholarly line when a model line is selected — fixing the
positional drift documented in spec §6.6.

Reusable across transcriptions: point ``--model-dir`` at any
``<page>/<page>_line_<N>.txt`` tree (e.g. a newer model from the 34-model grid)
and re-run; the alignment core is :func:`src.ocr.line_alignment.align_lines`.

Output JSON (``--output``):

    {
      "<page_key>": {
        "pairs": [ {model_idx, scholarly_no, score, model_text, scholarly_text}, ... ],
        "model_to_scholarly": { "<seg_idx>": <scholarly_no>, ... }  # matches only
      },
      ...
    }

``model_idx`` is the 0-based segmentation-line index (the file's ``_line_<N>``);
``scholarly_no`` is the scholarly edition's own line number (as printed in the
aligned txt). ``null`` on either side marks an unmatched line (a gap).

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/align_transcriptions.py \
        --model-dir data/processed/transcription/finetune_400_full_corpus \
        --scholarly-txt tests/ocr/AlbucE_aligned_20260628_142959.txt \
        --output data/processed/transcription/finetune_400_full_corpus/line_alignment.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from src.ocr.line_alignment import align_lines

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("align_transcriptions")

_HEADER_RE = re.compile(r"=+\s*IMAGE:\s*(?P<key>.+?)_full\s*=+")
_LINE_RE = re.compile(r"^(?P<no>\d+):\s?(?P<text>.*)$")
_MODEL_LINE_RE = re.compile(r"_line_(\d+)\.txt$")


def load_model_page(model_dir: Path, page_key: str) -> list[tuple[int, str]]:
    """Ordered ``[(seg_idx, text), ...]`` for lines that have non-empty text."""
    page = model_dir / page_key
    out: list[tuple[int, str]] = []
    for f in page.glob(f"{page_key}_line_*.txt"):
        m = _MODEL_LINE_RE.search(f.name)
        if not m:
            continue
        text = f.read_text(encoding="utf-8").strip()
        if text:
            out.append((int(m.group(1)), text))
    out.sort(key=lambda t: t[0])
    return out


def load_scholarly(scholarly_txt: Path) -> dict[str, list[tuple[int, str]]]:
    """Parse the aligned txt into ``{page_key: [(scholarly_no, text), ...]}``."""
    pages: dict[str, list[tuple[int, str]]] = {}
    cur: str | None = None
    for raw in scholarly_txt.read_text(encoding="utf-8").splitlines():
        h = _HEADER_RE.match(raw)
        if h:
            cur = h.group("key")
            pages.setdefault(cur, [])
            continue
        m = _LINE_RE.match(raw)
        if cur is not None and m:
            pages[cur].append((int(m.group("no")), m.group("text").rstrip()))
    return pages


def align_page(
    model: list[tuple[int, str]],
    scholarly: list[tuple[int, str]],
    **kw,
) -> dict:
    """Align one page; return the serialisable ``pairs`` + ``model_to_scholarly``."""
    pairs = align_lines([t for _, t in model], [t for _, t in scholarly], **kw)
    out_pairs = []
    m2s: dict[str, int] = {}
    for p in pairs:
        mi = model[p.source_idx][0] if p.source_idx is not None else None
        si = scholarly[p.target_idx][0] if p.target_idx is not None else None
        out_pairs.append(
            {
                "model_idx": mi,
                "scholarly_no": si,
                "score": round(p.score, 4),
                "model_text": model[p.source_idx][1] if p.source_idx is not None else None,
                "scholarly_text": scholarly[p.target_idx][1] if p.target_idx is not None else None,
            }
        )
        if p.is_match:
            m2s[str(mi)] = si
    return {"pairs": out_pairs, "model_to_scholarly": m2s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--scholarly-txt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gap-penalty", type=float, default=0.4)
    ap.add_argument("--min-match-score", type=float, default=0.34)
    args = ap.parse_args()

    scholarly = load_scholarly(args.scholarly_txt)
    result: dict[str, dict] = {}
    n_pages = n_matched = n_drift = 0
    for page_dir in sorted(p for p in args.model_dir.iterdir() if p.is_dir()):
        page_key = page_dir.name
        model = load_model_page(args.model_dir, page_key)
        if not model or page_key not in scholarly:
            continue
        page = align_page(
            model,
            scholarly[page_key],
            gap_penalty=args.gap_penalty,
            min_match_score=args.min_match_score,
        )
        result[page_key] = page
        n_pages += 1
        matched = page["model_to_scholarly"]
        n_matched += len(matched)
        # a page "drifts" if any matched model_idx != scholarly_no (positional
        # pairing would have highlighted the wrong line there)
        if any(int(k) != v for k, v in matched.items()):
            n_drift += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(
        "Aligned %d pages | %d matched line-pairs | %d pages where positional pairing drifts",
        n_pages,
        n_matched,
        n_drift,
    )
    log.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
