"""Classify OCR-vs-scholarly differences per page, attributed to each OCR line.

Base = scholarly edition; describes what the diplomatic OCR does relative to it
in six categories (abbreviation / orthographic / punctuation / addition /
deletion / substitution) with editorial TEI, via
:func:`src.ocr.line_diff.diff_page` (page-level so word-wrap resolves). Output
feeds the viewer so each OCR line can show its differences (spec §6.7).

Reusable per model — re-run pointing ``--model-dir`` at any transcription.

Output JSON (``--output``):

    {
      "<page_key>": {
        "counts": { "<type>": n, ... },
        "by_line": { "<seg_idx>": [ {type, base_text, ocr_text, ocr_line, tei}, ... ] }
      }, ...
    }

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/diff_transcriptions.py \
        --model-dir data/processed/transcription/finetune_400_full_corpus \
        --scholarly-txt tests/ocr/AlbucE_aligned_20260628_142959.txt \
        --output data/processed/transcription/finetune_400_full_corpus/line_diff.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from src.ocr.line_diff import diff_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("diff_transcriptions")

# Same parsers as scripts/ocr/align_transcriptions.py (inlined — scripts/ is not
# an importable package). Keep in sync if the transcription formats change.
_HEADER_RE = re.compile(r"=+\s*IMAGE:\s*(?P<key>.+?)_full\s*=+")
_LINE_RE = re.compile(r"^(?P<no>\d+):\s?(?P<text>.*)$")
_MODEL_LINE_RE = re.compile(r"_line_(\d+)\.txt$")


def load_scholarly(scholarly_txt: Path) -> dict[str, list[tuple[int, str]]]:
    """``{page_key: [(scholarly_no, text), ...]}`` from the aligned txt."""
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


def load_model_page(model_dir: Path, page_key: str) -> list[tuple[int, str]]:
    """Ordered ``[(seg_idx, text), ...]`` for lines with non-empty text."""
    out: list[tuple[int, str]] = []
    for f in (model_dir / page_key).glob(f"{page_key}_line_*.txt"):
        m = _MODEL_LINE_RE.search(f.name)
        if not m:
            continue
        text = f.read_text(encoding="utf-8").strip()
        if text:
            out.append((int(m.group(1)), text))
    out.sort(key=lambda t: t[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--scholarly-txt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    scholarly = load_scholarly(args.scholarly_txt)
    result: dict[str, dict] = {}
    grand = Counter()
    for page_dir in sorted(p for p in args.model_dir.iterdir() if p.is_dir()):
        page_key = page_dir.name
        model = load_model_page(args.model_dir, page_key)
        if not model or page_key not in scholarly:
            continue
        scholarly_lines = [text for _, text in scholarly[page_key]]
        diffs = diff_page(scholarly_lines, model)

        by_line: dict[str, list[dict]] = {}
        counts = Counter()
        for d in diffs:
            counts[d.type] += 1
            key = str(d.ocr_line)
            by_line.setdefault(key, []).append(d.as_dict())
        grand.update(counts)
        result[page_key] = {"counts": dict(counts), "by_line": by_line}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("Diffed %d pages | category totals: %s", len(result), dict(grand))
    log.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
