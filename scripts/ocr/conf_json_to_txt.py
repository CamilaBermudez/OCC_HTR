"""Derive the per-page txt transcription layout from a confidence-JSON dump.

``vit_transcribe_conf.py`` writes one ``<page>.json`` per page with the text +
per-token confidence of every line. The viewer's tabs 1-2 (and the align/diff
scripts) instead read the standard layout ``<page>/<page>_line_<N>.txt`` +
``<page>_full.txt``. This converts the former into the latter so a single
transcription pass serves both consumers (no second decode).

Usage:
    PROJECT_ROOT=. uv run python scripts/ocr/conf_json_to_txt.py \
        --conf-dir data/processed/transcription/vitlightreal_conf_fullms \
        --out-dir  data/processed/transcription/vitlightreal_full_corpus
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Line index at the end of a stem. Manually inserted half-lines use a "p"
# fraction (e.g. `_line_153p5` = 153.5, between 153 and 154) — same naming as
# the kept crops and every earlier full-corpus dir.
_LINE_NO = re.compile(r"_line_(\d+)(?:p(\d+))?$")


def _line_sort_key(stem: str) -> float:
    m = _LINE_NO.search(stem)
    if m is None:
        raise ValueError(f"stem without _line_<N> suffix: {stem!r}")
    return float(f"{m.group(1)}.{m.group(2) or 0}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conf-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    n_pages = n_lines = 0
    for pj in sorted(args.conf_dir.glob("[0-9]*.json")):
        data = json.loads(pj.read_text(encoding="utf-8"))
        page = data["page"]
        page_dir = args.out_dir / page
        page_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(data["lines"].items(), key=lambda kv: _line_sort_key(kv[0]))
        for stem, rec in ordered:
            (page_dir / f"{stem}.txt").write_text(rec["text"] + "\n", encoding="utf-8")
        (page_dir / f"{page}_full.txt").write_text(
            "\n".join(rec["text"] for _, rec in ordered) + "\n", encoding="utf-8"
        )
        n_pages += 1
        n_lines += len(ordered)
    print(f"DONE {n_lines} lines / {n_pages} pages -> {args.out_dir}")


if __name__ == "__main__":
    main()
