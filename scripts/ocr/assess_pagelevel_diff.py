"""Honest evaluation of the SHIPPED page-level diff, bucketed by root cause.

Unlike ``assess_line_errors_buckets.py`` (which diffs each aligned pair in
isolation and so inflates line-edge spill), this runs the real shipped path —
``diff_page`` over the *whole page* concatenated, exactly as the viewer does —
then buckets every emitted span. This is the true distribution of what the tool
shows the user. Reusable per model via ``--model-dir``.

    PROJECT_ROOT=. uv run python scripts/ocr/assess_pagelevel_diff.py \
        --model-dir data/processed/transcription/ocr_kept_20260622_120413 \
        --scholarly-txt tests/ocr/AlbucE_aligned_20260628_142959.txt
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from src.ocr.line_diff import _fold, diff_page, split_diffs
from src.ocr.word_align import diff_page_banded

_HEADER_RE = re.compile(r"=+\s*IMAGE:\s*(?P<key>.+?)_full\s*=+")
_LINE_RE = re.compile(r"^(?P<no>\d+):\s?(?P<text>.*)$")
_MODEL_LINE_RE = re.compile(r"_line_(\d+)\.txt$")
_ARTICLES = {"lo", "la", "los", "las", "lu", "l", "le", "els"}


def load_scholarly(path: Path) -> dict[str, list[str]]:
    pages: dict[str, list[str]] = {}
    cur: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        h = _HEADER_RE.match(raw)
        if h:
            cur = h.group("key")
            pages.setdefault(cur, [])
            continue
        m = _LINE_RE.match(raw)
        if cur is not None and m:
            pages[cur].append(m.group("text").rstrip())
    return pages


def load_model_page(model_dir: Path, page_key: str) -> list[tuple[int, str]]:
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


def bucket(dtype: str, o: str, b: str) -> str:
    if dtype == "punctuation":
        return "editorial_punct"
    if dtype == "abbreviation":
        return "abbrev_expansion"
    if dtype == "deletion":
        return "article_split" if _fold(b) in _ARTICLES else "content_del"
    if dtype == "addition":
        return "article_split" if _fold(o) in _ARTICLES else "content_add"
    return "substitution"


def _foldsim(a: str, b: str) -> float:
    fa, fb = _fold(a), _fold(b)
    if not fa and not fb:
        return 1.0
    return 1.0 - Levenshtein.distance(fa, fb) / max(len(fa), len(fb), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--scholarly-txt", type=Path, required=True)
    ap.add_argument(
        "--method",
        choices=("free", "banded"),
        default="free",
        help="free = shipped char-level diff_page; banded = anchored word-level "
        "NW (needs <model-dir>/line_alignment.json).",
    )
    args = ap.parse_args()

    scholarly = load_scholarly(args.scholarly_txt)
    align_all: dict = {}
    if args.method == "banded":
        import json

        align_all = json.loads((args.model_dir / "line_alignment.json").read_text(encoding="utf-8"))

    def run_diff(pk: str, model: list[tuple[int, str]]):
        if args.method == "banded":
            align = {int(k): v for k, v in align_all[pk]["model_to_scholarly"].items()}
            return diff_page_banded(scholarly[pk], model, align)
        return diff_page(scholarly[pk], model)

    raw_buckets = Counter()
    split_counts = Counter()
    n_pages = n_ocr_lines = n_raw = 0
    tight = loose = addel = 0
    subs: list[str] = []
    scrambles: list[str] = []
    for page_dir in sorted(p for p in args.model_dir.iterdir() if p.is_dir()):
        pk = page_dir.name
        model = load_model_page(args.model_dir, pk)
        if not model or pk not in scholarly:
            continue
        n_pages += 1
        n_ocr_lines += len(model)
        diffs = run_diff(pk, model)
        n_raw += len(diffs)
        for d in diffs:
            raw_buckets[bucket(d.type, d.ocr_text, d.base_text)] += 1
        substantive, editorial, scramble = split_diffs(diffs)
        split_counts["substantive"] += len(substantive)
        split_counts["editorial"] += len(editorial)
        split_counts["scramble"] += len(scramble)
        for d in substantive:
            if d.type == "substitution":
                if _foldsim(d.ocr_text, d.base_text) >= 0.5:
                    tight += 1
                else:
                    loose += 1
                if len(subs) < 60:
                    subs.append(f"{d.ocr_text!r}->{d.base_text!r}")
            else:
                addel += 1
        for d in scramble:
            if len(scrambles) < 10:
                scrambles.append(f"[{pk}] {d.type}:{(d.ocr_text or d.base_text)[:70]!r}...")

    print(f"model: {args.model_dir.name}  |  method: {args.method}")
    print(
        f"pages {n_pages} | OCR lines {n_ocr_lines} | raw diff spans {n_raw} "
        f"({n_raw/n_ocr_lines:.2f}/line)\n"
    )
    order = [
        ("editorial_punct", "EDITORIAL — editor punctuation"),
        ("abbrev_expansion", "EDITORIAL — brevigraph expansion"),
        ("article_split", "EDITORIAL — de+lo/la spacing"),
        ("content_del", "OCR error — dropped word"),
        ("content_add", "OCR error — over-generated word"),
        ("substitution", "OCR error — misread / variant"),
    ]
    print("--- BEFORE (raw shipped diff_page, by category) ---")
    print(f"{'bucket':<18} {'spans':>6} {'share':>7}   root cause")
    for k, desc in order:
        v = raw_buckets.get(k, 0)
        print(f"{k:<18} {v:>6} {100*v/n_raw:>6.1f}%   {desc}")

    print("\n--- AFTER (split_diffs: editorial suppressed + scramble guard) ---")
    s, e, sc = split_counts["substantive"], split_counts["editorial"], split_counts["scramble"]
    tot = s + e + sc
    print(
        f"  substantive (real OCR diffs) : {s:>6} ({100*s/tot:.0f}%)  -> {s/n_ocr_lines:.2f}/line"
    )
    print(f"  editorial   (suppressed)     : {e:>6} ({100*e/tot:.0f}%)")
    print(f"  scramble    (flag region)    : {sc:>6} ({100*sc/tot:.0f}%)")
    print(f"\n  NOISE REMOVED from the shown view: {100*(e+sc)/tot:.0f}% of raw spans")
    nsub = tight + loose
    print("\n--- substantive quality ---")
    print(
        f"  substitutions {nsub} | TIGHT(genuine) {100*tight/nsub:.0f}% ({tight/n_ocr_lines:.2f}/line)"
        f" | LOOSE(misalign) {100*loose/nsub:.0f}% ({loose/n_ocr_lines:.2f}/line)"
    )
    print(f"  add/del spans {addel} ({addel/n_ocr_lines:.2f}/line)")
    print(
        "\nsample SUBSTANTIVE substitutions (should be genuine misreads):\n  "
        + "\n  ".join(subs[:40])
    )
    print("\nsample SCRAMBLE spans (flagged, not shown as edits):\n  " + "\n  ".join(scrambles))


if __name__ == "__main__":
    main()
