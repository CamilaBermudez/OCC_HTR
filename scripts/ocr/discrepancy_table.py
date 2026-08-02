"""Export every OCR-vs-scholarly line discrepancy as a flat table for analysis.

One row per difference (from the same banded word-level diff the viewer uses,
spec §6.7.3), so patterns can be counted later — by category, by specific
confusion pair (``model_span`` -> ``scholarly_span``), or by page. Reuses
``word_align.diff_page_banded`` + ``line_diff`` so the table always matches what
the 3-way viewer shows.

Columns:
  page                 page key
  scholarly_row        scholarly line number the model line is aligned to
  model_row            model segmentation-line index (seg_idx)
  category             substitution | addition | deletion | abbreviation |
                       spacing | punctuation | orthographic
  group                substantive | editorial | scramble (viewer visibility)
  scholarly_span       the differing scholarly text (the "base")
  model_span           the differing model/OCR text
  scholarly_line_text  full scholarly line (context)
  model_line_text      full model line (context)

Usage (defaults to the banded word-NW diff; needs the model's line_alignment.json):
    PROJECT_ROOT=. uv run python scripts/ocr/discrepancy_table.py \
        --model-dir data/processed/transcription/finetune_400_full_corpus \
        --scholarly-txt tests/ocr/AlbucE_aligned_20260628_142959.txt \
        --output tests/ocr/evaluations/discrepancies/finetune_400.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.ocr.line_diff import diff_group, diff_page
from src.ocr.word_align import diff_page_banded

_HEADER_RE = re.compile(r"=+\s*IMAGE:\s*(?P<key>.+?)_full\s*=+")
_LINE_RE = re.compile(r"^(?P<no>\d+):\s?(?P<text>.*)$")
_MODEL_LINE_RE = re.compile(r"_line_(\d+)\.txt$")

FIELDS = [
    "page",
    "scholarly_row",
    "model_row",
    "category",
    "group",
    "scholarly_span",
    "model_span",
    "scholarly_line_text",
    "model_line_text",
]


def load_scholarly(path: Path) -> dict[str, list[tuple[int, str]]]:
    pages: dict[str, list[tuple[int, str]]] = {}
    cur: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
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
    ap.add_argument("--method", choices=("banded", "free"), default="banded")
    ap.add_argument("--alignment-json", type=Path, default=None)
    ap.add_argument(
        "--groups",
        default="substantive,editorial,scramble",
        help="comma list of groups to include (default: all).",
    )
    ap.add_argument(
        "--top-pairs", type=int, default=40, help="pairs per category in JSON by_category"
    )
    args = ap.parse_args()

    keep_groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    scholarly = load_scholarly(args.scholarly_txt)
    align_all: dict = {}
    if args.method == "banded":
        align_path = args.alignment_json or (args.model_dir / "line_alignment.json")
        align_all = json.loads(align_path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    cat_counts = Counter()
    pair_counts = Counter()  # (category, model_span, scholarly_span) for pattern analysis
    by_cat_pairs: dict[str, Counter] = {}  # category -> Counter[(model_span, scholarly_span)]

    for page_dir in sorted(p for p in args.model_dir.iterdir() if p.is_dir()):
        pk = page_dir.name
        model = load_model_page(args.model_dir, pk)
        if not model or pk not in scholarly:
            continue
        sch_pairs = scholarly[pk]
        sch_text = {no: t for no, t in sch_pairs}
        model_text = {seg: t for seg, t in model}
        sch_lines = [t for _, t in sch_pairs]

        if args.method == "banded" and pk in align_all:
            align = {int(k): v for k, v in align_all[pk]["model_to_scholarly"].items()}
            diffs = diff_page_banded(sch_lines, model, align)
        else:
            align = {}
            diffs = diff_page(sch_lines, model)

        for d in diffs:
            grp = diff_group(d)
            if grp not in keep_groups:
                continue
            seg = d.ocr_line
            sch_no = align.get(seg) if seg is not None else None
            cat_counts[d.type] += 1
            pair_counts[(d.type, d.ocr_text, d.base_text)] += 1
            by_cat_pairs.setdefault(d.type, Counter())[(d.ocr_text, d.base_text)] += 1
            rows.append(
                {
                    "page": pk,
                    "scholarly_row": sch_no if sch_no is not None else "",
                    "model_row": seg if seg is not None else "",
                    "category": d.type,
                    "group": grp,
                    "scholarly_span": d.base_text,
                    "model_span": d.ocr_text,
                    "scholarly_line_text": sch_text.get(sch_no, "") if sch_no else "",
                    "model_line_text": model_text.get(seg, "") if seg is not None else "",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".json":
        payload = {
            "model_dir": str(args.model_dir),
            "method": args.method,
            "groups": sorted(keep_groups),
            "n_discrepancies": len(rows),
            "category_totals": dict(cat_counts.most_common()),
            # top confusion pairs per category — the pattern view
            "by_category": {
                cat: [
                    {"model_span": mo, "scholarly_span": ba, "count": n}
                    for (mo, ba), n in counter.most_common(args.top_pairs)
                ]
                for cat, counter in sorted(
                    by_cat_pairs.items(), key=lambda kv: -sum(kv[1].values())
                )
            },
            "rows": rows,  # every discrepancy, for arbitrary slicing
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        with args.output.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    print(f"wrote {len(rows)} discrepancies -> {args.output}")
    print("\ncategory totals:")
    for c, n in cat_counts.most_common():
        print(f"  {c:14} {n}")
    print("\ntop confusion pairs per category (model -> scholarly | count):")
    for cat, counter in sorted(by_cat_pairs.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  [{cat}]  (total {sum(counter.values())})")
        for (mo, ba), n in counter.most_common(8):
            print(f"      {n:4}  {mo!r} -> {ba!r}")


if __name__ == "__main__":
    main()
