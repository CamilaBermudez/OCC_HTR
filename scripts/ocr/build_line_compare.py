"""Phase 1 of the model-comparison tab (spec §7.4.1): per-page comparison JSON.

For every kept physical line, combine:
  * the kept crop image path,
  * the scholarly text (content-matched to the page's scholarly lines — robust to
    the imperfect index alignment; scholarly ≈ 1:1 with physical lines),
  * catmus text + per-CHARACTER confidence (from catmus_conf_fullms),
  * ViT text + per-TOKEN confidence (from vit_conf_fullms),
  * mismatch flags — each model vs scholarly and vs the other model — computed on
    FOLDED text (lowercase, alphanumerics only) so editorial spacing/punctuation
    (§6.7) is not flagged; only substantive letter disagreements are.

Output: one JSON per page under --out-dir, ready for the carousel tab to lazy-load.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from rapidfuzz.fuzz import partial_ratio_alignment

sys.path.insert(0, ".")
from frontend.manuscript_data import _parse_scholarly  # noqa: E402


def fold(s: str) -> tuple[str, list[int]]:
    """Lowercase, alphanumerics only; return (folded, raw-index-of-each-folded-char)."""
    out, idx = [], []
    for i, ch in enumerate(s):
        if ch.isalnum():
            out.append(ch.lower())
            idx.append(i)
    return "".join(out), idx


def char_mismatch(model_text: str, ref_text: str) -> list[int]:
    """1 per raw model char that substantively disagrees with ref (folded compare)."""
    fm, fmi = fold(model_text)
    fr, _ = fold(ref_text)
    mism = [0] * len(model_text)
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=fr, b=fm, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            for j in range(j1, j2):
                mism[fmi[j]] = 1
    return mism


def build_scholarly_index(sp: dict[int, str]):
    """Concatenate a page's scholarly lines into one continuous text so a physical
    line can be matched to a SPAN (not a whole edition line — those don't align to
    manuscript line breaks: the edition sometimes keeps a whole sentence as one
    'line' spanning several physical lines). Returns
    (raw_concat, folded_concat, folded->raw index map, [(raw_start, raw_end, disp_no)])."""
    keys = sorted(sp)
    raw_parts, bounds, pos = [], [], 0
    for key in keys:
        t = sp[key]
        bounds.append((pos, pos + len(t), key + 1))  # +1 => aligned-file display no
        raw_parts.append(t)
        pos += len(t) + 1  # +1 for the joining space
    raw_concat = " ".join(raw_parts)
    folded, f2raw = fold(raw_concat)
    return raw_concat, folded, f2raw, bounds


def scholarly_for_line(ocr_folded, raw_concat, folded_concat, f2raw, bounds, min_score=55):
    """Best-matching scholarly SPAN for one physical line. Returns (text, disp_no, score)."""
    if not ocr_folded or not folded_concat:
        return "", None, 0
    a = partial_ratio_alignment(ocr_folded, folded_concat)
    if a is None or a.score < min_score or a.dest_end <= a.dest_start:
        return "", None, int(a.score) if a else 0
    rs, re_ = f2raw[a.dest_start], f2raw[a.dest_end - 1] + 1
    text = raw_concat[rs:re_].strip()
    no = next((disp for (ls, le, disp) in bounds if ls <= rs < le), None)
    return text, no, int(a.score)


def seg_of(stem: str) -> int:
    try:
        return int(stem.split("_line_")[1])
    except (IndexError, ValueError):
        return 1 << 30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catmus-dir", type=Path, default=Path("data/processed/transcription/catmus_conf_fullms")
    )
    ap.add_argument(
        "--vit-dir", type=Path, default=Path("data/processed/transcription/vit_conf_fullms")
    )
    ap.add_argument(
        "--scholarly-txt", type=Path, default=Path("tests/ocr/AlbucE_aligned_20260628_142959.txt")
    )
    ap.add_argument(
        "--crops-rel",
        default="20260618_160948/original/kept",
        help="kept-crops path relative to filtered_images (for the frontend image URL)",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed/line_compare"))
    ap.add_argument("--limit-pages", type=int, default=0)
    args = ap.parse_args()

    scholarly = _parse_scholarly(args.scholarly_txt)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cat_pages = sorted(p.stem for p in args.catmus_dir.glob("[0-9]*.json"))
    if args.limit_pages:
        cat_pages = cat_pages[: args.limit_pages]

    tot = 0
    for page in cat_pages:
        cat = json.loads((args.catmus_dir / f"{page}.json").read_text())["lines"]
        vit_path = args.vit_dir / f"{page}.json"
        vit = json.loads(vit_path.read_text())["lines"] if vit_path.exists() else {}
        sp = scholarly.get(page, {})
        raw_concat, folded_concat, f2raw, bounds = (
            build_scholarly_index(sp) if sp else ("", "", [], [])
        )

        out_lines = []
        for stem in sorted(cat, key=seg_of):
            c = cat[stem]
            v = vit.get(stem, {"text": "", "tokens": []})
            ctext = c["text"]
            # scholarly = best-matching SPAN in the continuous scholarly text
            stext, sno, ssim = scholarly_for_line(
                fold(ctext)[0], raw_concat, folded_concat, f2raw, bounds
            )

            # ViT display text = concatenation of its token surfaces (keeps token↔char exact)
            vtoks = v.get("tokens", [])
            vtext = "".join(t for t, _ in vtoks)
            offs, pos = [], 0
            for t, _ in vtoks:
                offs.append((pos, pos + len(t)))
                pos += len(t)

            c_ms = char_mismatch(ctext, stext) if stext else [0] * len(ctext)
            c_mv = char_mismatch(ctext, vtext)
            v_char_ms = char_mismatch(vtext, stext) if stext else [0] * len(vtext)
            v_char_mc = char_mismatch(vtext, ctext)

            chars = [
                [ctext[i], c["chars"][i][1] if i < len(c["chars"]) else 1.0, c_ms[i], c_mv[i]]
                for i in range(len(ctext))
            ]
            tokens = [
                [t, p, int(any(v_char_ms[a:b])), int(any(v_char_mc[a:b]))]
                for (t, p), (a, b) in zip(vtoks, offs, strict=False)
            ]

            out_lines.append(
                {
                    "stem": stem,
                    "seg": seg_of(stem),
                    "image": f"{args.crops_rel}/{page}/{stem}.png",
                    "scholarly": {"text": stext, "match_sim": ssim, "no": sno},
                    "catmus": {"text": ctext, "chars": chars},
                    "vit": {"text": vtext, "tokens": tokens},
                }
            )

        (args.out_dir / f"{page}.json").write_text(
            json.dumps(
                {"page": page, "n_lines": len(out_lines), "lines": out_lines}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        tot += len(out_lines)
        print(f"{page}: {len(out_lines)} lines", flush=True)

    print(f"DONE {tot} lines across {len(cat_pages)} pages -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
