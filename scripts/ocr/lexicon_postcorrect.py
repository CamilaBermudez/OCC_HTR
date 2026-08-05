"""Lexicon post-correction for OCR line predictions (catmus → corrected).

Conservative, OOV-only word-level correction. For each predicted token:
  * keep it untouched if its normalized form is IN the lexicon (in-vocab), or too
    short to fuzzy-match safely (length-aware threshold);
  * otherwise fuzzy-match the normalized core against the lexicon and, if a
    confident match exists, replace the letter-core with the lexicon's preferred
    spelling (punctuation + surrounding whitespace preserved).

Lexicon sources (NONE use the val GT — no leak):
  * DOM_lemma_variants.json          (curated medieval-Occitan dictionary)
  * --train-gt-dir  full_annotated   (600 real TRAIN lines — DIPLOMATIC space)
  * --medical-corpus categorized JSON (12k Occitan medical lines)

Preferred output spelling per normalized form: TRAIN-GT diplomatic > medical
corpus > DOM headword — so corrections stay in the diplomatic convention the
GT uses (keeps ⁊/tildes/etc.) rather than expanding abbreviations.

Usage:
  PROJECT_ROOT=. uv run python scripts/ocr/lexicon_postcorrect.py \
    --pred-dir data/processed/transcription/ocr_kept_20260622_120413 \
    --stem-dir data/processed/annotated_samples/OCR/validation \
    --dictionary data/raw/DOM_lemma_variants.json \
    --train-gt-dir data/processed/annotated_samples/OCR/full_annotated \
    --medical-corpus data/processed/synthetic_seeds/categorize_20260625_143327/medical_texts_categorized.json \
    --out-dir data/processed/transcription/catmus_lexcorr_val300 \
    --fuzzy-threshold 88
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz, process

from src.ocr.dictionary_evaluation import length_aware_threshold, normalize_old_occitan

# token = optional leading non-alpha, alpha core (letters incl. accented/combining
# via \w minus digits handled below), optional trailing non-alpha.
_TOKEN_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)


def _core_split(token: str) -> tuple[str, str, str]:
    """Split a whitespace-delimited token into (prefix, core, suffix)."""
    m = _TOKEN_RE.match(token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def build_lexicon(
    dictionary: Path | None, train_gt_dir: Path | None, medical_corpus: Path | None
) -> tuple[set[str], dict[str, str]]:
    """Return (valid_forms, form_to_output). form_to_output maps a normalized
    form to the preferred surface spelling (diplomatic train GT wins)."""
    valid: set[str] = set()
    out: dict[str, str] = {}

    def add(norm: str, surface: str, priority: int) -> None:
        # priority: higher wins. Track chosen priority in a parallel dict.
        if not norm:
            return
        valid.add(norm)
        prev = _PRIO.get(norm, -1)
        if priority > prev:
            out[norm] = surface
            _PRIO[norm] = priority

    _PRIO: dict[str, int] = {}

    # 1) DOM dictionary (priority 0) — headwords + variants, normalized surface.
    if dictionary:
        dom = json.loads(dictionary.read_text(encoding="utf-8"))
        for head, variants in dom.items():
            add(normalize_old_occitan(head), head.lower(), 0)
            if isinstance(variants, list):
                for var in variants:
                    add(normalize_old_occitan(var), var.lower(), 0)

    # 2) medical corpus (priority 1) — surface = the corpus token (lowercased).
    if medical_corpus:
        data = json.loads(medical_corpus.read_text(encoding="utf-8"))
        samples = data.get("samples", {}) if isinstance(data, dict) else {}
        for entry in samples.values():
            for tok in re.findall(r"\S+", entry.get("text", "").lower()):
                _, core, _ = _core_split(tok)
                add(normalize_old_occitan(core), core, 1)

    # 3) TRAIN GT diplomatic (priority 2, wins) — surface = the real GT token.
    if train_gt_dir:
        for f in glob.glob(str(train_gt_dir / "*.gt.txt")):
            for tok in re.findall(r"\S+", Path(f).read_text(encoding="utf-8").lower()):
                _, core, _ = _core_split(tok)
                add(normalize_old_occitan(core), core, 2)

    return valid, out


def find_pred(pred_dir: Path, stem: str) -> Path | None:
    """Locate <stem>.txt anywhere under pred_dir (flat or nested-by-page)."""
    flat = pred_dir / f"{stem}.txt"
    if flat.is_file():
        return flat
    hits = list(pred_dir.rglob(f"{stem}.txt"))
    return hits[0] if hits else None


def correct_line(
    text: str,
    valid: set[str],
    out: dict[str, str],
    valid_list: list[str],
    base_threshold: float,
    stats: Counter,
) -> str:
    corrected = []
    for token in text.split():
        prefix, core, suffix = _core_split(token)
        norm = normalize_old_occitan(core)
        stats["tokens"] += 1
        if not norm or norm in valid:
            corrected.append(token)  # empty core or in-vocab → untouched
            continue
        stats["oov"] += 1
        thr = length_aware_threshold(norm, base_threshold)
        if thr is None:
            corrected.append(token)  # too short to fuzzy safely
            continue
        match = process.extractOne(norm, valid_list, scorer=fuzz.ratio, score_cutoff=thr)
        if match is None:
            corrected.append(token)
            continue
        replacement = out[match[0]]
        stats["corrected"] += 1
        corrected.append(f"{prefix}{replacement}{suffix}")
    return " ".join(corrected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument(
        "--stem-dir",
        required=True,
        type=Path,
        help="Dir of <stem>.gt.txt whose stems define which lines to correct.",
    )
    ap.add_argument("--dictionary", type=Path)
    ap.add_argument("--train-gt-dir", type=Path)
    ap.add_argument("--medical-corpus", type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--fuzzy-threshold", type=float, default=88.0)
    args = ap.parse_args()

    valid, out = build_lexicon(args.dictionary, args.train_gt_dir, args.medical_corpus)
    valid_list = sorted(valid)
    print(
        f"lexicon: {len(valid)} normalized forms "
        f"(dict={bool(args.dictionary)} train_gt={bool(args.train_gt_dir)} "
        f"medical={bool(args.medical_corpus)})"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted(
        Path(f).name[: -len(".gt.txt")] for f in glob.glob(str(args.stem_dir / "*.gt.txt"))
    )
    stats: Counter = Counter()
    written = missing = 0
    for stem in stems:
        pred = find_pred(args.pred_dir, stem)
        if pred is None:
            missing += 1
            continue
        fixed = correct_line(
            pred.read_text(encoding="utf-8").strip(),
            valid,
            out,
            valid_list,
            args.fuzzy_threshold,
            stats,
        )
        (args.out_dir / f"{stem}.txt").write_text(fixed + "\n", encoding="utf-8")
        written += 1

    print(f"written {written} lines ({missing} preds missing) -> {args.out_dir}")
    print(
        f"tokens={stats['tokens']} oov={stats['oov']} "
        f"corrected={stats['corrected']} "
        f"({100*stats['corrected']/max(stats['tokens'],1):.1f}% of tokens, "
        f"{100*stats['corrected']/max(stats['oov'],1):.1f}% of OOV)"
    )


if __name__ == "__main__":
    main()
