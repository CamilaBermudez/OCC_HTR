"""Measure the CER floor imposed by each TrOCR-track tokenizer on the 300-val GT.

For every ground-truth line: encode -> decode with each tokenizer, then compute
CER of the round-trip vs the original. The result is the *lower bound* on CER
that any model using that tokenizer could achieve, even under a perfect
encoder. Comparing the two floors quantifies how much of the observed
Swin+BERT vs ViT+RoBERTa gap is attributable to tokenizer coverage rather than
cross-attention pretraining.

Also reports the specific characters that map to ``[UNK]`` under each
tokenizer, to distinguish "systematic loss" (a whole character class is
un-representable) from "edge cases" (one-off oddities at line boundaries).

Written for spec.md section 6.3.6.

Usage:
    python3 scripts/ocr/analyze_tokenizer_floor.py \\
        --val-dir data/processed/annotated_samples/OCR/validation
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import rapidfuzz
from dotenv import load_dotenv
from transformers import AutoTokenizer

DEFAULT_TOKENIZERS = {
    "mBERT (Swin+BERT decoder)": "bert-base-multilingual-cased",
    "RoBERTa BPE (ViT+RoBERTa decoder)": "microsoft/trocr-base-handwritten",
}


def load_gt(val_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    for gt_file in sorted(val_dir.glob("*.gt.txt")):
        text = gt_file.read_text(encoding="utf-8").strip()
        if text:
            pairs.append((gt_file.stem, text))
    return pairs


def round_trip(tokenizer, text: str, skip_special: bool) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids, skip_special_tokens=skip_special)


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0
    return rapidfuzz.distance.Levenshtein.distance(ref, hyp) / len(ref)


def analyze(name: str, model_id: str, gts: list[tuple[str, str]]) -> None:
    print(f"\n=== {name} - {model_id} ===")
    tok = AutoTokenizer.from_pretrained(model_id)
    unk_token = tok.unk_token or "<no-unk>"
    print(f"  vocab_size={len(tok.get_vocab())}, unk_token={unk_token!r}")

    total_dist_skip = 0
    total_dist_keep = 0
    total_chars = 0
    per_line_cer: list[tuple[float, str, str, str]] = []
    unk_char_counter: Counter = Counter()
    lines_with_unk = 0
    perfect_lines = 0

    for stem, ref in gts:
        hyp_skip = round_trip(tok, ref, skip_special=True)
        hyp_keep = round_trip(tok, ref, skip_special=False)

        d_skip = rapidfuzz.distance.Levenshtein.distance(ref, hyp_skip)
        d_keep = rapidfuzz.distance.Levenshtein.distance(ref, hyp_keep)

        total_dist_skip += d_skip
        total_dist_keep += d_keep
        total_chars += len(ref)
        per_line_cer.append((cer(ref, hyp_skip), stem, ref, hyp_skip))

        if unk_token in hyp_keep:
            lines_with_unk += 1
            i = 0
            while i < len(hyp_keep):
                if hyp_keep[i : i + len(unk_token)] == unk_token:
                    if i < len(ref):
                        unk_char_counter[ref[i]] += 1
                    i += len(unk_token)
                else:
                    i += 1
        if d_skip == 0:
            perfect_lines += 1

    corpus_cer_skip = total_dist_skip / total_chars if total_chars else 0.0
    corpus_cer_keep = total_dist_keep / total_chars if total_chars else 0.0

    print(f"  total GT chars: {total_chars}, lines: {len(gts)}")
    print(
        f"  perfect round-trip lines: {perfect_lines}/{len(gts)} "
        f"({100 * perfect_lines / len(gts):.1f}%)"
    )
    print(f"  lines containing at least one {unk_token}: {lines_with_unk}")
    print(
        f"  corpus floor CER (skip specials in decode): {corpus_cer_skip:.4f}  "
        f"-> implied char_acc ceiling: {1 - corpus_cer_skip:.4f}"
    )
    print(f"  corpus floor CER (keep specials in decode): {corpus_cer_keep:.4f}")

    if unk_char_counter:
        print(f"  top-15 characters that trigger {unk_token}:")
        for ch, count in unk_char_counter.most_common(15):
            hex_repr = " ".join(f"U+{ord(c):04X}" for c in ch)
            print(f"    {ch!r:8}  ({hex_repr})  count={count}")

    per_line_cer.sort(reverse=True)
    print("  5 worst round-trip lines (skip-specials decoding):")
    for c, stem, ref, hyp in per_line_cer[:5]:
        print(f"    cer={c:.3f}  {stem}")
        print(f"      REF: {ref}")
        print(f"      HYP: {hyp}")


def main() -> None:
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--val-dir",
        required=False,
        help="Folder of <stem>.gt.txt reference files. Default: "
        "data/processed/annotated_samples/OCR/validation",
    )
    args = parser.parse_args()

    val_dir = (
        Path(args.val_dir)
        if args.val_dir
        else project_root / "data/processed/annotated_samples/OCR/validation"
    )
    gts = load_gt(val_dir)
    print(f"Loaded {len(gts)} non-empty GT lines from {val_dir}")

    for name, model_id in DEFAULT_TOKENIZERS.items():
        analyze(name, model_id, gts)


if __name__ == "__main__":
    main()
