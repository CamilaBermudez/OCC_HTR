"""Per-word recall on the 300-val set, stratified by word frequency.

For every word type that appears in the annotated corpus (600 train + 300 val
= 900 lines), compute:

  * corpus frequency (occurrences across all 900 GT files)
  * val frequency (occurrences across the 300-val GT files)
  * per-model matched count on the 300-val
  * per-model recall = matched / val_frequency

Matched count uses multiset intersection at the line level:

    matched(word) = sum_over_lines min(count_in_GT_line, count_in_pred_line)

This is the standard bag-of-words recall used in OCR benchmarks — avoids the
pain of forcing positional alignment between GT and prediction tokens.

Model prediction folders are auto-detected to handle both flat layouts
(``<dir>/<stem>.txt`` — kraken, Medusa) and nested layouts (``<dir>/<inner>/<stem>.txt``
— TrOCR run subfolders).

Written for future spec.md section 6.5.

Usage:
    python3 scripts/ocr/word_frequency_recall.py \\
        --corpus-folder data/processed/annotated_samples/OCR/full_annotated \\
        --corpus-folder data/processed/annotated_samples/OCR/validation \\
        --val-folder data/processed/annotated_samples/OCR/validation \\
        --model catmus=data/processed/transcription/ocr_kept_20260622_120413 \\
        --model kraken_600=data/processed/transcription/finetune_20260705_070741_on_validation_300 \\
        --model kraken_medical=data/processed/transcription/finetune_20260706_151856_on_validation_300 \\
        --model kraken_matched_no_medical=data/processed/transcription/finetune_20260718_193601_on_validation_300 \\
        --model kraken_matched_medical=data/processed/transcription/finetune_20260719_085411_on_validation_300 \\
        --model medusa=data/processed/transcription/medusa_validation_300_20260710_clean \\
        --model vitroberta_medical=models/ocr/finetuned/trocr_20260712_150413/transcriptions_val300 \\
        --output-dir tests/ocr/evaluations/word_frequency_recall_20260721 \\
        --manifest tests/ocr/validation_300_manifest_.csv \\
        --filter-col validated_100 --filter-value 1 \\
        --top-k 30
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv


def tokenize(text: str, lowercase: bool) -> list[str]:
    """Whitespace-split; optionally lowercase. Empty tokens dropped."""
    t = text.strip()
    if lowercase:
        t = t.lower()
    return [tok for tok in t.split() if tok]


def load_stem_texts(folder: Path) -> dict[str, str]:
    """{stem: text} for every non-empty <stem>.gt.txt file in a folder."""
    out: dict[str, str] = {}
    for gt in sorted(folder.glob("*.gt.txt")):
        text = gt.read_text(encoding="utf-8").strip()
        if text:
            out[gt.stem.replace(".gt", "")] = text
    return out


def detect_prediction_leaf(pred_dir: Path, val_stems: set[str]) -> Path:
    """Auto-detect the folder that actually contains <stem>.txt files.

    If ``pred_dir`` itself has .txt files matching val_stems, use it as-is.
    Otherwise descend into the newest sub-directory that does. Handles the
    TrOCR nested layout (``dir/inner_ts_dir/*.txt``).
    """

    def txt_matches(d: Path) -> int:
        if not d.is_dir():
            return 0
        return sum(1 for p in d.glob("*.txt") if p.stem in val_stems)

    direct = txt_matches(pred_dir)
    if direct > 0:
        return pred_dir

    subdirs = sorted(
        [p for p in pred_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True
    )
    for sub in subdirs:
        if txt_matches(sub) > 0:
            return sub

    raise SystemExit(
        f"Could not locate any *.txt files matching val stems inside {pred_dir} "
        f"or its immediate subdirectories."
    )


def read_predictions(leaf_dir: Path, val_stems: set[str]) -> dict[str, str]:
    """{stem: predicted_text} for every val stem that has a matching prediction file."""
    out: dict[str, str] = {}
    for stem in val_stems:
        f = leaf_dir / f"{stem}.txt"
        if f.is_file():
            out[stem] = f.read_text(encoding="utf-8").strip()
    return out


def compute_matched(
    gt_texts: dict[str, str], pred_texts: dict[str, str], lowercase: bool
) -> tuple[Counter, int]:
    """Return (per-word matched Counter, n_lines_scored)."""
    matched: Counter = Counter()
    scored = 0
    for stem, gt in gt_texts.items():
        pred = pred_texts.get(stem)
        if pred is None:
            continue
        gt_tokens = Counter(tokenize(gt, lowercase))
        pred_tokens = Counter(tokenize(pred, lowercase))
        # multiset intersection = min per key
        for tok, gt_count in gt_tokens.items():
            m = min(gt_count, pred_tokens.get(tok, 0))
            if m > 0:
                matched[tok] += m
        scored += 1
    return matched, scored


def main() -> None:
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-folder",
        action="append",
        required=True,
        help="Folder(s) of <stem>.gt.txt files that define the frequency corpus. "
        "Repeat for multiple folders (e.g. full_annotated + validation).",
    )
    parser.add_argument(
        "--val-folder",
        required=True,
        help="Folder of <stem>.gt.txt files used as the reference for scoring. "
        "Typically the permanent 300-val folder.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="'name=path' pair; the path is the model's prediction folder. Repeat for multiple models.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Where to write the per-word CSV + summary MD. "
        "Default: tests/ocr/evaluations/word_frequency_recall_<TS>",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top-frequency words to include in the summary MD table. Default: 30.",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Keep original case. Default: lowercase (recommended for medieval texts with inconsistent caps).",
    )
    parser.add_argument(
        "--manifest",
        required=False,
        help="Optional manifest CSV with 'stem' column for stratification (matches bootstrap_ocr_ci.py flags).",
    )
    parser.add_argument("--filter-col", required=False)
    parser.add_argument("--filter-value", required=False)
    args = parser.parse_args()

    lowercase = not args.case_sensitive

    # Optional manifest filter
    keep_stems: set[str] | None = None
    if args.manifest or args.filter_col or args.filter_value:
        if not (args.manifest and args.filter_col and args.filter_value):
            raise SystemExit("--manifest, --filter-col, and --filter-value must be given together.")
        manifest_path = (
            Path(args.manifest)
            if Path(args.manifest).is_absolute()
            else project_root / args.manifest
        )
        target = args.filter_value.strip().lower()
        keep_stems = set()
        with manifest_path.open() as f:
            reader = csv.DictReader(f)
            if args.filter_col not in reader.fieldnames:
                raise SystemExit(
                    f"Column {args.filter_col!r} not in manifest {manifest_path}; available: {reader.fieldnames}"
                )
            for row in reader:
                if row[args.filter_col].strip().lower() == target:
                    keep_stems.add(row["stem"])
        print(
            f"Manifest filter: {manifest_path.name} where {args.filter_col}={args.filter_value} -> {len(keep_stems)} stems"
        )

    # Corpus frequency
    corpus_freq: Counter = Counter()
    total_corpus_lines = 0
    for folder in args.corpus_folder:
        folder_path = Path(folder) if Path(folder).is_absolute() else project_root / folder
        texts = load_stem_texts(folder_path)
        for text in texts.values():
            corpus_freq.update(tokenize(text, lowercase))
        total_corpus_lines += len(texts)
        print(f"Corpus folder {folder_path.name}: {len(texts)} non-empty lines")
    print(
        f"Total corpus: {total_corpus_lines} lines, {len(corpus_freq)} distinct word types, {sum(corpus_freq.values())} tokens"
    )

    # Val GT + optional filter
    val_folder_path = (
        Path(args.val_folder)
        if Path(args.val_folder).is_absolute()
        else project_root / args.val_folder
    )
    val_texts = load_stem_texts(val_folder_path)
    if keep_stems is not None:
        val_texts = {s: t for s, t in val_texts.items() if s in keep_stems}
    val_stems_set = set(val_texts)
    val_freq: Counter = Counter()
    for text in val_texts.values():
        val_freq.update(tokenize(text, lowercase))
    print(
        f"Val folder: {len(val_texts)} lines used for scoring, {len(val_freq)} distinct word types, {sum(val_freq.values())} tokens"
    )

    # Parse --model 'name=path'
    model_specs: list[tuple[str, Path]] = []
    for m in args.model:
        if "=" not in m:
            raise SystemExit(f"--model expects 'name=path', got: {m!r}")
        name, path = m.split("=", 1)
        p = Path(path) if Path(path).is_absolute() else project_root / path
        model_specs.append((name, p))

    # For each model: locate prediction leaf, read predictions, compute matched
    per_model_matched: dict[str, Counter] = {}
    per_model_scored: dict[str, int] = {}
    for name, pred_dir in model_specs:
        leaf = detect_prediction_leaf(pred_dir, val_stems_set)
        preds = read_predictions(leaf, val_stems_set)
        matched, scored = compute_matched(val_texts, preds, lowercase)
        per_model_matched[name] = matched
        per_model_scored[name] = scored
        overall_recall = sum(matched.values()) / max(sum(val_freq.values()), 1)
        print(
            f"Model {name!r}: leaf={leaf}, {scored} lines scored, "
            f"total matched tokens={sum(matched.values())}/{sum(val_freq.values())} "
            f"({overall_recall * 100:.2f}% overall recall)"
        )

    # Output
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = project_root / f"tests/ocr/evaluations/word_frequency_recall_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "word_recall_per_model.csv"
    md_path = out_dir / "word_recall_summary.md"

    # CSV: one row per word type
    model_names = [n for n, _ in model_specs]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["word", "corpus_freq", "val_freq"]
        for n in model_names:
            header += [f"matched_{n}", f"recall_{n}"]
        writer.writerow(header)
        for word in sorted(corpus_freq, key=lambda w: (-corpus_freq[w], w)):
            row = [word, corpus_freq[word], val_freq.get(word, 0)]
            for n in model_names:
                m = per_model_matched[n].get(word, 0)
                r = m / val_freq[word] if val_freq.get(word, 0) > 0 else ""
                row += [m, f"{r:.4f}" if r != "" else ""]
            writer.writerow(row)
    print(f"\nPer-word CSV: {csv_path}")

    # MD summary
    lines: list[str] = []
    lines.append("# Word-frequency-stratified recall on 300-val")
    lines.append("")
    lines.append(
        f"- Corpus for frequency: {total_corpus_lines} lines, "
        f"{len(corpus_freq)} distinct word types, {sum(corpus_freq.values())} tokens"
    )
    lines.append(
        f"- Val used for scoring: {len(val_texts)} lines, "
        f"{len(val_freq)} distinct word types, {sum(val_freq.values())} tokens"
    )
    lines.append(f"- Lowercase: {lowercase}")
    if keep_stems is not None:
        lines.append(
            f"- Manifest filter: `{args.filter_col}={args.filter_value}` -> {len(keep_stems)} stems"
        )
    lines.append("")

    lines.append(f"## Top-{args.top_k} most-frequent val words with per-model recall")
    lines.append("")
    header = ["Word", "val_freq"] + model_names
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    top_words = sorted(val_freq, key=lambda w: (-val_freq[w], w))[: args.top_k]
    for w in top_words:
        row = [w, str(val_freq[w])]
        for n in model_names:
            m = per_model_matched[n].get(w, 0)
            vf = val_freq[w]
            r = m / vf if vf > 0 else 0.0
            row.append(f"{r * 100:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Aggregate recall bands per model (bag-of-words recall)")
    lines.append("")
    lines.append(
        "Recall in three bands by val_freq: **top-30 (very frequent)**, "
        "**freq 2-30 (mid)**, **hapaxes (freq = 1)**."
    )
    lines.append("")

    # Mutually-exclusive bands:
    #   top-K words (by frequency, regardless of value)
    #   mid: everything else with val_freq >= 2
    #   hapax: val_freq == 1
    top30_words = set(top_words)

    def sum_over(words, model_name):
        m = sum(per_model_matched[model_name].get(w, 0) for w in words)
        t = sum(val_freq.get(w, 0) for w in words)
        return m, t

    mid_words = {w for w, vf in val_freq.items() if vf >= 2 and w not in top30_words}
    hapax_words = {w for w, vf in val_freq.items() if vf == 1}

    band_top30 = {n: sum_over(top30_words, n) for n in model_names}
    band_mid = {n: sum_over(mid_words, n) for n in model_names}
    band_hapax = {n: sum_over(hapax_words, n) for n in model_names}

    lines.append("| Model | top-30 recall | freq 2-30 recall | hapax recall | overall recall |")
    lines.append("|---|---|---|---|---|")
    for n in model_names:
        m_top, t_top = band_top30[n]
        m_mid, t_mid = band_mid[n]
        m_hap, t_hap = band_hapax[n]
        total_matched = sum(per_model_matched[n].values())
        total_val = sum(val_freq.values())
        lines.append(
            "| "
            + " | ".join(
                [
                    n,
                    f"{m_top}/{t_top} = {100 * m_top / max(t_top, 1):.1f}%",
                    f"{m_mid}/{t_mid} = {100 * m_mid / max(t_mid, 1):.1f}%",
                    f"{m_hap}/{t_hap} = {100 * m_hap / max(t_hap, 1):.1f}%",
                    f"{total_matched}/{total_val} = {100 * total_matched / max(total_val, 1):.1f}%",
                ]
            )
            + " |"
        )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary MD:   {md_path}")


if __name__ == "__main__":
    main()
