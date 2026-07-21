"""Bootstrap 95% confidence intervals for TrOCR (and other) OCR runs on 300-val.

Loads per-line CSVs produced by ``run_evaluate_ocr.py`` (columns:
``stem, model, n_chars_ref, n_words_ref, edit_chars, edit_words, cer, wer``),
merges them into one per-line-per-model table, and reports:

1. **Per-model 95 % CIs** on corpus CER, WER, char_acc, word_acc by
   resampling the 299 lines with replacement (default 10 000 iterations).
2. **Paired bootstrap comparisons** between requested model pairs: on
   each resample, compute the corpus metrics for both models on the
   *same* resampled lines and take the difference; report the CI on
   the difference. If 0 lies outside the CI, the ordering is
   significant at the 5 % level.

Written for spec.md section 6.3.7. Uses numpy only (no pandas dependency
beyond the CSV parse in the stdlib).

Usage:
    python3 scripts/ocr/bootstrap_ocr_ci.py \\
        --eval-dir tests/ocr/evaluations/five_trocr_vs_validation_300 \\
        --eval-dir tests/ocr/evaluations/staged_and_D_vs_val300 \\
        --eval-dir tests/ocr/evaluations/stage1a_vs_val300 \\
        --eval-dir tests/ocr/evaluations/swinbert_realonly_from_scratch_vs_validation_300 \\
        --n-boot 10000 --seed 42
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


def load_eval_csv(csv_path: Path) -> dict[str, dict[str, dict[str, int]]]:
    """Return {model: {stem: {n_chars_ref, n_words_ref, edit_chars, edit_words}}}."""
    by_model: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_model[row["model"]][row["stem"]] = {
                "n_chars_ref": int(row["n_chars_ref"]),
                "n_words_ref": int(row["n_words_ref"]),
                "edit_chars": int(row["edit_chars"]),
                "edit_words": int(row["edit_words"]),
            }
    return by_model


def load_manifest_filter(manifest_path: Path, filter_col: str, filter_value: str) -> set[str]:
    """Return the set of stems where <filter_col> equals <filter_value> (case-insensitive)."""
    target = filter_value.strip().lower()
    keep: set[str] = set()
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        if filter_col not in reader.fieldnames:
            raise ValueError(
                f"Column {filter_col!r} not in manifest {manifest_path}; "
                f"available columns: {reader.fieldnames}"
            )
        for row in reader:
            if row[filter_col].strip().lower() == target:
                keep.add(row["stem"])
    if not keep:
        raise ValueError(f"No rows in manifest matched {filter_col}={filter_value}")
    return keep


def load_all(
    eval_dirs: list[Path],
    keep_stems: set[str] | None = None,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """Return (common_stems, {model: {metric_col: np.ndarray[len(common_stems)]}}).

    Only stems present in every model are kept (guarantees paired resampling).
    If ``keep_stems`` is given, intersect against it too.
    """
    all_models: dict[str, dict[str, dict[str, int]]] = {}
    for eval_dir in eval_dirs:
        csvs = list(eval_dir.glob("*.csv"))
        if not csvs:
            print(f"WARN: no CSV in {eval_dir}, skipping")
            continue
        for csv_path in csvs:
            by_model = load_eval_csv(csv_path)
            for model, stems in by_model.items():
                if model in all_models:
                    raise ValueError(
                        f"Model {model!r} appears in multiple eval dirs; "
                        "rename one to avoid collision"
                    )
                all_models[model] = stems

    if not all_models:
        raise RuntimeError("No models loaded from any eval dir")

    common_stems = set.intersection(*(set(s) for s in all_models.values()))
    if keep_stems is not None:
        common_stems &= keep_stems
    common_stems_list = sorted(common_stems)
    if not common_stems_list:
        raise RuntimeError(
            "No stems are shared across all models"
            + (" and the manifest filter" if keep_stems is not None else "")
        )

    n = len(common_stems_list)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for model, stems in all_models.items():
        cols = {
            "n_chars_ref": np.empty(n, dtype=np.int64),
            "n_words_ref": np.empty(n, dtype=np.int64),
            "edit_chars": np.empty(n, dtype=np.int64),
            "edit_words": np.empty(n, dtype=np.int64),
        }
        for i, stem in enumerate(common_stems_list):
            row = stems[stem]
            for k in cols:
                cols[k][i] = row[k]
        arrays[model] = cols

    return common_stems_list, arrays


def corpus_metric(edits: np.ndarray, refs: np.ndarray) -> float:
    total = refs.sum()
    return float(edits.sum() / total) if total > 0 else 0.0


def bootstrap_model(
    model_arrays: dict[str, np.ndarray], n_boot: int, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    n = len(model_arrays["n_chars_ref"])
    idx = rng.integers(0, n, size=(n_boot, n))

    edit_c = model_arrays["edit_chars"][idx]
    ref_c = model_arrays["n_chars_ref"][idx]
    edit_w = model_arrays["edit_words"][idx]
    ref_w = model_arrays["n_words_ref"][idx]

    cer_samples = edit_c.sum(axis=1) / np.where(ref_c.sum(axis=1) == 0, 1, ref_c.sum(axis=1))
    wer_samples = edit_w.sum(axis=1) / np.where(ref_w.sum(axis=1) == 0, 1, ref_w.sum(axis=1))
    char_acc_samples = 1.0 - cer_samples
    word_acc_samples = 1.0 - wer_samples

    def summ(a: np.ndarray) -> dict[str, float]:
        return {
            "point": float(np.mean(a)),
            "median": float(np.median(a)),
            "low95": float(np.quantile(a, 0.025)),
            "high95": float(np.quantile(a, 0.975)),
        }

    return {
        "cer": summ(cer_samples),
        "wer": summ(wer_samples),
        "char_acc": summ(char_acc_samples),
        "word_acc": summ(word_acc_samples),
    }


def paired_bootstrap(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Bootstrap CI of the *difference* a - b on each metric.

    Uses paired resampling: the same random line indices are applied to
    both models on every iteration, so the CI reflects the fact that
    both models were evaluated on the same underlying lines.
    """
    n = len(a["n_chars_ref"])
    idx = rng.integers(0, n, size=(n_boot, n))

    def corpus_series(arr: dict[str, np.ndarray], is_char: bool) -> np.ndarray:
        edits = arr["edit_chars" if is_char else "edit_words"][idx]
        refs = arr["n_chars_ref" if is_char else "n_words_ref"][idx]
        ref_sum = refs.sum(axis=1)
        return edits.sum(axis=1) / np.where(ref_sum == 0, 1, ref_sum)

    d_cer = corpus_series(a, True) - corpus_series(b, True)
    d_wer = corpus_series(a, False) - corpus_series(b, False)
    d_char_acc = -d_cer
    d_word_acc = -d_wer

    def summ(a: np.ndarray) -> dict[str, float]:
        return {
            "point": float(np.mean(a)),
            "low95": float(np.quantile(a, 0.025)),
            "high95": float(np.quantile(a, 0.975)),
            "p_a_better": float(np.mean(a > 0)),
        }

    return {
        "d_cer": summ(d_cer),
        "d_wer": summ(d_wer),
        "d_char_acc": summ(d_char_acc),
        "d_word_acc": summ(d_word_acc),
    }


DEFAULT_PAIRS = [
    (
        "vitroberta_medical",
        "vitroberta_cometa",
        "corpus ablation: medical vs COMETA (pretrained arch, matched)",
    ),
    ("vitroberta_medical", "vitroberta_realonly", "medical aug vs no aug (pretrained arch)"),
    (
        "vitroberta_realonly",
        "runD2_vitroberta",
        "real-only vs Dataset D (re-render volume without external corpus)",
    ),
    ("stage2b_medical", "swinbert_medical", "staged Swin+BERT vs single-stage Swin+BERT (B'')"),
    ("stage2a_cometa", "stage2b_medical", "staged: does corpus choice matter?"),
    (
        "stage1a_cometa_pretrain",
        "swinbert_medical",
        "COMETA pretrain alone vs single-stage baseline",
    ),
    (
        "stage2b_medical",
        "stage1a_cometa_pretrain",
        "does manuscript FT add value on top of Stage 1?",
    ),
    ("vitroberta_medical", "stage2b_medical", "best pretrained TrOCR vs best staged Swin+BERT"),
]


def fmt_pct(x: float) -> str:
    return f"{100 * x:+6.2f}%"


def fmt_pct_unsigned(x: float) -> str:
    return f"{100 * x:6.2f}%"


def main() -> None:
    load_dotenv()
    project_root = Path(os.environ.get("PROJECT_ROOT", "."))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        action="append",
        required=True,
        help="Path to an eval folder containing <name>.csv per-line file. "
        "Repeat for multiple folders.",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="Additional 'model_a=model_b=label' triple for paired comparison. "
        "Repeat as needed. Defaults defined in script are always run.",
    )
    parser.add_argument(
        "--manifest",
        required=False,
        help="Optional path to a manifest CSV with a 'stem' column. When paired "
        "with --filter-col + --filter-value, restricts the bootstrap to the "
        "matching subset of stems.",
    )
    parser.add_argument(
        "--filter-col",
        required=False,
        help="Column in --manifest to filter on (e.g. 'validated_100').",
    )
    parser.add_argument(
        "--filter-value",
        required=False,
        help="Value to require in --filter-col (case-insensitive). Only stems "
        "where the column equals this value are kept.",
    )
    args = parser.parse_args()

    keep_stems: set[str] | None = None
    if args.manifest or args.filter_col or args.filter_value:
        if not (args.manifest and args.filter_col and args.filter_value):
            raise SystemExit("--manifest, --filter-col, and --filter-value must be given together.")
        manifest_path = (
            Path(args.manifest)
            if Path(args.manifest).is_absolute()
            else project_root / args.manifest
        )
        keep_stems = load_manifest_filter(manifest_path, args.filter_col, args.filter_value)
        print(
            f"Manifest filter: {manifest_path.name} "
            f"where {args.filter_col}={args.filter_value} -> {len(keep_stems)} stems selected"
        )

    eval_dirs = [Path(d) if Path(d).is_absolute() else project_root / d for d in args.eval_dir]
    stems, arrays = load_all(eval_dirs, keep_stems=keep_stems)
    print(f"Loaded {len(arrays)} models across {len(stems)} shared lines")
    print(f"Bootstrap: {args.n_boot} iterations, seed={args.seed}\n")

    rng = np.random.default_rng(args.seed)

    print("## Per-model 95 % CIs (paired-bootstrap on the 299-line held-out set)\n")
    header = f"| {'model':40s} | {'char_acc [95% CI]':30s} | {'word_acc [95% CI]':30s} | {'CER [95% CI]':30s} | {'WER [95% CI]':30s} |"
    sep = "|" + "|".join(["-" * (w + 2) for w in [40, 30, 30, 30, 30]]) + "|"
    print(header)
    print(sep)
    for model in sorted(arrays.keys()):
        res = bootstrap_model(arrays[model], args.n_boot, rng)
        row = (
            f"| {model:40s}"
            f" | {fmt_pct_unsigned(res['char_acc']['point'])} [{fmt_pct_unsigned(res['char_acc']['low95'])}, {fmt_pct_unsigned(res['char_acc']['high95'])}]"
            f" | {fmt_pct_unsigned(res['word_acc']['point'])} [{fmt_pct_unsigned(res['word_acc']['low95'])}, {fmt_pct_unsigned(res['word_acc']['high95'])}]"
            f" | {res['cer']['point']:.4f} [{res['cer']['low95']:.4f}, {res['cer']['high95']:.4f}]"
            f" | {res['wer']['point']:.4f} [{res['wer']['low95']:.4f}, {res['wer']['high95']:.4f}] |"
        )
        print(row)

    print("\n## Paired-bootstrap comparisons (A vs B)\n")
    print("Interpretation: 95 % CI on the *difference* A - B on the same resampled lines.")
    print("If 0 is outside the CI, the ordering is statistically significant (α = 0.05).")
    print("P(A better) = fraction of bootstrap resamples in which A's char_acc > B's.\n")

    pairs = list(DEFAULT_PAIRS)
    for p in args.pair or []:
        try:
            a, b, label = p.split("=", 2)
        except ValueError:
            raise SystemExit(f"--pair must be 'a=b=label', got {p!r}") from None
        pairs.append((a, b, label))

    header2 = f"| {'A':32s} | {'B':32s} | {'Δ char_acc [95% CI]':30s} | {'Δ word_acc [95% CI]':30s} | {'P(A > B)':10s} | note |"
    sep2 = "|" + "|".join(["-" * (w + 2) for w in [32, 32, 30, 30, 10, 6]]) + "|"
    print(header2)
    print(sep2)
    for a, b, label in pairs:
        if a not in arrays or b not in arrays:
            print(f"| {a:32s} | {b:32s} | (skipped — missing model) | | | {label} |")
            continue
        res = paired_bootstrap(arrays[a], arrays[b], args.n_boot, rng)
        p_a = res["d_char_acc"]["p_a_better"]
        sig = ""
        if p_a >= 0.975 or p_a <= 0.025:
            sig = " ✓sig"
        row = (
            f"| {a:32s}"
            f" | {b:32s}"
            f" | {fmt_pct(res['d_char_acc']['point'])} [{fmt_pct(res['d_char_acc']['low95'])}, {fmt_pct(res['d_char_acc']['high95'])}]"
            f" | {fmt_pct(res['d_word_acc']['point'])} [{fmt_pct(res['d_word_acc']['low95'])}, {fmt_pct(res['d_word_acc']['high95'])}]"
            f" | {p_a:6.3f}{sig:5s}"
            f" | {label} |"
        )
        print(row)


if __name__ == "__main__":
    main()
