import datetime
import json
import logging
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")


def setup_simple_logging(
    logs_dir: str, task_name: str = "dictionary_eval", run_name: str | None = None
):
    """Initialize logging with file + console handlers using consistent formatting."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(logs_dir) / f"{run_name}_{task_name}.log"

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console_handler = logging.StreamHandler()

    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("=== %s Run Started | Run: %s ===", task_name.upper(), run_name)
    logger.info("Log file: %s", log_file)

    return logger, str(log_file)


def normalize_old_occitan(word: str) -> str:
    """Lowercase and keep only alphabetic characters (strip digits, punctuation, spaces)."""
    return "".join(c.lower() for c in word if c.isalpha())


def prepare_occitan_dict(custom_dict: dict):
    """
    Expand headwords + their variants into a flat set of valid forms, plus a
    reverse map from any valid form to its canonical headword. If a variant
    appears under multiple headwords, the first one wins (JSON insertion order);
    headwords always take precedence over variants.
    """
    valid_forms = set()
    form_to_headword = {}

    for head, variants in custom_dict.items():
        h_norm = normalize_old_occitan(head)
        if h_norm:
            valid_forms.add(h_norm)
            form_to_headword[h_norm] = head

        if isinstance(variants, list):
            for var in variants:
                v_norm = normalize_old_occitan(var)
                if v_norm:
                    valid_forms.add(v_norm)
                    form_to_headword.setdefault(v_norm, head)

    return valid_forms, form_to_headword


def length_aware_threshold(word: str, base_threshold: float):
    """
    Return the fuzzy threshold to use for `word`, or None to skip fuzzy
    matching entirely.

      length <= 4  -> None (require exact match)
      length == 5  -> base_threshold + 10 (stricter)
      length >= 6  -> base_threshold
    """
    n = len(word)
    if n <= 4:
        return None
    if n <= 5:
        return min(base_threshold + 10, 100)
    return base_threshold


def load_transcription_dir(transcription_dir: Path) -> tuple[dict, int]:
    """Read every *.txt file at the top level of `transcription_dir` and return
    (word_count_dict, n_files_loaded) from the concatenated lowercased text."""
    txt_files = sorted(transcription_dir.glob("*.txt"))
    text_parts = []
    for txt_file in txt_files:
        with open(txt_file, encoding="utf-8") as fh:
            text_parts.append(fh.read().lower())
    text = "\n".join(text_parts)
    words = re.findall(r"\b\w+\b", text)
    return dict(Counter(words)), len(txt_files)


def load_dictionary(dictionary_path: Path) -> dict:
    """Load the lemma-variant JSON dictionary and lowercase all keys + variants."""
    with open(dictionary_path, encoding="utf-8") as f:
        custom_dict = json.load(f)
    return {
        k.lower(): (v if v is None else [item.lower() for item in v])
        for k, v in custom_dict.items()
    }


def evaluate_ocr_occitan(
    ocr_counter: dict,
    custom_dict: dict,
    fuzzy_threshold: float = 82,
    min_word_length: int = 2,
) -> dict:
    """
    Evaluate an OCR transcription against an Occitan dictionary.

    Returns a dict containing:
      - exact and combined (exact + fuzzy) word/token accuracy
      - OOV (out-of-vocabulary) counts after fuzzy matching
      - DataFrame of fuzzy match candidates
      - frequency-sorted list of words that matched nothing
    """
    valid_forms, form_to_headword = prepare_occitan_dict(custom_dict)
    valid_list = sorted(valid_forms)

    # Normalize OCR tokens, drop anything shorter than min_word_length.
    ocr_norm_map: dict[str, list] = {}
    dropped_short_tokens = 0
    for word, count in ocr_counter.items():
        w_norm = normalize_old_occitan(word)
        if not w_norm:
            continue
        if len(w_norm) < min_word_length:
            dropped_short_tokens += count
            continue
        if w_norm not in ocr_norm_map:
            ocr_norm_map[w_norm] = [word, 0]
        ocr_norm_map[w_norm][1] += count

    ocr_norm_words = set(ocr_norm_map.keys())
    exact_matches = ocr_norm_words & valid_forms
    unknown_words = ocr_norm_words - valid_forms

    total_tokens = sum(c for _, c in ocr_norm_map.values())
    exact_tokens = sum(ocr_norm_map[w][1] for w in exact_matches)

    word_acc_exact = len(exact_matches) / len(ocr_norm_words) if ocr_norm_words else 0
    token_acc_exact = exact_tokens / total_tokens if total_tokens else 0

    fuzzy_candidates = []
    fuzzy_matched_norm = set()

    for u_norm in unknown_words:
        threshold = length_aware_threshold(u_norm, fuzzy_threshold)
        if threshold is None:
            continue
        orig, freq = ocr_norm_map[u_norm]
        matches = process.extract(u_norm, valid_list, limit=1, scorer=fuzz.ratio)
        if not matches:
            continue
        best_form, best_score, _ = matches[0]
        if best_score >= threshold:
            fuzzy_matched_norm.add(u_norm)
            headword = form_to_headword.get(best_form, best_form)
            fuzzy_candidates.append(
                {
                    "ocr_original": orig,
                    "ocr_normalized": u_norm,
                    "frequency": freq,
                    "matched_form": best_form,
                    "suggested_headword": headword,
                    "fuzzy_score": best_score,
                    "word_length": len(u_norm),
                    "threshold_used": threshold,
                }
            )

    fuzzy_df = pd.DataFrame(fuzzy_candidates)
    if not fuzzy_df.empty:
        fuzzy_df = fuzzy_df.sort_values("frequency", ascending=False).reset_index(drop=True)

    fuzzy_tokens = sum(ocr_norm_map[w][1] for w in fuzzy_matched_norm)
    combined_word_acc = (
        (len(exact_matches) + len(fuzzy_matched_norm)) / len(ocr_norm_words)
        if ocr_norm_words
        else 0
    )
    combined_token_acc = (exact_tokens + fuzzy_tokens) / total_tokens if total_tokens else 0

    oov_norm = unknown_words - fuzzy_matched_norm
    top_unmatched = sorted(
        (
            {
                "ocr_original": ocr_norm_map[w][0],
                "ocr_normalized": w,
                "frequency": ocr_norm_map[w][1],
            }
            for w in oov_norm
        ),
        key=lambda r: r["frequency"],
        reverse=True,
    )
    top_unmatched_df = pd.DataFrame(top_unmatched)

    return {
        "word_accuracy_exact": word_acc_exact,
        "token_accuracy_exact": token_acc_exact,
        "word_accuracy_with_fuzzy": combined_word_acc,
        "token_accuracy_with_fuzzy": combined_token_acc,
        "total_unique_words": len(ocr_norm_words),
        "total_tokens": total_tokens,
        "known_words_exact": len(exact_matches),
        "fuzzy_match_words": len(fuzzy_matched_norm),
        "oov_words": len(oov_norm),
        "oov_tokens": sum(ocr_norm_map[w][1] for w in oov_norm),
        "dropped_short_tokens": dropped_short_tokens,
        "fuzzy_candidates": fuzzy_df,
        "top_unmatched": top_unmatched_df,
    }


def log_evaluation_summary(logger: logging.Logger, results: dict) -> None:
    """Emit the OCR vs. Dictionary Evaluation summary block to the logger."""
    lines = [
        "",
        "=" * 60,
        "OCR vs. Dictionary Evaluation",
        "=" * 60,
        f"Total unique words (after filtering):  {results['total_unique_words']:>6}",
        f"Total tokens:                          {results['total_tokens']:>6}",
        f"Dropped (too-short OCR noise):         {results['dropped_short_tokens']:>6}",
        "",
        f"Exact matches      (words):            {results['known_words_exact']:>6}",
        f"Fuzzy matches      (words):            {results['fuzzy_match_words']:>6}",
        f"Out-of-vocabulary  (words):            {results['oov_words']:>6}",
        f"Out-of-vocabulary  (tokens):           {results['oov_tokens']:>6}",
        "",
        f"Word accuracy  (exact only):    {results['word_accuracy_exact'] * 100:>5.1f}%",
        f"Word accuracy  (exact + fuzzy): {results['word_accuracy_with_fuzzy'] * 100:>5.1f}%",
        f"Token accuracy (exact only):    {results['token_accuracy_exact'] * 100:>5.1f}%",
        f"Token accuracy (exact + fuzzy): {results['token_accuracy_with_fuzzy'] * 100:>5.1f}%",
    ]
    for line in lines:
        logger.info(line)


def run_dictionary_evaluation(
    transcription_dir: str | Path,
    dictionary_path: str | Path,
    output_dir: str | Path | None = None,
    fuzzy_threshold: float = 82,
    min_word_length: int = 2,
    logs_dir: str | Path | None = None,
    task_name: str = "dictionary_eval",
    run_name: str | None = None,
    log_config: bool = True,
) -> dict:
    if logs_dir is None:
        logs_dir = Path(PROJECT_ROOT) / "logs" / "dictionary_eval"
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger, log_file = setup_simple_logging(
        logs_dir=str(logs_dir), task_name=task_name, run_name=run_name
    )

    transcription_dir = Path(transcription_dir)
    dictionary_path = Path(dictionary_path)

    if log_config:
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=PROJECT_ROOT,
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            git_commit = "unknown"

        config = {
            "run_name": run_name,
            "git_commit": git_commit,
            "timestamp": datetime.datetime.now().isoformat(),
            "transcription_dir": str(transcription_dir),
            "dictionary_path": str(dictionary_path),
            "output_dir": str(output_dir) if output_dir else None,
            "fuzzy_threshold": fuzzy_threshold,
            "min_word_length": min_word_length,
            "environment": {"PROJECT_ROOT": os.environ.get("PROJECT_ROOT")},
        }
        logger.info("Configuration: %s", json.dumps(config, indent=2))

    if not transcription_dir.is_dir():
        logger.error("Transcription folder not found: %s", transcription_dir)
        return {"success": False}
    if not dictionary_path.is_file():
        logger.error("Dictionary file not found: %s", dictionary_path)
        return {"success": False}

    logger.info("Loading transcription files from %s", transcription_dir)
    freq_dict_trans, n_files = load_transcription_dir(transcription_dir)
    logger.info(
        "Loaded %d file(s) | %d unique tokens | %d total occurrences",
        n_files,
        len(freq_dict_trans),
        sum(freq_dict_trans.values()),
    )

    logger.info("Loading dictionary: %s", dictionary_path)
    custom_dict = load_dictionary(dictionary_path)
    logger.info("Dictionary headwords: %d", len(custom_dict))

    logger.info(
        "Evaluating (fuzzy_threshold=%s, min_word_length=%s)",
        fuzzy_threshold,
        min_word_length,
    )
    results = evaluate_ocr_occitan(
        freq_dict_trans,
        custom_dict,
        fuzzy_threshold=fuzzy_threshold,
        min_word_length=min_word_length,
    )

    log_evaluation_summary(logger, results)

    if output_dir is not None:
        run_output_dir = Path(output_dir) / run_name
        run_output_dir.mkdir(parents=True, exist_ok=True)
        fuzzy_csv = run_output_dir / "fuzzy_candidates.csv"
        oov_csv = run_output_dir / "top_unmatched.csv"
        metrics_json = run_output_dir / "metrics.json"
        results["fuzzy_candidates"].to_csv(fuzzy_csv, index=False)
        results["top_unmatched"].to_csv(oov_csv, index=False)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in results.items() if not isinstance(v, pd.DataFrame)},
                f,
                indent=2,
            )
        logger.info("Saved fuzzy candidates: %s", fuzzy_csv)
        logger.info("Saved OOV list:        %s", oov_csv)
        logger.info("Saved metrics JSON:    %s", metrics_json)

    logger.info("Log file: %s", log_file)
    results["success"] = True
    results["log_file"] = log_file
    return results
