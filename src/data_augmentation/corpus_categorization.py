"""Corpus line-level categorization for synthetic-seed selection."""

import datetime
import json
import logging
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

LineMatcher = Callable[[str], bool]


def _iter_file_lines(text: str) -> Iterator[str]:
    """Yield non-empty stripped lines split on newlines."""
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            yield line


def _iter_cut_lines(text: str, length_pool: list[int], rng: random.Random) -> Iterator[str]:
    """Yield pseudo-lines built by cutting whitespace-tokenised text into
    chunks of N words where N is drawn from ``length_pool``.

    The pool is the empirical OCR line-length distribution (e.g. the list
    of 13,647 per-line word counts collected by
    ``notebooks/ocr/ocr_line_length_stats.ipynb``), so the resulting
    pseudo-corpus has the same length shape as what the OCR model
    actually produces. The final partial chunk is yielded as-is even if
    it's shorter than the sampled length.
    """
    assert length_pool, "length_pool must be non-empty for cut mode"
    words = text.split()
    if not words:
        return
    i = 0
    n = len(words)
    while i < n:
        target = rng.choice(length_pool)
        chunk = words[i : i + target]
        yield " ".join(chunk)
        i += target


def setup_categorization_logging(logs_dir: str | Path, run_name: str):
    """File + console logger for a categorization run. Mirrors the pattern in
    `src/data_preprocessing/crop_image_segments.py`."""
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(logs_dir) / f"{run_name}_categorization.log"

    logger = logging.getLogger("categorization")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for h in (file_handler, console):
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger, str(log_file)


def _get_git_commit() -> str:
    """Short git SHA at PROJECT_ROOT, or 'unknown' if unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.environ.get("PROJECT_ROOT", "."),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def word_pattern(word: str, case_insensitive: bool = True) -> LineMatcher:
    flags = re.IGNORECASE if case_insensitive else 0
    regex = re.compile(rf"\b{re.escape(word)}\b", flags)

    def matches(line: str) -> bool:
        return bool(regex.search(line))

    matches.__name__ = f"word_pattern_{word}"
    matches.__doc__ = (
        f"Whole-word match for {word!r} "
        f"(case-{'insensitive' if case_insensitive else 'sensitive'})."
    )
    return matches


def substring_pattern(substring: str, case_insensitive: bool = True) -> LineMatcher:
    """Return a matcher that fires when `substring` appears anywhere in the line.

    Unlike `word_pattern`, no word boundary is required, so
    ``substring_pattern("un")`` matches both standalone ``"un"`` and ``"un"``
    inside larger words like ``"unexpected"`` or ``"comun"``.
    """
    if case_insensitive:
        needle = substring.lower()

        def matches(line: str) -> bool:
            return needle in line.lower()
    else:
        needle = substring

        def matches(line: str) -> bool:
            return needle in line

    matches.__name__ = f"substring_pattern_{substring}"
    matches.__doc__ = (
        f"Substring match for {substring!r} "
        f"(case-{'insensitive' if case_insensitive else 'sensitive'})."
    )
    return matches


# ──────────────────────────────────────────────
#  Pattern: strict Roman numeral
# ──────────────────────────────────────────────

_RE_ROMAN_VALID = re.compile(
    r"^m{0,4}(cm|cd|d?c{0,4})(xc|xl|l?x{0,4})(ix|iv|v?i{0,4})j?$", re.IGNORECASE
)
_RE_ROMAN_CANDIDATE = re.compile(
    r"(?<![A-Za-z])(\.?)([ivxlcdm]+j?)(\.?)(?![A-Za-z])", re.IGNORECASE
)


def has_roman_numeral(line: str) -> bool:
    """True if the line contains at least one token that strict-parses as a
    Roman numeral.
    """
    for m in _RE_ROMAN_CANDIDATE.finditer(line):
        pre, tok, post = m.group(1), m.group(2), m.group(3)
        dotted = bool(pre) or bool(post)
        if not dotted and len(tok) < 3:
            continue
        if not _RE_ROMAN_VALID.match(tok):
            continue
        return True
    return False


DEFAULT_PATTERNS: dict[str, LineMatcher] = {
    "am": word_pattern("am"),
    "ma": word_pattern("ma"),
    "roman_numeral": has_roman_numeral,
}


def categorize_corpus(
    corpus_dir: str | Path,
    output_dir: str | Path,
    patterns: dict[str, LineMatcher] | None = None,
    *,
    encoding: str = "utf-8",
    logs_dir: str | Path | None = None,
    run_name: str | None = None,
    output_filename: str = "cometa_categorized.json",
    cut_to_lines: bool = False,
    length_pool: list[int] | None = None,
    keep_all: bool = False,
    seed: int = 42,
) -> dict:
    """Scan a corpus and label every line that matches any of ``patterns``.

    Two optional behaviours for source text without manuscript-style line
    breaks (e.g. a transcribed paragraph corpus):

    - ``cut_to_lines=True`` reads each file as one stream of words and
      cuts it into pseudo-lines of length drawn from ``length_pool``
      (typically the empirical OCR per-line word counts). This is the
      "give me OCR-shaped fake lines from a paragraph corpus" mode.

    - ``keep_all=True`` skips pattern filtering and emits every line
      with category ``["all"]``. Use when you want every line of the
      corpus to become a synthetic seed, not just lines containing
      specific patterns. ``patterns`` may be ``None`` in this mode.
    """
    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    if not keep_all:
        patterns = patterns if patterns is not None else dict(DEFAULT_PATTERNS)
        assert patterns, "patterns is empty — pass DEFAULT_PATTERNS or a custom dict"
    if cut_to_lines:
        assert length_pool, "cut_to_lines=True requires a non-empty length_pool"
    rng = random.Random(seed)

    if run_name is None:
        run_name = f"categorize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Final output lands under `output_dir / run_name / output_filename` so
    # every run's artefacts (JSON + future per-run extras) live together
    # under the run subdirectory — matches the convention in the other
    # src/ scripts (parchment, medieval text, augmentation).
    save_dir = output_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / output_filename

    # Logger setup — file + console if logs_dir, console only otherwise.
    if logs_dir:
        logger, log_file = setup_categorization_logging(logs_dir, run_name)
    else:
        logger = logging.getLogger("categorization")
        if not logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        log_file = None

    logger.info(f"=== Categorization Started | Run: {run_name} ===")

    files = sorted(corpus_dir.glob("*.txt"))
    assert files, f"No *.txt files found in {corpus_dir}"

    config_summary = {
        "run": run_name,
        "git": _get_git_commit(),
        "corpus_dir": str(corpus_dir),
        "output_path": str(output_path),
        "n_files": len(files),
        "categories": ["all"] if keep_all else list(patterns.keys()),
        "cut_to_lines": cut_to_lines,
        "keep_all": keep_all,
        "length_pool_size": len(length_pool) if length_pool else 0,
        "seed": seed,
    }
    logger.info(f"Config: {json.dumps(config_summary)}")

    samples: dict[str, dict] = {}
    per_cat_count: Counter = Counter()
    per_cat_files: dict[str, set] = defaultdict(set)

    for path in files:
        text = path.read_text(encoding=encoding, errors="replace")
        if cut_to_lines:
            line_iter = enumerate(_iter_cut_lines(text, length_pool, rng), start=1)
        else:
            line_iter = enumerate(_iter_file_lines(text), start=1)
        for i, line in line_iter:
            if keep_all:
                cats = ["all"]
            else:
                cats = [name for name, match in patterns.items() if match(line)]
                if not cats:
                    continue
            samples[f"{path.name}:{i}"] = {"categories": cats, "text": line}
            for c in cats:
                per_cat_count[c] += 1
                per_cat_files[c].add(path.name)

    multi_count = sum(1 for v in samples.values() if len(v["categories"]) > 1)

    doc = {
        "summary": {
            **config_summary,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "total_lines_matched": len(samples),
            "lines_per_category": {
                c: per_cat_count[c] for c in (["all"] if keep_all else patterns)
            },
            "files_per_category": {
                c: sorted(per_cat_files[c]) for c in (["all"] if keep_all else patterns)
            },
            "multi_category_lines": multi_count,
        },
        "samples": samples,
    }

    output_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    logger.info(
        f"Categorization complete: {len(samples)} lines matched, " f"{multi_count} multi-category"
    )
    for c in ["all"] if keep_all else patterns:
        logger.info(
            f"  {c:<18} {per_cat_count[c]:>6} lines across " f"{len(per_cat_files[c])} file(s)"
        )
    logger.info(f"Output JSON: {output_path}")
    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return doc
