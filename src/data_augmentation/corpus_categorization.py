"""Corpus line-level categorization for synthetic-seed selection.

Walks every `*.txt` under a corpus directory, labels each non-empty line by
the categories it matches, and writes a consolidated JSON dict keyed by
``filename:lineno``. Used to surface lines that contain orthographic
patterns the HTR model currently struggles with (e.g. ``am``, ``ma``, Roman
numerals) so they can be fed into the synthetic-image renderer.

Patterns are pluggable: pass a `dict[str, Callable[[str], bool]]` mapping
category name to a line matcher. The `word_pattern` factory builds a
whole-word matcher in one line; `has_roman_numeral` is the prebuilt
strict-Roman matcher used in the default set. Mix and match:

    from src.data_augmentation.corpus_categorization import (
        DEFAULT_PATTERNS, categorize_corpus, has_roman_numeral, word_pattern,
    )

    # Extend the defaults with a couple more whole-word patterns.
    my_patterns = DEFAULT_PATTERNS | {
        "um": word_pattern("um"),
        "mo": word_pattern("mo"),
    }
    categorize_corpus(corpus_dir, output_path, patterns=my_patterns)
"""

import datetime
import json
import logging
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

LineMatcher = Callable[[str], bool]


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


# ──────────────────────────────────────────────
#  Pattern factory: whole-word match
# ──────────────────────────────────────────────


def word_pattern(word: str, case_insensitive: bool = True) -> LineMatcher:
    """Return a matcher that fires when `word` appears as a whole word.

    Whole-word means the match is bounded by `\\b` on both sides — so
    ``word_pattern("am")`` matches `"am"` but not `"amor"` or `"name"`.
    """
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


# ──────────────────────────────────────────────
#  Pattern: strict Roman numeral
# ──────────────────────────────────────────────

_RE_ROMAN_VALID = re.compile(
    r"^m{0,4}(cm|cd|d?c{0,4})(xc|xl|l?x{0,4})(ix|iv|v?i{0,4})j?$",
    re.IGNORECASE,
)

# Candidate Roman tokens: pure [ivxlcdm] (with optional trailing j), with
# capture groups for surrounding dots so we can require dot-bracketing for
# short forms.
_RE_ROMAN_CANDIDATE = re.compile(
    r"(?<![A-Za-z])(\.?)([ivxlcdm]+j?)(\.?)(?![A-Za-z])",
    re.IGNORECASE,
)


def has_roman_numeral(line: str) -> bool:
    """True if the line contains at least one token that strict-parses as a
    Roman numeral.

    Dotted forms (e.g. ``.III.``, ``.X.``) are accepted at any length because
    the dots are a strong manuscript convention for "this is a numeral."
    Undotted forms require ≥3 characters to exclude the Occitan words
    ``mi`` / ``li`` / ``vi`` / ``xi`` — all of which parse as valid Romans
    (1001 / 51 / 6 / 11) but are ordinary vocabulary in context.

    One residual false positive: undotted ``dix`` (valid Roman 509) is also
    an Occitan verb form. Blocklist it explicitly if it contaminates
    downstream training.
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


# ──────────────────────────────────────────────
#  Default pattern set
# ──────────────────────────────────────────────

DEFAULT_PATTERNS: dict[str, LineMatcher] = {
    "am": word_pattern("am"),
    "ma": word_pattern("ma"),
    "roman_numeral": has_roman_numeral,
}


# ──────────────────────────────────────────────
#  Main driver
# ──────────────────────────────────────────────


def categorize_corpus(
    corpus_dir: str | Path,
    output_path: str | Path,
    patterns: dict[str, LineMatcher] | None = None,
    *,
    encoding: str = "utf-8",
    logs_dir: str | Path | None = None,
    run_name: str | None = None,
) -> dict:
    """Walk ``corpus_dir/*.txt`` and label each non-empty line by the
    categories it matches. Save the result as a consolidated JSON at
    ``output_path``.

    Args:
        corpus_dir: Directory containing `*.txt` corpus files (top-level only).
        output_path: Output JSON file path. Parents will be created.
        patterns: Mapping of ``category_name -> line_matcher``. If None, uses
            ``DEFAULT_PATTERNS`` (``am``, ``ma``, ``roman_numeral``).
        encoding: Text-file encoding (passed to ``Path.read_text``).
        logs_dir: Optional directory for the plain-text run log. If None,
            logs go to the console only.
        run_name: Optional run identifier; used to name the log file and
            recorded in the JSON summary. Defaults to a timestamp.

    Returns:
        The full JSON document (also written to ``output_path``).
    """
    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path)
    patterns = patterns if patterns is not None else dict(DEFAULT_PATTERNS)
    assert patterns, "patterns is empty — pass DEFAULT_PATTERNS or a custom dict"

    if run_name is None:
        run_name = f"categorize_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
        "categories": list(patterns.keys()),
    }
    logger.info(f"Config: {json.dumps(config_summary)}")

    samples: dict[str, dict] = {}
    per_cat_count: Counter = Counter()
    per_cat_files: dict[str, set] = defaultdict(set)

    for path in files:
        for i, raw in enumerate(
            path.read_text(encoding=encoding, errors="replace").splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line:
                continue
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
            "lines_per_category": {c: per_cat_count[c] for c in patterns},
            "files_per_category": {c: sorted(per_cat_files[c]) for c in patterns},
            "multi_category_lines": multi_count,
        },
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    logger.info(
        f"Categorization complete: {len(samples)} lines matched, " f"{multi_count} multi-category"
    )
    for c in patterns:
        logger.info(
            f"  {c:<18} {per_cat_count[c]:>6} lines across " f"{len(per_cat_files[c])} file(s)"
        )
    logger.info(f"Output JSON: {output_path}")
    if log_file:
        logger.info(f"Run log (text): {log_file}")

    return doc
