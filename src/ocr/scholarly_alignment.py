"""
Two-pass alignment of a noisy per-line OCR transcription to a clean reference text.

Aligns the scholarly edition (continuous text, no manuscript lineation) to the
manuscript's line structure using an auxiliary OCR transcription as the guide
(spec: produced tests/ocr/AlbucE_aligned_*.txt). Extracted verbatim from
notebooks/ocr/text_alignment.ipynb (cell 1); CLI wrapper:
scripts/ocr/run_scholarly_alignment.py.

Pass 1 (per-page anchored DP)
-----------------------------
For each OCR page, find the rough anchor in the reference via n-gram
similarity, then run a word-level DP against a window of reference
words. From the DP, record ONE number per OCR line: the maximum
(absolute) reference index that any OCR word in that line matched.

Pass 2 (lossless partition)
---------------------------
Walk the entire reference word array exactly once and partition every
word into a per-line bucket using the anchors from pass 1.
- Forward-clamp any backward drift in the anchors (DP is locally optimal
  but can produce out-of-order hi values across adjacent lines).
- Linearly interpolate anchors for lines that matched nothing, using the
  nearest known neighbours on each side.
- The first line's range starts at 0; the last line's range ends at the
  last reference word index. Together this guarantees that the union of
  per-line ranges is exactly [0, n_ref_words - 1] — no gaps, no overlaps.

Net effect: concatenating all aligned lines reproduces the reference
file exactly, just with line breaks inserted at the boundaries OCR
detected. enforce_page_boundaries is no longer needed (the partition is
monotonic by construction) but is kept as a no-op for API compatibility.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

# ──────────────────────────────────────────────
#  1.  Text normalisation helpers
# ──────────────────────────────────────────────


def strip_diacritics(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalise(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.lower()
    text = strip_diacritics(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenise(text: str) -> list[str]:
    return normalise(text).split()


def _normalise_token(raw: str) -> str:
    """Per-token version of ``normalise`` — no cross-token hyphen joins."""
    w = strip_diacritics(raw.lower())
    return re.sub(r"[^a-z0-9]", "", w)


def build_ref_arrays(full_ref_text: str) -> tuple[list[str], list[str], list[int]]:
    """Split reference text into raw + normalised arrays plus an index map.

    Returns ``(raw_tokens, norm_tokens, raw_idx_of_norm)`` where:
      - ``raw_tokens = full_ref_text.split()`` — every whitespace token.
      - ``norm_tokens`` is the per-raw-token normalised form, skipping tokens
        that normalise to the empty string (pure punctuation, etc.).
      - ``raw_idx_of_norm[i]`` is the index in ``raw_tokens`` that produced
        ``norm_tokens[i]``.

    The partition algorithm operates on the normalised space; at emit time we
    use ``raw_idx_of_norm`` to convert each (norm_lo, norm_hi) range back to
    a raw range, preserving the original text including punctuation.
    """
    raw_tokens = full_ref_text.split()
    norm_tokens: list[str] = []
    raw_idx_of_norm: list[int] = []
    for j, raw in enumerate(raw_tokens):
        norm = _normalise_token(raw)
        if norm:
            norm_tokens.append(norm)
            raw_idx_of_norm.append(j)
    return raw_tokens, norm_tokens, raw_idx_of_norm


# ──────────────────────────────────────────────
#  2.  Fast n-gram anchor search
# ──────────────────────────────────────────────


def char_ngrams(text: str, n: int = 3) -> set[str]:
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    ga, gb = char_ngrams(a, n), char_ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def find_anchor(
    ocr_words: list[str],
    ref_words: list[str],
    search_from: int = 0,
    search_to: int | None = None,
    expected_idx: int | None = None,
    probe_len: int = 30,
    step: int = 10,
    window_mult: int = 6,
    min_anchor_score: float = 0.10,
    proximity_weight: float = 0.15,
) -> tuple[int, float]:
    """N-gram anchor search, tightly bounded and biased toward expected_idx.

    Bounds: [search_from, search_to). When expected_idx is given, the raw
    n-gram score is penalised proportional to |i - expected_idx|, which
    breaks ties towards the predicted position and resists matching on
    later occurrences of repeated medieval phrases.
    """
    probe = " ".join(ocr_words[:probe_len])
    window_size = probe_len * window_mult

    max_end = len(ref_words) - probe_len
    if search_to is None:
        search_to = max_end
    bounded_end = min(max_end, max(search_from + 1, search_to))

    fallback = expected_idx if expected_idx is not None else search_from
    fallback = max(search_from, min(fallback, max(search_from, bounded_end - 1)))

    best_idx = fallback
    best_raw = -1.0
    best_score = -1.0

    span = max(1, bounded_end - search_from)

    for i in range(search_from, bounded_end, step):
        window = " ".join(ref_words[i : i + window_size])
        raw = ngram_similarity(probe, window)
        if expected_idx is not None:
            dist = abs(i - expected_idx) / span
            adjusted = raw - proximity_weight * dist
        else:
            adjusted = raw
        if adjusted > best_score:
            best_score = adjusted
            best_raw = raw
            best_idx = i

    if best_raw < min_anchor_score:
        return fallback, best_raw

    return best_idx, best_raw


# ──────────────────────────────────────────────
#  3.  Word-level DP alignment with merge / split moves
# ──────────────────────────────────────────────


def word_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _match_score(
    ocr_w: str,
    ref_w: str,
    match_reward: float,
    mismatch_penalty: float,
    sim_threshold: float,
) -> float:
    sim = word_similarity(ocr_w, ref_w)
    return match_reward if sim >= sim_threshold else mismatch_penalty * (1 - sim)


def align_word_sequences(
    ocr_words: list[str],
    ref_words: list[str],
    match_reward: float = 2.0,
    mismatch_penalty: float = -1.0,
    gap_penalty: float = -0.5,
    merge_penalty: float = -0.3,
    sim_threshold: float = 0.6,
) -> list[tuple[int | None, int | None]]:
    """Semi-global DP alignment supporting diag/up/left/merge/split moves."""
    m, n = len(ocr_words), len(ref_words)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    tb = [[(0, 0)] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        tb[i][0] = (i - 1, 0)
    for j in range(1, n + 1):
        dp[0][j] = 0.0
        tb[0][j] = (0, j - 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sm = _match_score(
                ocr_words[i - 1],
                ref_words[j - 1],
                match_reward,
                mismatch_penalty,
                sim_threshold,
            )
            diag = dp[i - 1][j - 1] + sm
            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty

            best, best_prev = diag, (i - 1, j - 1)
            if up > best:
                best, best_prev = up, (i - 1, j)
            if left > best:
                best, best_prev = left, (i, j - 1)

            if i >= 2:
                merged = ocr_words[i - 2] + ocr_words[i - 1]
                ms = _match_score(
                    merged,
                    ref_words[j - 1],
                    match_reward,
                    mismatch_penalty,
                    sim_threshold,
                )
                mv = dp[i - 2][j - 1] + ms + merge_penalty
                if mv > best:
                    best, best_prev = mv, (i - 2, j - 1)

            if j >= 2:
                merged_r = ref_words[j - 2] + ref_words[j - 1]
                ss = _match_score(
                    ocr_words[i - 1],
                    merged_r,
                    match_reward,
                    mismatch_penalty,
                    sim_threshold,
                )
                sv = dp[i - 1][j - 2] + ss + merge_penalty
                if sv > best:
                    best, best_prev = sv, (i - 1, j - 2)

            dp[i][j] = best
            tb[i][j] = best_prev

    last_row = dp[m]
    j_end = max(range(n + 1), key=lambda j: last_row[j])

    pairs: list[tuple[int | None, int | None]] = []
    i, j = m, j_end
    while i > 0 or j > 0:
        pi, pj = tb[i][j]
        di, dj = i - pi, j - pj
        if di == 1 and dj == 1:
            pairs.append((i - 1, j - 1))
        elif di == 1 and dj == 0:
            pairs.append((i - 1, None))
        elif di == 0 and dj == 1:
            pairs.append((None, j - 1))
        elif di == 2 and dj == 1:
            pairs.append((i - 1, j - 1))
            pairs.append((i - 2, j - 1))
        elif di == 1 and dj == 2:
            pairs.append((i - 1, j - 1))
            pairs.append((i - 1, j - 2))
        i, j = pi, pj

    pairs.reverse()
    return pairs


# ──────────────────────────────────────────────
#  4.  Pass-2 partition
# ──────────────────────────────────────────────


def _robust_line_hi(ref_indices: list[int], line_word_count: int) -> int:
    """Pick a per-line 'last covered ref index' that is robust to spurious
    far-forward word matches.

    The previous implementation took ``max(ref_indices)``, which let one
    OCR word with a far-out coincidental match (e.g. a repeated stop word
    re-appearing 600 ref-words later in a noisy page) balloon the line's
    range and dump the bulk of the page into a single line.

    Strategy: sort the matches and walk left to right, cutting at the
    first gap larger than ``max(20, line_word_count * 3)``. Any tail of
    matches sitting alone past a large gap is dropped as an outlier. The
    last surviving (largest) match is returned. ``ref_indices`` must be
    non-empty.
    """
    sorted_ix = sorted(ref_indices)
    gap_limit = max(20, line_word_count * 3)
    cut = len(sorted_ix)
    for i in range(len(sorted_ix) - 1):
        if sorted_ix[i + 1] - sorted_ix[i] > gap_limit:
            cut = i + 1
            break
    return sorted_ix[cut - 1]


def _per_line_hi(
    line_word_counts: list[int],
    pairs: list[tuple[int | None, int | None]],
    anchor_idx: int,
) -> list[int | None]:
    """For each OCR line, return a robust absolute ref index for the line's
    last covered word. None for lines with no matches."""
    ocr_to_refs: dict[int, list[int]] = {}
    for ocr_i, ref_i in pairs:
        if ocr_i is not None and ref_i is not None:
            ocr_to_refs.setdefault(ocr_i, []).append(ref_i)

    his: list[int | None] = []
    cursor = 0
    for word_count in line_word_counts:
        if word_count == 0:
            his.append(None)
            continue
        ref_indices = [
            r
            for k in range(word_count)
            if (cursor + k) in ocr_to_refs
            for r in ocr_to_refs[cursor + k]
        ]
        if ref_indices:
            his.append(anchor_idx + _robust_line_hi(ref_indices, word_count))
        else:
            his.append(None)
        cursor += word_count
    return his


def partition_reference_to_lines(
    line_his: list[int | None],
    n_ref_words: int,
) -> list[tuple[int, int] | None]:
    """Partition reference words 0..n_ref_words-1 into per-line ranges.

    Inputs:
        line_his: per-line max-matched absolute ref index in reading order
                  (None for unmatched lines).
        n_ref_words: total reference word count.

    Output: list of (lo, hi) inclusive ranges, one per line. Ranges are
    non-overlapping, monotonic, and collectively cover [0, n_ref_words-1].
    A range is None only when the whole document had zero matches AND the
    line is a zero-OCR-words line (degenerate inputs).
    """
    n_lines = len(line_his)
    if n_lines == 0:
        return []

    # Step 1: forward-clamp known anchors so hi is monotonic non-decreasing.
    his: list[int | None] = list(line_his)
    last_known = -1
    for i in range(n_lines):
        if his[i] is not None:
            if his[i] < last_known:
                his[i] = last_known
            else:
                last_known = his[i]

    # Step 2: interpolate anchors for lines that matched nothing.
    known_idx = [i for i, h in enumerate(his) if h is not None]
    if not known_idx:
        # Zero matches across the whole doc — distribute evenly so we still
        # emit every reference word.
        per_line = max(1, n_ref_words // n_lines)
        his = [min(n_ref_words - 1, (i + 1) * per_line - 1) for i in range(n_lines)]
    else:
        ext_pos = [-1] + known_idx + [n_lines]
        ext_hi = [-1] + [his[i] for i in known_idx] + [n_ref_words - 1]
        for i in range(n_lines):
            if his[i] is not None:
                continue
            for k in range(len(ext_pos) - 1):
                if ext_pos[k] < i < ext_pos[k + 1]:
                    span_pos = ext_pos[k + 1] - ext_pos[k]
                    span_hi = ext_hi[k + 1] - ext_hi[k]
                    frac = (i - ext_pos[k]) / span_pos
                    his[i] = int(round(ext_hi[k] + span_hi * frac))
                    break

    # Step 3: pin the last line to cover the document tail.
    his[-1] = max(his[-1] if his[-1] is not None else -1, n_ref_words - 1)

    # Step 4: convert per-line hi into (lo, hi) ranges.
    ranges: list[tuple[int, int] | None] = []
    prev_hi = -1
    for i in range(n_lines):
        lo = prev_hi + 1
        hi = his[i]
        if hi < lo:
            ranges.append(None)
        else:
            ranges.append((lo, hi))
            prev_hi = hi
    return ranges


# ──────────────────────────────────────────────
#  5.  Driver — two-pass alignment
# ──────────────────────────────────────────────


def align_ocr_to_reference(
    full_ref_text: str,
    ocr_dir: Path,
    *,
    probe_len: int = 25,
    ref_window_words: int = 300,
    anchor_back_slack: int = 1500,
    anchor_fwd_slack: int = 400,
    min_anchor_score: float = 0.10,
    proximity_weight: float = 0.15,
    recheck_match_threshold: float = 0.80,
    recheck_proximity_weight: float = 0.05,
    per_page_restart: bool = True,
) -> tuple[list[tuple[str, list[str]]], list[dict]]:
    """Two-pass aligner. Pass 1: per-page anchored DP, recording one anchor
    per OCR line. Pass 2: partition the entire reference into per-line
    buckets. Lossless: every reference word appears in exactly one output
    line."""
    ref_words_raw, ref_words_norm, raw_idx_of_norm = build_ref_arrays(full_ref_text)
    n_ref = len(ref_words_norm)
    n_raw = len(ref_words_raw)

    ocr_files = sorted(
        ocr_dir.glob("*.txt"),
        key=lambda p: [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", p.name)],
    )

    if per_page_restart:
        total_raw_ocr = sum(len(p.read_text(encoding="utf-8").split()) for p in ocr_files)
        ref_per_raw_ocr = n_ref / total_raw_ocr if total_raw_ocr else 0.0
    else:
        ref_per_raw_ocr = 0.0

    # ---------- Stage 1a: load all pages, tokenise, predict expected_anchor ----------
    # We need every page's expected_anchor before we can compute the page's
    # actual anchor, because each page's DP window must end where the next
    # page's window begins. Without this, page N's DP can match OCR words
    # to reference words that actually belong to page N+1, pushing the
    # next page's first matched line forward and excluding the true anchor.
    pages_raw: list[dict] = []
    cumulative_raw_ocr = 0
    for ocr_path in ocr_files:
        raw_ocr = ocr_path.read_text(encoding="utf-8")
        raw_word_count = len(raw_ocr.split())
        ocr_lines_raw = raw_ocr.splitlines()
        ocr_words_norm: list[str] = []
        line_word_counts: list[int] = []
        for line in ocr_lines_raw:
            words = tokenise(line)
            line_word_counts.append(len(words))
            ocr_words_norm.extend(words)
        expected_anchor = int(cumulative_raw_ocr * ref_per_raw_ocr) if per_page_restart else 0
        pages_raw.append(
            {
                "name": ocr_path.stem,
                "raw_word_count": raw_word_count,
                "ocr_words_norm": ocr_words_norm,
                "line_word_counts": line_word_counts,
                "expected_anchor": expected_anchor,
            }
        )
        cumulative_raw_ocr += raw_word_count

    # ---------- Stage 1b: independently find each page's anchor ----------
    # Asymmetric search bounds: wide backward, tight forward. The expected
    # prediction (cumulative_raw_ocr * ratio) systematically overshoots
    # after runs of noisy-OCR pages, so widen backward; keep forward tight
    # so the n-gram finder cannot drift into later repeated phrases. We do
    # NOT impose a forward-monotonic floor here — that constraint would
    # bias every page's lower bound by the previous page's spurious
    # forward-overshoot. Monotonicity is restored explicitly in Stage 1c.
    anchors: list[int] = []
    scores: list[float] = []
    for page in pages_raw:
        if not page["ocr_words_norm"]:
            anchors.append(anchors[-1] if anchors else 0)
            scores.append(0.0)
            continue
        expected = page["expected_anchor"]
        search_from = max(0, expected - anchor_back_slack)
        search_to = min(n_ref, expected + anchor_fwd_slack)
        a, s = find_anchor(
            page["ocr_words_norm"],
            ref_words_norm,
            search_from=search_from,
            search_to=search_to,
            expected_idx=expected,
            probe_len=min(probe_len, len(page["ocr_words_norm"])),
            min_anchor_score=min_anchor_score,
            proximity_weight=proximity_weight,
        )
        anchors.append(a)
        scores.append(s)

    # ---------- Stage 1c: enforce anchor monotonicity ----------
    # The independent search per page can occasionally produce out-of-order
    # anchors when a page is very noisy. Forward-clamp so anchor_i >=
    # anchor_{i-1} (and at least anchor_{i-1} + 1 when the previous page
    # had OCR words, otherwise +0 to avoid stealing a position from the
    # next page).
    for i in range(1, len(anchors)):
        if anchors[i] <= anchors[i - 1]:
            anchors[i] = anchors[i - 1] + (1 if pages_raw[i - 1]["ocr_words_norm"] else 0)
            anchors[i] = min(anchors[i], n_ref - 1)

    # ---------- Stage 1d: per-page DP bounded by the next page's anchor ----------
    pages_meta: list[dict] = []
    flat_line_his: list[int | None] = []
    for i, page in enumerate(pages_raw):
        anchor_idx = anchors[i]
        anchor_score = scores[i]
        expected = page["expected_anchor"]
        if not page["ocr_words_norm"]:
            pages_meta.append(
                {
                    "name": page["name"],
                    "line_count": len(page["line_word_counts"]),
                    "anchor_idx": anchor_idx,
                    "anchor_score": 0.0,
                    "expected_anchor": expected,
                    "matched_lines": 0,
                    "total_lines": len(page["line_word_counts"]),
                    "total_words": 0,
                }
            )
            flat_line_his.extend([None] * len(page["line_word_counts"]))
            continue

        # DP window: anchor + enough room to cover all OCR words plus a
        # reference-side buffer. We deliberately do NOT cap at the next
        # page's anchor — if that anchor is too early, capping starves
        # this page's bottom-of-page OCR lines from finding their true
        # matches and produces a balloon at the top of the next page.
        # Letting windows overlap lets each page's DP match its real
        # content; cross-page overlap is resolved by Pass 2's
        # forward-clamp + the gap reassignment that follows.
        ref_end = min(
            n_ref,
            anchor_idx + len(page["ocr_words_norm"]) + ref_window_words,
        )
        ref_window_norm = ref_words_norm[anchor_idx:ref_end]

        pairs = align_word_sequences(page["ocr_words_norm"], ref_window_norm)
        page_line_his = _per_line_hi(page["line_word_counts"], pairs, anchor_idx)

        flat_line_his.extend(page_line_his)
        matched_lines = sum(1 for h in page_line_his if h is not None)
        pages_meta.append(
            {
                "name": page["name"],
                "line_count": len(page["line_word_counts"]),
                "anchor_idx": anchor_idx,
                "anchor_score": anchor_score,
                "expected_anchor": expected,
                "matched_lines": matched_lines,
                "total_lines": len(page["line_word_counts"]),
                "total_words": len(page["ocr_words_norm"]),
            }
        )

        print(
            f"✓ {page['name']}: anchor={anchor_idx} "
            f"(score={anchor_score:.3f}, expected≈{expected}), "
            f"matched_lines={matched_lines}/{len(page['line_word_counts'])}"
        )

    # ---------- Stage 1e: re-anchor pages with poor DP coverage ----------
    # Detection: matched_lines / total_lines < recheck_match_threshold.
    # Sweep experiments on the previously-failing pages (50, 58, 65) show
    # the underlying problem is always the same — the cumulative-OCR
    # prediction has drifted by 500-800 words from the true page start,
    # so the proximity penalty drags the chosen anchor away from a
    # genuinely-higher n-gram peak. Re-running find_anchor with a
    # smaller proximity_weight lets the n-gram peak win when it is
    # genuinely better. We only swap in the new result if both the raw
    # n-gram score and the DP coverage improve, so well-anchored pages
    # are never made worse.
    line_offsets: list[int] = []
    offset = 0
    for meta in pages_meta:
        line_offsets.append(offset)
        offset += meta["line_count"]

    for i, (page, meta) in enumerate(zip(pages_raw, pages_meta, strict=False)):
        total = meta["total_lines"]
        if not page["ocr_words_norm"] or total == 0:
            continue
        if meta["matched_lines"] / total >= recheck_match_threshold:
            continue
        expected = page["expected_anchor"]
        search_from = max(0, expected - anchor_back_slack)
        search_to = min(n_ref, expected + anchor_fwd_slack)
        new_anchor, new_raw = find_anchor(
            page["ocr_words_norm"],
            ref_words_norm,
            search_from=search_from,
            search_to=search_to,
            expected_idx=expected,
            probe_len=min(probe_len, len(page["ocr_words_norm"])),
            min_anchor_score=min_anchor_score,
            proximity_weight=recheck_proximity_weight,
        )
        if new_anchor == meta["anchor_idx"]:
            continue
        ref_end = min(n_ref, new_anchor + len(page["ocr_words_norm"]) + ref_window_words)
        new_window = ref_words_norm[new_anchor:ref_end]
        new_pairs = align_word_sequences(page["ocr_words_norm"], new_window)
        new_his = _per_line_hi(page["line_word_counts"], new_pairs, new_anchor)
        new_matched = sum(1 for h in new_his if h is not None)
        # Accept only if both signals improve: more matched lines AND a
        # higher raw n-gram score. Either alone is not enough — n-gram can
        # peak on coincidental phrase overlap, and matched_lines can
        # increase trivially by anchoring inside a denser region.
        if new_matched > meta["matched_lines"] and new_raw > meta["anchor_score"]:
            old_anchor = meta["anchor_idx"]
            old_matched = meta["matched_lines"]
            old_score = meta["anchor_score"]
            start = line_offsets[i]
            for k, h in enumerate(new_his):
                flat_line_his[start + k] = h
            meta["anchor_idx"] = new_anchor
            meta["anchor_score"] = new_raw
            meta["matched_lines"] = new_matched
            print(
                f"↻ {page['name']}: re-anchored {old_anchor}→{new_anchor}, "
                f"score {old_score:.3f}→{new_raw:.3f}, "
                f"matched_lines {old_matched}→{new_matched}/{total}"
            )

    # ---------- Pass 2 ----------
    norm_ranges = partition_reference_to_lines(flat_line_his, n_ref)

    # Push cross-page gaps backward. If page i's first non-empty line has
    # lo < page i's anchor, that means page i-1 didn't reach far enough
    # forward (its OCR missed bottom-of-page content) and the unmatched
    # gap got dumped into page i's first line. Reassign the gap to page
    # i-1's last non-empty line so the visual misalignment shows up at
    # the bottom of the previous page (where the OCR genuinely failed)
    # rather than as a balloon at the top of the next page (where the
    # OCR was fine).
    cursor = 0
    for i, page in enumerate(pages_meta):
        if i == 0 or page["line_count"] == 0:
            cursor += page["line_count"]
            continue
        # First non-empty range in current page
        page_first_idx = None
        for k in range(cursor, cursor + page["line_count"]):
            if norm_ranges[k] is not None:
                page_first_idx = k
                break
        if page_first_idx is None:
            cursor += page["line_count"]
            continue
        page_anchor = page["anchor_idx"]
        first_lo, first_hi = norm_ranges[page_first_idx]
        if first_lo < page_anchor and page_anchor <= first_hi:
            # Last non-empty range in any previous page
            prev_last_idx = None
            for k in range(cursor - 1, -1, -1):
                if norm_ranges[k] is not None:
                    prev_last_idx = k
                    break
            if prev_last_idx is not None:
                prev_lo, prev_hi = norm_ranges[prev_last_idx]
                # Only extend if it produces a valid range
                if prev_lo <= page_anchor - 1:
                    norm_ranges[prev_last_idx] = (prev_lo, page_anchor - 1)
                    norm_ranges[page_first_idx] = (page_anchor, first_hi)
        cursor += page["line_count"]

    # Convert each (norm_lo, norm_hi) into a (raw_lo, raw_hi) range. Each
    # raw token between two adjacent normalised tokens (pure punctuation that
    # normalised to empty) is folded into the line that owns the FOLLOWING
    # normalised token. Net effect: every raw token is in exactly one line.
    raw_ranges: list[tuple[int, int] | None] = []
    prev_raw_hi = -1
    last_nonempty_idx = -1
    for idx, rng in enumerate(norm_ranges):
        if rng is None:
            raw_ranges.append(None)
            continue
        norm_lo, norm_hi = rng
        raw_lo = prev_raw_hi + 1
        raw_hi = raw_idx_of_norm[norm_hi]
        if raw_hi < raw_lo:
            raw_ranges.append(None)
            continue
        raw_ranges.append((raw_lo, raw_hi))
        prev_raw_hi = raw_hi
        last_nonempty_idx = idx
    # Pin tail: any raw tokens after the last consumed raw_hi go to the last
    # non-empty range (covers trailing punctuation past the last norm token).
    if last_nonempty_idx >= 0 and prev_raw_hi < n_raw - 1:
        lo, _ = raw_ranges[last_nonempty_idx]
        raw_ranges[last_nonempty_idx] = (lo, n_raw - 1)

    aligned_doc: list[tuple[str, list[str]]] = []
    page_info: list[dict] = []
    cursor = 0
    for page in pages_meta:
        page_ranges = raw_ranges[cursor : cursor + page["line_count"]]
        aligned_lines = []
        for rng in page_ranges:
            if rng is None:
                aligned_lines.append("")
            else:
                lo, hi = rng
                aligned_lines.append(" ".join(ref_words_raw[lo : hi + 1]))
        aligned_doc.append((page["name"], aligned_lines))
        page_info.append({**page, "line_ranges": page_ranges})
        cursor += page["line_count"]

    return aligned_doc, page_info


# ──────────────────────────────────────────────
#  6.  Boundary enforcer — no-op shim for backwards compatibility
# ──────────────────────────────────────────────


def enforce_page_boundaries(
    aligned_doc: list[tuple[str, list[str]]],
    page_info: list[dict],
    full_ref_text: str,
    *,
    mode: str = "trim",
    verbose: bool = True,
) -> list[tuple[str, list[str]]]:
    """No-op now. The partition built by ``align_ocr_to_reference`` is
    monotonic by construction, so cross-page overlap is impossible."""
    if verbose:
        print("enforce_page_boundaries: no-op (partition is monotonic by design)")
    return aligned_doc


# ──────────────────────────────────────────────
#  7.  Save helpers
# ──────────────────────────────────────────────


def save_aligned(
    aligned_doc: list[tuple[str, list[str]]],
    out_path: Path,
    side_by_side: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for img_name, lines in aligned_doc:
            f.write(f"\n{'='*10} IMAGE: {img_name} {'='*10}\n")
            # 0-based per-page line number prefix; resets each page.
            for k, line in enumerate(lines, start=0):
                f.write(f"{k}: {line}\n")
    print(f"\n💾 Saved to: {out_path}")


def save_side_by_side(
    ocr_dir: Path,
    aligned_doc: list[tuple[str, list[str]]],
    out_path: Path,
    col_width: int = 55,
) -> None:
    ocr_map = {p.stem: p for p in ocr_dir.glob("*.txt")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        header = f"{'OCR':<{col_width}}  {'ALIGNED REF'}"
        f.write(header + "\n" + "─" * (col_width * 2 + 4) + "\n")
        for img_name, ref_lines in aligned_doc:
            f.write(f"\n{'='*10} {img_name} {'='*10}\n")
            ocr_lines = (
                ocr_map[img_name].read_text(encoding="utf-8").splitlines()
                if img_name in ocr_map
                else [""] * len(ref_lines)
            )
            # 0-based per-page line number prefix; resets each page.
            n = max(len(ocr_lines), len(ref_lines))
            for k in range(n):
                ocr_line = ocr_lines[k] if k < len(ocr_lines) else ""
                ref_line = ref_lines[k] if k < len(ref_lines) else ""
                ocr_trunc = ocr_line[:col_width].ljust(col_width)
                f.write(f"{k}: {ocr_trunc}  {ref_line}\n")
    print(f"📊 Side-by-side saved to: {out_path}")
