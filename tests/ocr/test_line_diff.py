"""Regression tests for the diff classifier (src/ocr/line_diff.py).

Covers the 2026-08-16 fixes (spec §6.7):
  - word-boundary SPACING shifts ("eley sa" <-> "e leysa") are detected on the raw
    diff spans and suppressed, instead of fragmenting into false substitutions;
  - the subsequence ABBREVIATION heuristic requires a multi-word expansion (a space
    in the base), so a dropped-letter misread ("meg" <- "mieg") stays a substitution
    while a real contraction ("del" <- "de lo") stays an abbreviation.

No pytest dependency: plain asserts + a __main__ runner (also pytest-discoverable).

    PROJECT_ROOT=. uv run python tests/ocr/test_line_diff.py
"""

from __future__ import annotations

from src.ocr import line_diff as L


def _types(base: str, ocr: str) -> list[tuple[str, str]]:
    """(type, 'ocr->base') for each emitted diff of scholarly `base` vs `ocr`."""
    return [
        (d.type, f"{d.ocr_text}->{d.base_text}")
        for d in L._diff_core(base, ocr, list(range(len(ocr))))
    ]


def test_abbrev_requires_multiword_expansion():
    # real scribal contractions: OCR function word is a subsequence of a MULTI-word base
    assert L.classify_region("de lo", "del") == "abbreviation"
    assert L.classify_region("a lo", "al") == "abbreviation"
    assert L.classify_region("de los", "dels") == "abbreviation"
    # dropped-letter misread of a SINGLE word must NOT be called an abbreviation
    assert L.classify_region("mieg", "meg") == "substitution"
    assert L.classify_region("aquel", "aqel") == "substitution"


def test_brevigraph_still_abbreviation():
    # a non-ASCII scribal mark is an abbreviation regardless of the expansion shape
    assert L.classify_region("com", "cõ") == "abbreviation"


def test_word_boundary_spacing_suppressed():
    # identical modulo whitespace -> pure spacing shift -> no diffs
    for base, ocr in [
        ("e leysa", "eley sa"),
        ("un apostema", "una postema"),
        ("la gremas", "lagremas"),
        ("E si", "Esi"),
        ("de lu", "delu"),
    ]:
        assert _types(base, ocr) == [], (base, ocr, _types(base, ocr))


def test_real_diffs_preserved():
    # a genuine misread stays a substitution (whole word shown)
    assert _types("majors", "marors") == [("substitution", "marors->majors")]
    # spacing shift next to a real misread: only the misread survives
    assert _types("foo bar e leysa", "foo bXr eley sa") == [("substitution", "bXr->bar")]
    # a genuine reordering is NOT a spacing shift (despace differs) -> not suppressed
    assert _types("ab cd", "cd ab") != []


def test_whole_word_add_delete():
    # (span may grow to whole words per _expand; assert type + content, not exact span)
    add = _types("hello", "hello mon")
    assert len(add) == 1 and add[0][0] == "addition" and "mon" in add[0][1]
    dele = _types("hello mon", "hello")
    assert len(dele) == 1 and dele[0][0] == "deletion" and "mon" in dele[0][1]


def test_word_align_viewer_engine():
    # The viewer uses the word-level engine (word_align.diff_page_banded), which reuses
    # classify_region. Verify the same two fixes end-to-end there.
    from src.ocr.word_align import diff_page_banded

    def wa(sch: str, ocr: str) -> list[tuple[str, str]]:
        return [
            (d.type, f"{d.ocr_text}->{d.base_text}")
            for d in diff_page_banded([sch], [(0, ocr)], {0: 1})
        ]

    # 2:2 word-boundary shift -> ONE spacing chip, not two false substitutions
    assert wa("e leysa", "eley sa") == [("spacing", "eley sa->e leysa")]
    assert wa("la gremas", "lagremas") == [("spacing", "lagremas->la gremas")]
    # dropped-letter misread stays a substitution (not a false abbreviation)
    assert wa("mieg", "meg") == [("substitution", "meg->mieg")]
    # real contraction still an abbreviation; genuine misread preserved
    assert wa("de lo", "del") == [("abbreviation", "del->de lo")]
    assert wa("majors", "marors") == [("substitution", "marors->majors")]


def _run() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
