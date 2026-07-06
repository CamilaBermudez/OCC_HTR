# Sixth annotation batch (20260703_100000) — capital C/E targeted

100 line samples sampled with seed=48 across 53 pages.

Filter: OCR seed text contains at least one word-initial uppercase
'C' or 'E' (regex: `(?<![A-Za-z])[CE]`). These are the characters
the fine-tuned model still misses disproportionately — the
augmentation pipeline pushes them via glyphs/C_capitol and
glyphs/E_capitol stamps but real annotated coverage is thin.

Composition: 34 lines with a capital C, 67 lines with a capital E (some overlap).

Excludes the 500 already-annotated stems in the
corrected pool (batches 1-5, all merged). All .gt.txt files are
guaranteed non-empty.

## Files
- `<stem>.png` — line image
- `<stem>.gt.txt` — pre-filled with OCR; correct in place
- `_INDEX.csv` — page + stem listing, with per-line C/E flags

## Status — verified and merged on 2026-07-05

All 100 `<stem>.png` + `<stem>.gt.txt` pairs were hand-verified and moved
into `data/processed/annotated_samples/OCR/full_annotated/`, bringing the
corrected pool from 500 -> 600 lines. A copy of the batch also lives at
`data/processed/annotated_samples/OCR/batch_6/` for per-batch reporting.

This folder is kept (without the image/text pairs) so `_INDEX.csv`
remains the authoritative record of *which* 100 stems belong to this
batch — with the `has_capital_C` / `has_capital_E` flags — for later
slicing back into a C/E-only diagnostic subset.

## Post-merge corrections
- `22_f_017v_018_line_170`: originally sampled but the crop was a
  nearly-blank strip (OCR seed was empty; the sampler's non-empty
  filter didn't catch it because the annotator cleared the seed to
  match the blank image). Swapped for `72_f_067v_068_line_148`.
- `67_f_062v_063_line_101`: originally copied from the RAW
  `extraction_20260618_154440/` folder, which still had the
  uncorrected double-column crop. Re-copied from
  `filtered_images/20260618_160948/original/kept/`.
- Both are the reason the sampler was formalized under
  `src/data_preprocessing/sample_annotation_batch.py` — it now
  sources from `filtered_images/…/kept/` by default so future
  batches can't repeat these mistakes.
