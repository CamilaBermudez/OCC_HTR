# Permanent validation set (20260706_163211)

300 line samples sampled with seed=100 across 70 pages. These are
intended to become the **permanent held-out benchmark** for every
future model comparison — training MUST never touch these stems.

Line PNGs sourced from
`data/processed/filtered_images/20260618_160948/original/kept` — the
filtered / manually-corrected line-crop folder, NOT the raw extraction
folder. This ensures annotators see the same crops the OCR pipeline
runs on (post ink-bleed filter + double-column corrections).

Excludes the 600 stems already in the training pool
(`data/processed/annotated_samples/OCR/full_annotated/`, batches 1-6).
All `.gt.txt` files are guaranteed non-empty at sampling time.

## Files
- `<stem>.png` — line image
- `<stem>.gt.txt` — pre-filled with OCR seed; correct in place
- `_INDEX.csv` — page + stem listing

## Status — annotated and merged on 2026-07-07

All 300 `<stem>.png` + `<stem>.gt.txt` pairs were hand-verified and
moved into `data/processed/annotated_samples/OCR/validation/` — the
sibling of `full_annotated/` and now the permanent held-out benchmark
for every model comparison.

Post-merge notes:
- 1 line (`43_f_038v_039_line_22.gt.txt`) has an empty ground truth —
  the annotator cleared the OCR seed because the underlying crop is a
  nearly-blank strip. The evaluation script (`scripts/ocr/run_evaluate_ocr.py`)
  auto-skips empty references and increments a "skipped" counter, so
  effective val size is 299 lines for OCR-metric purposes.

The sampler's default `SAMPLE_EXCLUDES` list already points at both
`full_annotated/` and `validation/`, so every future
`make sample_annotation_batch` will automatically exclude these 300
stems from the eligible pool — impossible to accidentally sample a
validation line into a training batch.

This folder is kept (without the image/text pairs) so `_INDEX.csv`
remains the authoritative record of *which* 300 stems belong to this
benchmark.

## Why a permanent val set (thesis-methodology note)

Prior comparisons used whichever recent annotation batch happened to
be untouched by each model — a moving target, and it made head-to-head
comparisons awkward once the same batch had entered training for
subsequent runs. A fixed 300-line held-out set that never enters
training gives every model the same apples-to-apples score. Once
annotated, this set becomes the default `--gt-dir` for
`scripts/ocr/run_evaluate_ocr.py` in all thesis reporting.
