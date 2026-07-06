# Real-manuscript validation sample — extra batch (20260626_114534)

99 additional line crops sampled with seed=43 across 71
pages, NOT overlapping with the 101 already-annotated stems in:
  - tests/ocr/real_val_sample_20260622_220845/
  - data/processed/annotated_samples/OCR/full_annotated/

## Files
- `<stem>.png` — line image (real manuscript crop)
- `<stem>.gt.txt` — **ground truth** (pre-filled with OCR prediction; correct in place)
- `_INDEX.csv` — page + stem listing for reference

## How to integrate after correcting
Once you've verified the .gt.txt against the .png, copy the verified pairs into
`data/processed/annotated_samples/OCR/full_annotated/` (or a new dated `real_corrected_<ts>/`).
The finetune script uses fractions of whatever's in `FINETUNE_REAL_FOLDER` so
the new samples are picked up automatically as soon as they land there.
