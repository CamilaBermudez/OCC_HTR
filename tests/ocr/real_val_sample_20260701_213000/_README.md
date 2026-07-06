# Fifth annotation batch (20260701_213000)

100 line samples sampled with seed=47 across 52 pages.
Excludes the 400 already-annotated stems in the corrected pool
(which itself contains the 4 prior batches). All .gt.txt files are
guaranteed non-empty.

## Files
- `<stem>.png` — line image
- `<stem>.gt.txt` — pre-filled with OCR; correct in place
- `_INDEX.csv` — page + stem listing for the 100 sampled lines

## Status — verified and merged on 2026-07-01

All 100 `<stem>.png` + `<stem>.gt.txt` pairs were hand-verified and moved
into `data/processed/annotated_samples/OCR/full_annotated/`, bringing the corrected pool
from 400 -> 500 lines.

This folder is kept (without the image/text pairs) so `_INDEX.csv`
remains the authoritative record of *which* 100 stems belong to this
batch. Use that list to slice the corrected pool back into the batch-5
subset if needed.
