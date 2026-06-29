# Fourth annotation batch (20260629_175815)

100 line samples sampled with seed=46 across 71 pages.
Excludes the 301 already-annotated stems in the 3 prior batches
and the verified pool. All .gt.txt files are guaranteed non-empty.

## Files
- `<stem>.png` — line image
- `<stem>.gt.txt` — pre-filled with OCR; correct in place
- `_INDEX.csv` — page + stem listing for the 100 sampled lines

## Status — verified and merged on 2026-06-29

All 100 `<stem>.png` + `<stem>.gt.txt` pairs were hand-verified and moved
into `tests/ocr/real_corrected_20260625/`, bringing the corrected pool
from 300 -> 400 lines.

This folder is kept (without the image/text pairs) so `_INDEX.csv`
remains the authoritative record of *which* 100 stems belong to this
batch. Use that list to slice the corrected pool back into the
held-out-4 subset if needed — e.g. when reporting per-batch metrics or
re-running the 3-model benchmark (catmus / fine-tuned / Medusa) on the
exact same 100 lines.
