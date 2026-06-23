# Real-manuscript validation sample (20260622_220845)

100 line crops sampled from `data/processed/extracted_lines/extraction_20260618_154440`,
stratified across 71 pages (seed=42).

## Files
- `<stem>.png` — line image (real manuscript crop)
- `<stem>.gt.txt` — **ground truth** (pre-filled with OCR prediction; correct in place)
- `_INDEX.csv` — page + stem listing for reference

## How to correct
1. Open each `.png` and its sibling `.gt.txt` side by side.
2. Edit the `.gt.txt` to match what you see in the image.
3. Save. The folder will then be usable as `--real-val-folder` for
   fine-tuning evaluation.

Convention: kraken/ketos uses `<image>.gt.txt` for ground truth.
