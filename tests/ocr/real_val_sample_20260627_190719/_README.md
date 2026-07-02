# Third annotation batch (20260627_190719)

100 additional line crops sampled with seed=45 across 71
pages, NOT overlapping with the 201 already-annotated stems in:
  - tests/ocr/real_val_sample_20260622_220845
  - tests/ocr/real_val_sample_20260626_114534
  - data/processed/annotated_samples/OCR/500_samples

## Workflow
- `<stem>.png` — line image (real manuscript crop)
- `<stem>.gt.txt` — pre-filled with OCR prediction; correct in place
- `_INDEX.csv` — page + stem listing

After review, copy verified pairs into `data/processed/annotated_samples/OCR/500_samples/`.
The finetune script's fractions auto-scale as the pool grows.
