# OCR evaluation — catmus_vs_600_rename_check

Ground truth: `data/processed/annotated_samples/OCR/full_annotated` (600 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 600 | 0.0416 | 0.9584 | 0.1522 | 0.8478 | 0.0278 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
