# OCR evaluation — catmus_lexcorr_cmp

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `catmus_lexcorr88` from `data/processed/transcription/catmus_lexcorr88_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 299 | 0.0397 | 0.9603 | 0.1488 | 0.8512 | 0.0278 | 0.1429 | 0 |
| catmus_lexcorr88 | 299 | 0.0511 | 0.9489 | 0.1949 | 0.8051 | 0.0465 | 0.1667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
