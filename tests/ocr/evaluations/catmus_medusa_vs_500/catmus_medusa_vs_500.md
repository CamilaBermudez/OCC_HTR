# OCR evaluation — catmus_medusa_vs_500

Ground truth: `data/processed/annotated_samples/OCR/500_samples` (500 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `medusa` from `data/processed/transcription/medusa_all_500_20260702`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 500 | 0.0406 | 0.9594 | 0.1433 | 0.8567 | 0.0270 | 0.1429 | 0 |
| medusa | 500 | 0.1578 | 0.8422 | 0.4008 | 0.5992 | 0.0500 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
