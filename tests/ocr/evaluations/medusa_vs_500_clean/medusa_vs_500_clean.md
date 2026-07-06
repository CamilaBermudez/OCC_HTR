# OCR evaluation — medusa_vs_500_clean

Ground truth: `tests/ocr/real_corrected_20260625` (500 lines)

Models compared:
- `medusa` from `data/processed/transcription/medusa_all_500_20260702`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| medusa | 500 | 0.1578 | 0.8422 | 0.4008 | 0.5992 | 0.0500 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
