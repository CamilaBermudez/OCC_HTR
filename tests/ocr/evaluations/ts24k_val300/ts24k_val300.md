# OCR evaluation — ts24k_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `ts24k` from `data/processed/transcription/ts24k_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| ts24k | 299 | 0.0541 | 0.9459 | 0.2601 | 0.7399 | 0.0488 | 0.2500 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
