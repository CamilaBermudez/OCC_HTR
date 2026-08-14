# OCR evaluation — ts12k_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `ts12k` from `data/processed/transcription/ts12k_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| ts12k | 299 | 0.0506 | 0.9494 | 0.2479 | 0.7521 | 0.0345 | 0.2222 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
