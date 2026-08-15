# OCR evaluation — mixedfix_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `mixedfix` from `data/processed/transcription/mixedfix_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| mixedfix | 299 | 0.0451 | 0.9549 | 0.2280 | 0.7720 | 0.0286 | 0.1667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
