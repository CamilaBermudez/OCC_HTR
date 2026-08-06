# OCR evaluation — kraken_600real_8020_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_600real_8020` from `data/processed/transcription/kraken_600real_8020_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_600real_8020 | 299 | 0.0290 | 0.9710 | 0.1799 | 0.8201 | 0.0250 | 0.1250 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
