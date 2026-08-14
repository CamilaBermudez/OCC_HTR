# OCR evaluation — ts6k_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `ts6k` from `data/processed/transcription/ts6k_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| ts6k | 299 | 0.0525 | 0.9475 | 0.2557 | 0.7443 | 0.0400 | 0.2500 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
