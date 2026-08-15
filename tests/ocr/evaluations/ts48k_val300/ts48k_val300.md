# OCR evaluation — ts48k_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `ts48k` from `data/processed/transcription/ts48k_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| ts48k | 299 | 0.0608 | 0.9392 | 0.2786 | 0.7214 | 0.0541 | 0.2500 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
