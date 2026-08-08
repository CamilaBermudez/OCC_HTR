# OCR evaluation — vitroberta_med4k_gentle_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `gentle` from `data/processed/transcription/vitroberta_med4k_gentle_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| gentle | 299 | 0.0470 | 0.9530 | 0.2372 | 0.7628 | 0.0294 | 0.2222 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
