# OCR evaluation — med4k_stretch_retrain_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `med4k_stretch_retrain` from `data/processed/transcription/med4k_stretch_retrain_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| med4k_stretch_retrain | 299 | 0.0473 | 0.9527 | 0.2382 | 0.7618 | 0.0286 | 0.1667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
