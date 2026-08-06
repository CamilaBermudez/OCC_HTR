# OCR evaluation — kraken_600_plus100minim_jit_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_600_plus100minim_jit` from `data/processed/transcription/kraken_600_plus100minim_jit_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_600_plus100minim_jit | 299 | 0.0359 | 0.9641 | 0.2071 | 0.7929 | 0.0263 | 0.1667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
