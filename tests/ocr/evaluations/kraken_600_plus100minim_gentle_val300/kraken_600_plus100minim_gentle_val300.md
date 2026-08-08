# OCR evaluation — kraken_600_plus100minim_gentle_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_600_plus100minim_gentle` from `data/processed/transcription/kraken_600_plus100minim_gentle_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_600_plus100minim_gentle | 299 | 0.0318 | 0.9682 | 0.1925 | 0.8075 | 0.0256 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
