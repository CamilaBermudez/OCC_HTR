# OCR evaluation — kraken_T1_1font_vs_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_T1_1font` from `data/processed/transcription/kraken_T1_1font_vs_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_T1_1font | 299 | 0.1219 | 0.8781 | 0.5527 | 0.4473 | 0.1000 | 0.5000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
