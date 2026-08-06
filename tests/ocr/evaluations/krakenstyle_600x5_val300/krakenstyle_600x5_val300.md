# OCR evaluation — krakenstyle_600x5_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `krakenstyle_600x5` from `data/processed/transcription/krakenstyle_600x5_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| krakenstyle_600x5 | 299 | 0.0766 | 0.9234 | 0.3140 | 0.6860 | 0.0556 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
