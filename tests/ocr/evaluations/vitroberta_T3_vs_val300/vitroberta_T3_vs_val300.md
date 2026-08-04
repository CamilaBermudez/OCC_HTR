# OCR evaluation — vitroberta_T3_vs_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vitroberta_T3_1font` from `data/processed/transcription/vitroberta_T3_1font_vs_val300`
- `vitroberta_T3_mf` from `data/processed/transcription/vitroberta_T3_mf_vs_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vitroberta_T3_1font | 299 | 0.0815 | 0.9185 | 0.3029 | 0.6971 | 0.0667 | 0.2857 | 0 |
| vitroberta_T3_mf | 299 | 0.0902 | 0.9098 | 0.3209 | 0.6791 | 0.0750 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
