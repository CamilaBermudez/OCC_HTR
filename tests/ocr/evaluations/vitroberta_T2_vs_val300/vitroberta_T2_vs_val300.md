# OCR evaluation — vitroberta_T2_vs_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vitroberta_T2_1font` from `data/processed/transcription/vitroberta_T2_1font_vs_val300`
- `vitroberta_T2_mf` from `data/processed/transcription/vitroberta_T2_mf_vs_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vitroberta_T2_1font | 299 | 0.0729 | 0.9271 | 0.2883 | 0.7117 | 0.0556 | 0.2500 | 0 |
| vitroberta_T2_mf | 299 | 0.0702 | 0.9298 | 0.2800 | 0.7200 | 0.0541 | 0.2500 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
