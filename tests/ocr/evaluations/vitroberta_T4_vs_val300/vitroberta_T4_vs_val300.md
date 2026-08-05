# OCR evaluation — vitroberta_T4_vs_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vitroberta_T4_1font` from `data/processed/transcription/vitroberta_T4_1font_vs_val300`
- `vitroberta_T4_mf` from `data/processed/transcription/vitroberta_T4_mf_vs_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vitroberta_T4_1font | 299 | 0.1217 | 0.8783 | 0.3709 | 0.6291 | 0.1026 | 0.3750 | 0 |
| vitroberta_T4_mf | 299 | 0.0926 | 0.9074 | 0.3243 | 0.6757 | 0.0811 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
