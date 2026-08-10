# OCR evaluation — vitroberta_T1_stretch_bpe_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `T1` from `data/processed/transcription/vitroberta_T1_stretch_bpe_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| T1 | 299 | 0.0443 | 0.9557 | 0.2299 | 0.7701 | 0.0278 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
