# OCR evaluation — med4k_stretchtok_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `med4k_stretch_tok` from `data/processed/transcription/med4k_stretchtok_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| med4k_stretch_tok | 299 | 0.0455 | 0.9545 | 0.2324 | 0.7676 | 0.0286 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
