# OCR evaluation — med4k_stretch_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `med4k_stretch_orig` from `data/processed/transcription/med4k_stretch_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| med4k_stretch_orig | 299 | 0.0513 | 0.9487 | 0.2494 | 0.7506 | 0.0417 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
