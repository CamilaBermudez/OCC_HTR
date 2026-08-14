# OCR evaluation — realonly_stretch_bpe_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `realonly_stretch_bpe` from `data/processed/transcription/realonly_stretch_bpe_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| realonly_stretch_bpe | 299 | 0.0569 | 0.9431 | 0.2839 | 0.7161 | 0.0488 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
