# OCR evaluation — med4k_reruns_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `med4k_pad` from `data/processed/transcription/med4k_pad_val300`
- `med4k_pad_tok` from `data/processed/transcription/med4k_padtok_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| med4k_pad | 299 | 0.0752 | 0.9248 | 0.2961 | 0.7039 | 0.0571 | 0.2500 | 0 |
| med4k_pad_tok | 299 | 0.0747 | 0.9253 | 0.3058 | 0.6942 | 0.0588 | 0.2857 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
