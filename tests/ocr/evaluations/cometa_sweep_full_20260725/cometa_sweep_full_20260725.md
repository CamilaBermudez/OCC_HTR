# OCR evaluation — cometa_sweep_full_20260725

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vitroberta_cometa_500` from `data/processed/transcription/vitroberta_cometa_500_val300_20260724`
- `vitroberta_cometa_1000` from `data/processed/transcription/vitroberta_matched_cometa_val300_20260722`
- `vitroberta_cometa_2000` from `data/processed/transcription/vitroberta_cometa_2000_val300_20260724`
- `vitroberta_cometa_4000` from `data/processed/transcription/vitroberta_cometa_4000_val300_20260724`
- `swinbert_cometa_500` from `data/processed/transcription/swinbert_cometa_500_val300_20260724`
- `swinbert_cometa_1000` from `data/processed/transcription/swinbert_ss_matched_cometa_val300_20260722`
- `swinbert_cometa_2000` from `data/processed/transcription/swinbert_cometa_2000_val300_20260724`
- `swinbert_cometa_4000` from `data/processed/transcription/swinbert_cometa_4000_val300_20260724`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vitroberta_cometa_500 | 299 | 0.0642 | 0.9358 | 0.2902 | 0.7098 | 0.0526 | 0.2857 | 0 |
| vitroberta_cometa_1000 | 299 | 0.0655 | 0.9345 | 0.2790 | 0.7210 | 0.0526 | 0.2500 | 0 |
| vitroberta_cometa_2000 | 299 | 0.0597 | 0.9403 | 0.2679 | 0.7321 | 0.0500 | 0.2500 | 0 |
| vitroberta_cometa_4000 | 299 | 0.0562 | 0.9438 | 0.2635 | 0.7365 | 0.0465 | 0.2500 | 0 |
| swinbert_cometa_500 | 299 | 0.7735 | 0.2265 | 0.9874 | 0.0126 | 0.7778 | 1.0000 | 0 |
| swinbert_cometa_1000 | 299 | 0.8048 | 0.1952 | 1.0719 | -0.0719 | 0.8056 | 1.0000 | 0 |
| swinbert_cometa_2000 | 299 | 0.8058 | 0.1942 | 1.0710 | -0.0710 | 0.8049 | 1.0000 | 0 |
| swinbert_cometa_4000 | 299 | 0.8175 | 0.1825 | 0.9995 | 0.0005 | 0.8205 | 1.0000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
