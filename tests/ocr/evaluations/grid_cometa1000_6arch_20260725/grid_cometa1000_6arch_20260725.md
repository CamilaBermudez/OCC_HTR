# OCR evaluation — grid_cometa1000_6arch_20260725

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `swinbert_cometa1000` from `data/processed/transcription/swinbert_ss_matched_cometa_val300_20260722`
- `vitroberta_cometa1000` from `data/processed/transcription/vitroberta_matched_cometa_val300_20260722`
- `swinroberta_cometa1000` from `data/processed/transcription/swinroberta_A_val300_20260724`
- `vitbert_cometa1000` from `data/processed/transcription/vitbert_A_val300_20260724`
- `swingpt2_cometa1000` from `data/processed/transcription/swingpt2_A_v2_val300_20260724`
- `vitgpt2_cometa1000` from `data/processed/transcription/vitgpt2_A_v2_val300_20260724`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| swinbert_cometa1000 | 299 | 0.8048 | 0.1952 | 1.0719 | -0.0719 | 0.8056 | 1.0000 | 0 |
| vitroberta_cometa1000 | 299 | 0.0655 | 0.9345 | 0.2790 | 0.7210 | 0.0526 | 0.2500 | 0 |
| swinroberta_cometa1000 | 299 | 0.7190 | 0.2810 | 1.0331 | -0.0331 | 0.7179 | 1.0000 | 0 |
| vitbert_cometa1000 | 299 | 0.7942 | 0.2058 | 1.0710 | -0.0710 | 0.8000 | 1.0000 | 0 |
| swingpt2_cometa1000 | 299 | 0.7414 | 0.2586 | 0.9903 | 0.0097 | 0.7500 | 1.0000 | 0 |
| vitgpt2_cometa1000 | 299 | 0.7946 | 0.2054 | 1.1473 | -0.1473 | 0.8000 | 1.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
