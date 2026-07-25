# OCR evaluation — grid_medical1000_6arch_20260725

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `swinbert_medical1000` from `data/processed/transcription/swinbert_ss_medical_val300_20260722`
- `vitroberta_medical1000` from `data/processed/transcription/vitroberta_medical_val300_20260722`
- `swinroberta_medical1000` from `data/processed/transcription/swinroberta_B_val300_20260724`
- `vitbert_medical1000` from `data/processed/transcription/vitbert_B_val300_20260724`
- `swingpt2_medical1000` from `data/processed/transcription/swingpt2_B_v2_val300_20260724`
- `vitgpt2_medical1000` from `data/processed/transcription/vitgpt2_B_v2_val300_20260724`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| swinbert_medical1000 | 299 | 0.8762 | 0.1238 | 1.1497 | -0.1497 | 0.8718 | 1.1429 | 0 |
| vitroberta_medical1000 | 299 | 0.0611 | 0.9389 | 0.2654 | 0.7346 | 0.0513 | 0.2500 | 0 |
| swinroberta_medical1000 | 299 | 0.7264 | 0.2736 | 1.0379 | -0.0379 | 0.7297 | 1.0000 | 0 |
| vitbert_medical1000 | 299 | 0.8015 | 0.1985 | 1.0822 | -0.0822 | 0.8000 | 1.0000 | 0 |
| swingpt2_medical1000 | 299 | 0.7971 | 0.2029 | 1.0404 | -0.0404 | 0.8000 | 1.0000 | 0 |
| vitgpt2_medical1000 | 299 | 0.8187 | 0.1813 | 1.0841 | -0.0841 | 0.8235 | 1.0000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
