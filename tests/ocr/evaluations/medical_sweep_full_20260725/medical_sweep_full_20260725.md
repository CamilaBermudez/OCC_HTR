# OCR evaluation — medical_sweep_full_20260725

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `vitroberta_medical_500` from `data/processed/transcription/vitroberta_medical_500_val300_20260724`
- `vitroberta_medical_1000` from `data/processed/transcription/vitroberta_medical_val300_20260722`
- `vitroberta_medical_2000` from `data/processed/transcription/vitroberta_medical_2000_val300_20260724`
- `vitroberta_medical_4000` from `data/processed/transcription/vitroberta_medical_4000_val300_20260724`
- `swinbert_medical_500` from `data/processed/transcription/swinbert_medical_500_val300_20260724`
- `swinbert_medical_1000` from `data/processed/transcription/swinbert_ss_medical_val300_20260722`
- `swinbert_medical_2000` from `data/processed/transcription/swinbert_medical_2000_val300_20260724`
- `swinbert_medical_4000` from `data/processed/transcription/swinbert_medical_4000_val300_20260724`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| vitroberta_medical_500 | 299 | 0.0619 | 0.9381 | 0.2732 | 0.7268 | 0.0513 | 0.2500 | 0 |
| vitroberta_medical_1000 | 299 | 0.0611 | 0.9389 | 0.2654 | 0.7346 | 0.0513 | 0.2500 | 0 |
| vitroberta_medical_2000 | 299 | 0.0555 | 0.9445 | 0.2713 | 0.7287 | 0.0500 | 0.2500 | 0 |
| vitroberta_medical_4000 | 299 | 0.0513 | 0.9487 | 0.2494 | 0.7506 | 0.0417 | 0.2000 | 0 |
| swinbert_medical_500 | 299 | 0.7578 | 0.2422 | 1.0574 | -0.0574 | 0.7561 | 1.0000 | 0 |
| swinbert_medical_1000 | 299 | 0.8762 | 0.1238 | 1.1497 | -0.1497 | 0.8718 | 1.1429 | 0 |
| swinbert_medical_2000 | 299 | 0.7774 | 0.2226 | 1.0190 | -0.0190 | 0.7750 | 1.0000 | 0 |
| swinbert_medical_4000 | 299 | 0.7897 | 0.2103 | 1.0559 | -0.0559 | 0.7941 | 1.0000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
