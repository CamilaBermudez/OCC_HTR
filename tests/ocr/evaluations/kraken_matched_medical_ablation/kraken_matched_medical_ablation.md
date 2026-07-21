# OCR evaluation — kraken_matched_medical_ablation

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_600_3000reren` from `data/processed/transcription/finetune_20260718_193601_on_validation_300`
- `kraken_600_3000reren_1000medical` from `data/processed/transcription/finetune_20260719_085411_on_validation_300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_600_3000reren | 299 | 0.0906 | 0.9094 | 0.4225 | 0.5775 | 0.0714 | 0.3750 | 0 |
| kraken_600_3000reren_1000medical | 299 | 0.1338 | 0.8662 | 0.5542 | 0.4458 | 0.1111 | 0.5000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
