# OCR evaluation — seven_way_vs_validation_300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `medusa_cleaned` from `data/processed/transcription/medusa_validation_300_20260710_clean`
- `kraken_400_real` from `data/processed/transcription/finetune_20260629_235819_on_validation_300`
- `kraken_500_real` from `data/processed/transcription/finetune_20260701_233056_on_validation_300`
- `kraken_600_real` from `data/processed/transcription/finetune_20260705_070741_on_validation_300`
- `kraken_600_real_medical` from `data/processed/transcription/finetune_20260706_151856_on_validation_300`
- `trocr_swin_bert_aug` from `data/processed/transcription/trocr_20260710_142341_on_validation_300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 299 | 0.0387 | 0.9613 | 0.1434 | 0.8566 | 0.0278 | 0.1250 | 0 |
| medusa_cleaned | 299 | 0.0490 | 0.9510 | 0.3106 | 0.6894 | 0.0435 | 0.2857 | 0 |
| kraken_400_real | 299 | 0.0420 | 0.9580 | 0.2358 | 0.7642 | 0.0286 | 0.2000 | 0 |
| kraken_500_real | 299 | 0.0390 | 0.9610 | 0.2188 | 0.7812 | 0.0278 | 0.1667 | 0 |
| kraken_600_real | 299 | 0.0380 | 0.9620 | 0.2144 | 0.7856 | 0.0278 | 0.1667 | 0 |
| kraken_600_real_medical | 299 | 0.0407 | 0.9593 | 0.2275 | 0.7725 | 0.0278 | 0.1667 | 0 |
| trocr_swin_bert_aug | 299 | 0.7101 | 0.2899 | 0.9611 | 0.0389 | 0.7209 | 1.0000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
