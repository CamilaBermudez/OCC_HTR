# OCR evaluation — partition_val_80

Ground truth: `/private/tmp/claude-501/-Users-camilabermudezvalderrama-Documents-LMU-STATISTICS---DATA-SCIENCE-MASTER-SS2026-Thesis-OCC-HTR/87f8aa39-81a6-4b20-b16f-d6fc6f614c6e/scratchpad/gt_val` (79 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `medusa` from `data/processed/transcription/medusa_all_500_20260702`
- `finetune_400` from `data/processed/transcription/finetune_400_full_corpus`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 79 | 0.0365 | 0.9635 | 0.1272 | 0.8728 | 0.0263 | 0.1250 | 0 |
| medusa | 79 | 0.0457 | 0.9543 | 0.2760 | 0.7240 | 0.0286 | 0.2500 | 0 |
| finetune_400 | 79 | 0.0351 | 0.9649 | 0.1828 | 0.8172 | 0.0250 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
