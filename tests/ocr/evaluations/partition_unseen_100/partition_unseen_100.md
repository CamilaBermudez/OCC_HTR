# OCR evaluation — partition_unseen_100

Ground truth: `/private/tmp/claude-501/-Users-camilabermudezvalderrama-Documents-LMU-STATISTICS---DATA-SCIENCE-MASTER-SS2026-Thesis-OCC-HTR/87f8aa39-81a6-4b20-b16f-d6fc6f614c6e/scratchpad/gt_unseen` (99 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `medusa` from `data/processed/transcription/medusa_all_500_20260702`
- `finetune_400` from `data/processed/transcription/finetune_400_full_corpus`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 99 | 0.0436 | 0.9564 | 0.1580 | 0.8420 | 0.0278 | 0.1429 | 0 |
| medusa | 99 | 0.0434 | 0.9566 | 0.2917 | 0.7083 | 0.0476 | 0.2857 | 0 |
| finetune_400 | 99 | 0.0376 | 0.9624 | 0.1940 | 0.8060 | 0.0263 | 0.1429 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
