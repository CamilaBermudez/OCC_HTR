# OCR evaluation — three_way_vs_batch5

Ground truth: `data/processed/annotated_samples/OCR/batch_5` (100 lines)

Models compared:
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `medusa` from `data/processed/transcription/medusa_all_500_20260702`
- `finetune_400` from `data/processed/transcription/finetune_400_on_batch5`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| catmus_baseline | 100 | 0.0489 | 0.9511 | 0.1676 | 0.8324 | 0.0282 | 0.1429 | 0 |
| medusa | 100 | 0.0907 | 0.9093 | 0.3423 | 0.6577 | 0.0494 | 0.3333 | 0 |
| finetune_400 | 100 | 0.0483 | 0.9517 | 0.2372 | 0.7628 | 0.0286 | 0.1833 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
