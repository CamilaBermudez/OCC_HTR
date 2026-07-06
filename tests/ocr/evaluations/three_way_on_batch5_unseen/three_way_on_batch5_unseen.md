# OCR evaluation — three_way_on_batch5_unseen

Ground truth: `data/processed/transcription/_stage_batch5_gt` (100 lines)

Models compared:
- `finetune_run7` from `data/processed/transcription/finetune_run7_on_batch5`
- `catmus_baseline` from `data/processed/transcription/ocr_kept_20260622_120413`
- `medusa` from `data/processed/transcription/medusa_all_500_20260702`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| finetune_run7 | 100 | 0.0524 | 0.9476 | 0.2159 | 0.7841 | 0.0290 | 0.1667 | 0 |
| catmus_baseline | 100 | 0.0489 | 0.9511 | 0.1676 | 0.8324 | 0.0282 | 0.1429 | 0 |
| medusa | 100 | 0.0907 | 0.9093 | 0.3423 | 0.6577 | 0.0494 | 0.3333 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
