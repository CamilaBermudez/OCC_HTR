# OCR evaluation — kraken_oldpool_ablation_eval_20260726

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_OLDpool_leakfixed` from `data/processed/transcription/kraken_oldpool_ablation_val300_20260726`
- `kraken_NEWpool_leakfixed` from `data/processed/transcription/kraken_matched_nomedical_leakfixed_val300_20260722`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_OLDpool_leakfixed | 299 | 0.7971 | 0.2029 | 1.9120 | -0.9120 | 0.7941 | 1.8571 | 0 |
| kraken_NEWpool_leakfixed | 299 | 0.0982 | 0.9018 | 0.4439 | 0.5561 | 0.0811 | 0.4286 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
