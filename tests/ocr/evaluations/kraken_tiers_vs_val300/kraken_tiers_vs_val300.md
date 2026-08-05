# OCR evaluation — kraken_tiers_vs_val300

Ground truth: `data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `kraken_T1_1font` from `data/processed/transcription/kraken_T1_1font_vs_val300`
- `kraken_T1_mf` from `data/processed/transcription/kraken_T1_mf_vs_val300`
- `kraken_T2_1font` from `data/processed/transcription/kraken_T2_1font_vs_val300`
- `kraken_T2_mf` from `data/processed/transcription/kraken_T2_mf_vs_val300`
- `kraken_T3_1font` from `data/processed/transcription/kraken_T3_1font_vs_val300`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| kraken_T1_1font | 299 | 0.1219 | 0.8781 | 0.5527 | 0.4473 | 0.1000 | 0.5000 | 0 |
| kraken_T1_mf | 299 | 0.1796 | 0.8204 | 0.7258 | 0.2742 | 0.1500 | 0.7143 | 0 |
| kraken_T2_1font | 299 | 0.2258 | 0.7742 | 0.7968 | 0.2032 | 0.2000 | 0.7778 | 0 |
| kraken_T2_mf | 299 | 0.3345 | 0.6655 | 1.0729 | -0.0729 | 0.3125 | 1.1111 | 0 |
| kraken_T3_1font | 299 | 0.3968 | 0.6032 | 1.0024 | -0.0024 | 0.3846 | 1.0000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
