# OCR evaluation — pansier_ns

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `leader_0.9549` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/leader_0.9549_pred/leader_0.9549`
- `pansierAns_1000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierAns_1000_pred/pansierAns_1000`
- `pansierAns_2000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierAns_2000_pred/pansierAns_2000`
- `pansierAns_300` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierAns_300_pred/pansierAns_300`
- `pansierAns_600` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierAns_600_pred/pansierAns_600`
- `pansierBns_1000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierBns_1000_pred/pansierBns_1000`
- `pansierBns_2000` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierBns_2000_pred/pansierBns_2000`
- `pansierBns_300` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierBns_300_pred/pansierBns_300`
- `pansierBns_600` from `/work/dlc2workfs3/zehlet-cayn/pansier_eval_ns/pansierBns_600_pred/pansierBns_600`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| leader_0.9549 | 299 | 0.0452 | 0.9548 | 0.2295 | 0.7705 | 0.0286 | 0.1667 | 0 |
| pansierAns_1000 | 299 | 0.0468 | 0.9532 | 0.2343 | 0.7657 | 0.0286 | 0.1667 | 0 |
| pansierAns_2000 | 299 | 0.0439 | 0.9561 | 0.2280 | 0.7720 | 0.0286 | 0.1667 | 0 |
| pansierAns_300 | 299 | 0.0470 | 0.9530 | 0.2382 | 0.7618 | 0.0294 | 0.2000 | 0 |
| pansierAns_600 | 299 | 0.0451 | 0.9549 | 0.2309 | 0.7691 | 0.0294 | 0.2000 | 0 |
| pansierBns_1000 | 299 | 0.0432 | 0.9568 | 0.2363 | 0.7637 | 0.0286 | 0.2000 | 0 |
| pansierBns_2000 | 299 | 0.0437 | 0.9563 | 0.2319 | 0.7681 | 0.0278 | 0.2000 | 0 |
| pansierBns_300 | 299 | 0.0506 | 0.9494 | 0.2533 | 0.7467 | 0.0294 | 0.2222 | 0 |
| pansierBns_600 | 299 | 0.0441 | 0.9559 | 0.2333 | 0.7667 | 0.0286 | 0.2000 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
