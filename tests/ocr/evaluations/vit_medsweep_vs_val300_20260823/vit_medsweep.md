# OCR evaluation — vit_medsweep

Ground truth: `/work/dlc2workfs3/zehlet-cayn/data/processed/annotated_samples/OCR/validation` (300 lines)

Models compared:
- `leader_0.9548` from `/work/dlc2workfs3/zehlet-cayn/vit_medsweep_eval/leader_0.9548_pred/leader_0.9548`
- `vit_lightreal_med1k` from `/work/dlc2workfs3/zehlet-cayn/vit_medsweep_eval/vit_lightreal_med1k_pred/vit_lightreal_med1k`
- `vit_lightreal_med4k` from `/work/dlc2workfs3/zehlet-cayn/vit_medsweep_eval/vit_lightreal_med4k_pred/vit_lightreal_med4k`
- `vit_lightreal_med7k` from `/work/dlc2workfs3/zehlet-cayn/vit_medsweep_eval/vit_lightreal_med7k_pred/vit_lightreal_med7k`

| model | lines | CER | char_acc | WER | word_acc | CER_median | WER_median | missing |
|---|---|---|---|---|---|---|---|---|
| leader_0.9548 | 299 | 0.0452 | 0.9548 | 0.2295 | 0.7705 | 0.0286 | 0.1667 | 0 |
| vit_lightreal_med1k | 299 | 0.0434 | 0.9566 | 0.2246 | 0.7754 | 0.0278 | 0.1667 | 0 |
| vit_lightreal_med4k | 299 | 0.0383 | 0.9617 | 0.2173 | 0.7827 | 0.0278 | 0.1818 | 0 |
| vit_lightreal_med7k | 299 | 0.0379 | 0.9621 | 0.2071 | 0.7929 | 0.0278 | 0.1667 | 0 |

_CER = character error rate (corpus-level: sum of edits / sum of reference characters). char_acc = 1 - CER. Median = per-line median, less sensitive to one bad line. Missing = GT stems with no prediction file from that model._
