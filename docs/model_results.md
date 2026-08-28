# AlbucE OCR — model results (300-line validation)

Six line recognisers on the **300-line held-out validation set** (299 lines with predictions).
95% CIs = paired bootstrap, 10 000×, seed 42. Diplomatic (base-letter) transcription.

**Protocol note.** *Catmus* and *Medusa* are scored on the full-page eval — their fair protocol
(`tests/ocr/evaluations/seven_way_vs_validation_300/`). The fine-tuned *kraken* and *TrOCR* models
are scored natively on the line crops. Catmus's per-crop transcriber score (0.9552) is a ~0.5pp-harsh
harness artifact and is **not** used here; its fair number is 0.9613.

## Corpus-level

| Model | Arch | CER | char-acc [95% CI] | WER | word-acc [95% CI] | Size |
|---|---|---|---|---|---|---|
| kraken · CTC + LM (ours) | CTC + char-LM | **0.0256** | **0.9744 [0.9707, 0.9780]** | 0.1627 | 0.8373 [0.8143, 0.8597] | 4.08 M / 16 MB |
| kraken · CTC (ours) | CTC | 0.0290 | 0.9710 [0.9670, 0.9749] | 0.1798 | 0.8202 [0.7962, 0.8441] | 4.08 M / 16 MB |
| TrOCR · light-aug (ours) | ViT+RoBERTa | 0.0383 | 0.9617 [0.9573, 0.9660] | 0.2174 | 0.7826 [0.7590, 0.8059] | 282.6 M / 1.1 GB |
| Catmus (baseline) | Kraken CTC, frozen | 0.0387 | 0.9613 [0.9562, 0.9663] | **0.1434** | **0.8566 [0.8389, 0.8738]** | 4.08 M / 16 MB |
| TrOCR · med4k (ours) | ViT+RoBERTa | 0.0452 | 0.9548 [0.9496, 0.9596] | 0.2293 | 0.7707 [0.7462, 0.7938] | 282.6 M / 1.1 GB |
| Medusa (baseline) | Qwen-VL 9B | 0.0490 | 0.9510 [0.9459, 0.9558] | 0.3106 | 0.6894 [0.6593, 0.7191] | 9 B / ~18 GB |

## Per-line median

| Model | median CER | median char-acc | median WER | median word-acc |
|---|---|---|---|---|
| kraken · CTC + LM | **0.0238** | **0.9762** | 0.1250 | 0.8750 |
| kraken · CTC | 0.0250 | 0.9750 | 0.1250 | 0.8750 |
| TrOCR · light-aug | 0.0278 | 0.9722 | 0.1818 | 0.8182 |
| Catmus | 0.0278 | 0.9722 | 0.1250 | 0.8750 |
| TrOCR · med4k | 0.0286 | 0.9714 | 0.1667 | 0.8333 |
| Medusa | 0.0435 | 0.9565 | 0.2857 | 0.7143 |

## Takeaways

- **kraken CTC + LM** leads on **character accuracy** (0.9744) and CER — the per-position char-LM
  rescorer fixes minim substitutions on top of the CTC model.
- **Catmus** (frozen off-the-shelf CATMuS-medieval) leads on **word accuracy** (0.8566) and WER
  (0.1434) — the fewest whole-word errors, though its char-acc sits mid-field. Char-acc leader ≠
  word-acc leader.
- **Medusa** (9 B VLM) is last on every metric; the 4 M-param kraken pipeline is both more accurate
  and ~2000× smaller.

Sources: kraken/TrOCR per-line from their eval CSVs; Catmus/Medusa from `seven_way_vs_validation_300`.
Styled version: `docs/model_results.html` (published artifact). Bootstrap: `scripts/ocr/bootstrap_ocr_ci.py`.
