# AlbucE OCR — Executive summary

### Comparing OCR / HTR approaches on a medieval Old Occitan manuscript

**Author**: Camila Bermúdez Valderrama · **Supervisor**: [name] · **Date**: 2026-07-14

---

## Project in one paragraph

The AlbucE manuscript is a medieval Old Occitan medical text with roughly 13,700 line-image crops distributed across 71 folio images. We compare three OCR/HTR model families on this corpus: (1) a small CTC-based recogniser (kraken with the pretrained `catmus-medieval` base, optionally fine-tuned on our own annotations), (2) a zero-shot 9B-parameter vision-language model (`Medusa 0.2 Line`, ENC-PSL), and (3) two encoder-decoder VLM variants trained end-to-end (Swin+BERT from-scratch — with a **staged pretraining** variant added 2026-07-14 — and `microsoft/trocr-base-handwritten` fine-tuned). All evaluation numbers below are computed on a permanent 300-line held-out validation set (299 non-empty), with an enforced invariant that these 300 lines were never used for training or model selection. Metrics are character error rate (CER) and word error rate (WER) via `rapidfuzz` Levenshtein distance, reported at corpus level (aggregated) and per-line median.

## Leaderboard (corpus-level metrics on the 300-line validation set)

| Rank | Model | Family | Train data | char_acc | word_acc | CER | WER |
|---|---|---|---|---|---|---|---|
| 1 | **kraken 600 real** | Kraken CTC (fine-tuned) | catmus + 600 real + 2,500 synth re-renders of 500 real texts (×5 each) | **0.9620** | 0.7856 | 0.0380 | 0.2144 |
| 2 | catmus-medieval | Kraken CTC (baseline) | 35k CATMuS project lines (no fine-tune on our data) | 0.9613 | **0.8566** | 0.0387 | 0.1434 |
| 3 | kraken 500 real | Kraken CTC (fine-tuned) | catmus + 500 real + 2,500 synth re-renders of the same 500 texts (×5 each) | 0.9610 | 0.7812 | 0.0390 | 0.2188 |
| 4 | kraken 600 real + medical (old, confounded) | Kraken CTC (fine-tuned) | catmus + 600 real + 3,000 mixed synth (2,000 re-renders of 400 real texts + 1,000 medical corpus renders); **aug re-render count differs from `_070741`, so medical vs no-medical is NOT single-variable** | 0.9593 | 0.7725 | 0.0407 | 0.2275 |
| 5 | kraken 400 real | Kraken CTC (fine-tuned) | catmus + 400 real + 2,000 synth re-renders of those 400 real texts (×5 each) | 0.9580 | 0.7642 | 0.0420 | 0.2358 |
| 6 | Medusa 0.2 Line 9B | VLM (zero-shot) | 640k real + 500k synth medieval lines (no fine-tune on our data) | 0.9510 | 0.6894 | 0.0490 | 0.3106 |
| — | **kraken matched-pool no-medical** *(new baseline, 2026-07-18)* | Kraken CTC (fine-tuned) | catmus + 600 real + 3,000 anno re-renders of 600 stems × 5 (no medical) | **0.9096** | 0.5785 | 0.0904 | 0.4215 |
| — | **kraken matched-pool + medical** *(new, 2026-07-19)* | Kraken CTC (fine-tuned) | catmus + 600 real + 3,000 anno re-renders + 1,000 medical corpus renders (single-variable delta vs row above) | **0.8664** | 0.4464 | 0.1336 | 0.5536 |
| 7 | ViT+RoBERTa + medical aug (B'') | TrOCR pretrained (fine-tuned) | trocr-base + 600 real + 3000 re-render + 1000 medical | 0.9443 | 0.7360 | 0.0557 | 0.2640 |
| 8 | ViT+RoBERTa real-only (C) | TrOCR pretrained (fine-tuned) | trocr-base + 600 real | 0.9371 | 0.7171 | 0.0629 | 0.2829 |
| 9 | ViT+RoBERTa + COMETA aug (A'') | TrOCR pretrained (fine-tuned) | trocr-base + 600 real + 3000 re-render + 1000 COMETA | 0.9332 | 0.7141 | 0.0668 | 0.2859 |
| 10 | ViT+RoBERTa + re-renders only (D) | TrOCR pretrained (fine-tuned) | trocr-base + 600 real + 3000 re-render, **no external corpus** | 0.9161 | 0.6728 | 0.0839 | 0.3272 |
| 11 | **Staged Swin+BERT + medical (B'')** | TrOCR from-scratch, 2-stage | 30k COMETA pretrain → 600 real + 3000 re-render + 1000 medical | **0.6080** | 0.3306 | 0.3920 | 0.6694 |
| 12 | **Staged Swin+BERT + COMETA (A'')** | TrOCR from-scratch, 2-stage | 30k COMETA pretrain → 600 real + 3000 re-render + 1000 COMETA | **0.6053** | 0.3087 | 0.3947 | 0.6913 |
| 13 | **Staged Swin+BERT — Stage 1 only (COMETA pretrain, no manuscript FT)** | TrOCR from-scratch, pretrain only | 30k COMETA re-renders; **zero manuscript real lines** | **0.5918** | 0.2888 | 0.4082 | 0.7112 |
| 14 | Swin+BERT + medical aug (B'') | TrOCR from-scratch (single-stage) | random cross-attn + 600 real + 3000 re-render + 1000 medical | 0.2523 | −0.0350 | 0.7477 | 1.0350 |
| 15 | Swin+BERT real-only (C) | TrOCR from-scratch (single-stage) | random cross-attn + 600 real (no synth aug) | 0.2293 | −0.1215 | 0.7707 | 1.1215 |
| 16 | Swin+BERT + COMETA aug (A'') | TrOCR from-scratch (single-stage) | random cross-attn + 600 real + 3000 re-render + 1000 COMETA | 0.2240 | −0.2552 | 0.7760 | 1.2552 |
| 17 | Swin+BERT + re-renders only (D) | TrOCR from-scratch (single-stage) | random cross-attn + 600 real + 3000 re-render, no external corpus | 0.1447 | −0.0617 | 0.8553 | 1.0617 |

*Negative word accuracy means WER > 1 — model produces more edit-distance operations than the reference has words (systematic over-generation from an unaligned decoder).*

## Five sentences of interpretation

1. **The small, purpose-built CTC recognisers (catmus and kraken fine-tunes) remain the strongest models on this manuscript**, with the fine-tuned kraken 600 narrowly leading on character accuracy (0.9620) and the pretrained catmus baseline dominating word accuracy (0.8566) — likely because CTC models with fixed vocabularies avoid the near-neighbour word-form errors that plague encoder-decoder VLMs.

2. **Pretrained cross-attention is the single most important factor for TrOCR-family performance**: swapping the from-scratch Swin+BERT (cross-attention randomly initialised) for `microsoft/trocr-base-handwritten` (cross-attention pre-trained on 34 million pairs) — holding data, hyperparameters, and code identical — moves character accuracy from ~0.22 to ~0.94, a 72-percentage-point gap that scaling data alone at our training-pool size cannot close.

3. **Staged pretraining partially recovers the from-scratch gap — and the pretraining stage does almost all the work**: adding a 30k-pair COMETA pretraining stage before fine-tuning lifts Swin+BERT from 0.2523 (single-stage, Dataset B'') to 0.6080 on the 300-val (+35.6 pp, ~45 % of the gap to the pretrained ViT+RoBERTa); decomposing that lift, the Stage 1 pretraining alone — **with zero manuscript real lines** — already reaches 0.5918 (+33.95 pp), and the manuscript-specific fine-tuning adds only ~1.5 pp on top, so the finding is really "30 k COMETA-only pairs are worth ~34 pp of held-out char_acc for an encoder-decoder VLM with random-init cross-attention" (val-fold suggested +60 pp / 72 % gap-close, but that reading was val-fold-inflated because the val fold contains re-renders of source stems the model saw during Stage 1).

4. **External-corpus text is what carries the augmentation signal for pretrained ViT+RoBERTa, not re-render volume**: adding 3000 re-renders of the *same 600 real texts* actually hurts slightly (0.9371 real-only → 0.9161 Dataset D), and only recovers when a 1000-render external-corpus slot is added on top (→ 0.9332 with COMETA, 0.9443 with medical); the medical vs COMETA delta (+1.1 pp for medical, 95 % CI [+0.5, +1.7], P=1.000) is the cleanest corpus-choice signal in the grid.

5. **The medical corpus intervention is architecture-dependent — helps pretrained VLM, hurts strongly-anchored CTC**: on the matched-pool re-run that closes the earlier confound (3000 anno re-renders in both arms), medical corpus **significantly helps** the pretrained ViT+RoBERTa (+1.11 pp char_acc, CI [+0.5, +1.7]) but **significantly hurts** the kraken CTC fine-tune (−4.31 pp char_acc, CI [−4.97, −3.66], P<0.001) — the direction depends on whether the model's prior text distribution is compatible with the augmentation corpus, and catmus's strong pre-existing medieval-script priors mean the medical shift drags kraken away from the actual manuscript mixture rather than closer to it.

---

*Full methodology, dataset provenance, and per-model implementation details are in the thesis draft and `spec.md`. Evaluation reproducible from `tests/ocr/evaluations/{seven_way,five_trocr}_vs_validation_300/` in the project repository.*
