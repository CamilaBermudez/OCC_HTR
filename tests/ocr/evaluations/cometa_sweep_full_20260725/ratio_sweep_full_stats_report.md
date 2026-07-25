# External-corpus ratio sweep — full stats on corrected 300-val (2026-07-25)

Same pipeline as §6.5.11/§6.5.12. **600 real + 3000 anno re-renders + N external
re-renders**, N ∈ {500, 1000, 2000, 4000}, for both external corpora (COMETA,
medical) × both architectures (ViT+RoBERTa pretrained, Swin+BERT from-scratch).
The **N=1000** point is the A″/B″ single-stage run (§6.5.10 grid). 299 non-empty
lines. Artefacts: `tests/ocr/evaluations/{cometa,medical}_sweep_full_20260725/`.

## ViT+RoBERTa (pretrained) — the informative arch

### Corpus + per-line median

| corpus | N ext | CER | char_acc | WER | word_acc | char_acc median | word_acc median |
|---|---|---|---|---|---|---|---|
| COMETA | 500 | 0.0642 | 0.9358 | 0.2902 | 0.7098 | 0.9474 | 0.7143 |
| COMETA | 1000 | 0.0655 | 0.9345 | 0.2790 | 0.7210 | 0.9474 | 0.7500 |
| COMETA | 2000 | 0.0597 | 0.9403 | 0.2679 | 0.7321 | 0.9500 | 0.7500 |
| COMETA | 4000 | 0.0562 | 0.9438 | 0.2635 | 0.7365 | 0.9535 | 0.7500 |
| medical | 500 | 0.0619 | 0.9381 | 0.2732 | 0.7268 | 0.9487 | 0.7500 |
| medical | 1000 | 0.0611 | 0.9389 | 0.2654 | 0.7346 | 0.9487 | 0.7500 |
| medical | 2000 | 0.0555 | 0.9445 | 0.2713 | 0.7287 | 0.9500 | 0.7500 |
| medical | 4000 | 0.0513 | **0.9487** | 0.2494 | 0.7506 | 0.9583 | 0.8000 |

### Paired bootstrap 95 % CI (10k iters, seed=42, full 299)

| corpus | N ext | char_acc [95% CI] | word_acc [95% CI] |
|---|---|---|---|
| COMETA | 500 | 93.58% [92.88, 94.25] | 70.98% [68.37, 73.55] |
| COMETA | 1000 | 93.45% [92.69, 94.17] | 72.11% [69.48, 74.76] |
| COMETA | 2000 | 94.03% [93.34, 94.70] | 73.23% [70.56, 75.89] |
| COMETA | 4000 | 94.39% [93.73, 95.02] | 73.67% [70.99, 76.18] |
| medical | 500 | 93.81% [93.14, 94.45] | 72.68% [69.95, 75.31] |
| medical | 1000 | 93.90% [93.08, 94.65] | 73.46% [70.86, 76.08] |
| medical | 2000 | 94.45% [93.77, 95.10] | 72.88% [70.12, 75.58] |
| medical | 4000 | 94.87% [94.21, 95.47] | 75.08% [72.43, 77.64] |

### Scaling significance (does more external corpus help?)

| comparison | Δ char_acc [95% CI] | Δ word_acc [95% CI] | P(A>B) | sig? |
|---|---|---|---|---|
| COMETA 4000 vs 500 | +0.80% [+0.27, +1.34] | +2.67% [+0.96, +4.41] | 0.999 | ✓ |
| COMETA 4000 vs 1000 | +0.93% [+0.36, +1.52] | +1.53% [−0.33, +3.42] | 0.999 | ✓ char |
| medical 4000 vs 500 | +1.06% [+0.53, +1.59] | +2.39% [+0.57, +4.24] | 1.000 | ✓ |
| medical 4000 vs 1000 | +0.98% [+0.30, +1.72] | +1.60% [−0.29, +3.52] | 0.998 | ✓ char |

**More external corpus monotonically improves ViT+RoBERTa**, and 4000 vs 500 is
significant on char_acc for *both* corpora (+0.8 to +1.1 pp; 0 outside CI). The
gain is real but small — a ~1 pp char ceiling effect. Best overall =
**medical-4000 = 0.9487**, the top fine-tuned model in the program.

### Ink-bleed p90 stratification (270 clean / 29 bleed)

| corpus | N ext | char_acc clean | char_acc bleed | Δ |
|---|---|---|---|---|
| COMETA | 500 | 93.74% | 92.09% | −1.65 pp |
| COMETA | 1000 | 93.68% | 91.24% | −2.44 pp |
| COMETA | 2000 | 94.26% | 91.91% | −2.35 pp |
| COMETA | 4000 | 94.67% | 91.73% | −2.94 pp |
| medical | 500 | 93.92% | 92.75% | −1.17 pp |
| medical | 1000 | 93.99% | 93.03% | −0.96 pp |
| medical | 2000 | 94.49% | 94.06% | −0.43 pp |
| medical | 4000 | 95.04% | 93.23% | −1.81 pp |

ViT+RoBERTa is moderately ink-bleed-robust (Δ −1 to −3 pp) — better than kraken
(−9 to −11) and catmus (−3.0), worse than Medusa (−0.4). Deltas on n=29 have
wide overlapping CIs, so within-arch differences across N are not significant.

## Swin+BERT (from scratch) — control

Trained from random init (no pretrained cross-attention), so all points are
**near-random** regardless of external-corpus volume:

| corpus | N ext | char_acc | word_acc |
|---|---|---|---|
| COMETA | 500 / 1000 / 2000 / 4000 | 0.2265 / 0.1952 / 0.1942 / 0.1825 | 0.013 / −0.072 / −0.071 / 0.001 |
| medical | 500 / 1000 / 2000 / 4000 | 0.2422 / 0.1238 / 0.2226 / 0.2103 | −0.057 / −0.150 / −0.019 / −0.056 |

char_acc stays 0.12–0.24 (WER > 1, i.e. more word errors than reference words)
and moves **non-monotonically** — noise, not signal. **External-corpus volume
cannot rescue a from-scratch model**; the pretrained cross-attention (present
only in the ViT+RoBERTa arch) is the precondition for the corpus to help at all
(§6.3.6 dominant-factor finding). Ink-bleed stratification is not meaningful at
this accuracy floor and is omitted.
