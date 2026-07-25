# 6-arch grid — 600-only + 600+3000 full stats on corrected 300-val (2026-07-26)

Completes the 6-arch × 4-scenario grid: the two scenarios that were still
training (§6.5.10 fill runs) — **600 real only** and **600 real + 3000 anno
re-renders** (no external corpus) — for all six encoder×decoder combos. Same
pipeline as §6.5.11–§6.5.14. Models transcribed on the VM 2026-07-26, pulled +
sha-verified. 299 non-empty lines. Eval folders
`tests/ocr/evaluations/grid_{600only,600_3000}_6arch_20260726/`.

## Scenario 600-only

### Corpus + per-line median

| arch | CER | char_acc | WER | word_acc | char_acc median | word_acc median |
|---|---|---|---|---|---|---|
| ViT+RoBERTa | 0.0578 | **0.9422** | 0.2718 | 0.7282 | 0.9524 | 0.7500 |
| Swin+GPT2 | 0.7419 | 0.2581 | 1.0068 | −0.0068 | 0.2571 | 0.0000 |
| ViT+GPT2 | 0.7439 | 0.2561 | 0.9956 | 0.0044 | 0.2571 | 0.0000 |
| ViT+BERT | 0.8328 | 0.1672 | 1.1954 | −0.1954 | 0.2308 | 0.0000 |
| Swin+BERT | 0.8652 | 0.1348 | 1.4895 | −0.4895 | 0.2308 | −0.1667 |
| Swin+xlm-RoBERTa | 0.9997 | 0.0003 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

### Paired bootstrap 95 % CI (10k, seed=42)

| arch | char_acc [95% CI] | word_acc [95% CI] |
|---|---|---|
| ViT+RoBERTa | 94.22 [93.55, 94.86] | 72.82 [70.13, 75.44] |
| Swin+GPT2 | 25.82 [25.06, 26.53] | −0.68 [−2.20, 0.75] |
| ViT+GPT2 | 25.62 [24.91, 26.29] | 0.44 [−0.84, 1.65] |
| ViT+BERT | 16.72 [14.30, 19.08] | −19.58 [−24.23, −15.05] |
| Swin+BERT | 13.47 [7.28, 18.36] | −49.02 [−60.03, −39.39] |
| Swin+xlm-RoBERTa | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] |

### Ink-bleed p90 (char_acc, clean n=270 / bleed n=29)

| arch | clean (p90=F) [95% CI] | bleed (p90=T) [95% CI] | Δ |
|---|---|---|---|
| ViT+RoBERTa | 94.58 [93.92, 95.20] | 90.87 [88.11, 93.44] | −3.71 |
| Swin+GPT2 | 25.73 [24.90, 26.52] | 26.60 [25.24, 27.96] | +0.87 |
| ViT+GPT2 | 25.64 [24.90, 26.37] | 25.27 [23.30, 27.27] | −0.37 |
| ViT+BERT | 16.24 [13.60, 18.73] | 21.14 [14.08, 27.73] | +4.90 |
| Swin+BERT | 14.49 [8.16, 19.04] | 3.77 [−27.47, 22.52] | −10.72 |
| Swin+xlm-RoBERTa | 0.03 [0.00, 0.09] | 0.00 [0.00, 0.00] | −0.03 |

## Scenario 600+3000 (anno re-renders, no external corpus)

### Corpus + per-line median

| arch | CER | char_acc | WER | word_acc | char_acc median | word_acc median |
|---|---|---|---|---|---|---|
| ViT+RoBERTa | 0.0793 | **0.9207** | 0.3223 | 0.6777 | 0.9333 | 0.7143 |
| Swin+xlm-RoBERTa | 0.7668 | 0.2332 | 1.0836 | −0.0836 | 0.2286 | 0.0000 |
| ViT+GPT2 | 0.7711 | 0.2289 | 1.0117 | −0.0117 | 0.2353 | 0.0000 |
| ViT+BERT | 0.7827 | 0.2173 | 1.0073 | −0.0073 | 0.2188 | 0.0000 |
| Swin+GPT2 | 0.7981 | 0.2019 | 1.0627 | −0.0627 | 0.2000 | 0.0000 |
| Swin+BERT | 0.8460 | 0.1540 | 1.2270 | −0.2270 | 0.1538 | −0.2857 |

### Paired bootstrap 95 % CI (10k, seed=42)

| arch | char_acc [95% CI] | word_acc [95% CI] |
|---|---|---|
| ViT+RoBERTa | 92.08 [91.30, 92.82] | 67.78 [65.09, 70.38] |
| Swin+xlm-RoBERTa | 23.32 [22.39, 24.24] | −8.35 [−10.72, −6.12] |
| ViT+GPT2 | 22.89 [21.95, 23.82] | −1.16 [−2.24, −0.15] |
| ViT+BERT | 21.73 [20.81, 22.53] | −0.73 [−1.96, 0.39] |
| Swin+GPT2 | 20.19 [19.34, 21.05] | −6.27 [−8.01, −4.62] |
| Swin+BERT | 15.40 [14.31, 16.47] | −22.71 [−25.61, −19.82] |

### Ink-bleed p90 (char_acc, clean n=270 / bleed n=29)

| arch | clean (p90=F) [95% CI] | bleed (p90=T) [95% CI] | Δ |
|---|---|---|---|
| ViT+RoBERTa | 92.24 [91.46, 92.99] | 90.47 [87.45, 93.27] | −1.77 |
| Swin+xlm-RoBERTa | 23.10 [22.16, 24.05] | 25.46 [22.11, 29.40] | +2.36 |
| ViT+GPT2 | 23.07 [22.07, 24.01] | 21.10 [17.53, 24.06] | −1.97 |
| ViT+BERT | 21.75 [20.79, 22.61] | 21.49 [19.23, 23.70] | −0.26 |
| Swin+GPT2 | 19.89 [19.03, 20.77] | 22.89 [20.08, 26.07] | +3.00 |
| Swin+BERT | 15.49 [14.40, 16.52] | 14.49 [9.87, 18.53] | −1.00 |

## ViT+RoBERTa across all 4 scenarios (the key story)

| scenario | char_acc | word_acc | char_acc [95% CI] |
|---|---|---|---|
| 600-only | **0.9422** | 0.7282 | 94.22 [93.55, 94.86] |
| 600+3000 | 0.9207 | 0.6777 | 92.08 [91.30, 92.82] |
| +COMETA(1000) | 0.9345 | 0.7210 | 93.45 [92.69, 94.18] |
| +medical(1000) | 0.9389 | 0.7346 | 93.89 [93.07, 94.63] |

Paired bootstrap (same resampled lines):
- **600-only vs 600+3000 = +2.14 % [+1.54, +2.74] (P=1.000 ✓sig)** — adding 3000
  *synthetic* re-renders **significantly hurts** the pretrained arch.
- **+medical(1000) vs 600+3000 = +1.82 % [+1.09, +2.47] (P=1.000 ✓sig)** — adding
  external real-text corpus **significantly recovers** most of that loss.
- 600-only vs +medical(1000) = +0.33 % [−0.35, +1.07] (ns) — 600-only ≈
  +medical(1000): the corpus recovers *back to*, not above, the real-only level.

**Interpretation.** For ViT+RoBERTa the ranking is
**600-only ≈ +medical(1000) > +COMETA(1000) > 600+3000**. Synthetic re-renders
alone *dilute* the real-manuscript signal; it takes external **text** diversity
(corpus) to justify the extra rendered images, and only at high volume
(4000-external = 0.9438/0.9487, §6.5.13) does the augmented pool finally exceed
plain 600-real. Take-away: **more synthetic renders ≠ better; text diversity is
what pays.**

## Cross-arch

Only **ViT+RoBERTa** does real OCR in every scenario (0.92–0.94). The five
from-scratch archs stay **near-random** (≤0.28). Two notable floors:
- **Swin+xlm-RoBERTa 600-only = 0.0003** — total collapse: with only 600 lines
  and a 250k-token multilingual vocab it never learns to emit text (near-empty
  output). It only reaches its ~0.23–0.28 floor once re-renders/corpus are added.
- **Swin+BERT 600-only word_acc = −0.49** (WER ≈ 1.49) — worst word-level output
  in the program; the from-scratch cross-attention over-generates wildly on the
  smallest data.

Ink-bleed Δ is only interpretable for ViT+RoBERTa (−3.71 on 600-only, −1.77 on
600+3000); the near-random archs show noisy/positive Δ (they aren't decoding
glyphs, so bleed can't hurt them).
