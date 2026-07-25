# 6-arch grid — A″/B″ (1000-external) full stats on corrected 300-val (2026-07-25)

Same pipeline as §6.5.11–§6.5.13. The two **completed** scenarios of the
6-arch × 4-scenario grid: **600 + 3000 re-renders + 1000 external**, external ∈
{COMETA = A″, medical = B″}, for all six encoder×decoder combos. (The other two
scenarios — 600-only and 600+3000 — are the fill runs still training on the VM,
§6.5.10.) GPT-2 archs use the **v2 (pad/eos-fix)** transcriptions (§6.5.7).
299 non-empty lines. Artefacts:
`tests/ocr/evaluations/grid_{cometa,medical}1000_6arch_20260725/`.

Only **ViT+RoBERTa** carries pretrained cross-attention (`microsoft/trocr-base
-handwritten`); the other five are trained from random init on 600 manuscript
lines + re-renders. This is the §6.3.6 dominant-factor result in one grid:
pretrained cross-attention is the precondition for functional OCR.

## Scenario A″ = 600+3000+COMETA(1000)

### Corpus + per-line median

| arch | CER | char_acc | WER | word_acc | char_acc median | word_acc median |
|---|---|---|---|---|---|---|
| ViT+RoBERTa | 0.0655 | **0.9345** | 0.2790 | 0.7210 | 0.9474 | 0.7500 |
| Swin+xlm-RoBERTa | 0.7190 | 0.2810 | 1.0331 | −0.0331 | 0.2821 | 0.0000 |
| Swin+GPT2 | 0.7414 | 0.2586 | 0.9903 | 0.0097 | 0.2500 | 0.0000 |
| ViT+BERT | 0.7942 | 0.2058 | 1.0710 | −0.0710 | 0.2000 | 0.0000 |
| ViT+GPT2 | 0.7946 | 0.2054 | 1.1473 | −0.1473 | 0.2000 | −0.1429 |
| Swin+BERT | 0.8048 | 0.1952 | 1.0719 | −0.0719 | 0.1944 | 0.0000 |

### Paired bootstrap 95 % CI (10k iters, seed=42)

| arch | char_acc [95% CI] | word_acc [95% CI] |
|---|---|---|
| ViT+RoBERTa | 93.45 [92.69, 94.18] | 72.12 [69.41, 74.82] |
| Swin+xlm-RoBERTa | 28.09 [27.29, 28.90] | −3.30 [−5.14, −1.54] |
| Swin+GPT2 | 25.85 [25.04, 26.63] | 0.96 [−0.20, 2.07] |
| ViT+BERT | 20.58 [19.70, 21.48] | −7.11 [−9.05, −5.24] |
| ViT+GPT2 | 20.55 [19.72, 21.38] | −14.73 [−16.87, −12.67] |
| Swin+BERT | 19.53 [18.66, 20.38] | −7.21 [−9.16, −5.31] |

### Ink-bleed p90 (char_acc, clean n=270 / bleed n=29)

| arch | clean (p90=F) [95% CI] | bleed (p90=T) [95% CI] | Δ |
|---|---|---|---|
| ViT+RoBERTa | 93.68 [92.96, 94.40] | 91.26 [87.98, 94.07] | −2.42 |
| Swin+xlm-RoBERTa | 28.07 [27.23, 28.91] | 28.29 [25.30, 31.54] | +0.22 |
| Swin+GPT2 | 25.66 [24.77, 26.53] | 27.64 [25.70, 29.63] | +1.98 |
| ViT+BERT | 20.57 [19.74, 21.41] | 20.72 [16.06, 26.09] | +0.15 |
| ViT+GPT2 | 20.75 [19.89, 21.60] | 18.45 [15.29, 21.07] | −2.30 |
| Swin+BERT | 19.61 [18.68, 20.50] | 18.66 [16.06, 21.29] | −0.95 |

## Scenario B″ = 600+3000+medical(1000)

### Corpus + per-line median

| arch | CER | char_acc | WER | word_acc | char_acc median | word_acc median |
|---|---|---|---|---|---|---|
| ViT+RoBERTa | 0.0611 | **0.9389** | 0.2654 | 0.7346 | 0.9487 | 0.7500 |
| Swin+xlm-RoBERTa | 0.7264 | 0.2736 | 1.0379 | −0.0379 | 0.2703 | 0.0000 |
| Swin+GPT2 | 0.7971 | 0.2029 | 1.0404 | −0.0404 | 0.2000 | 0.0000 |
| ViT+BERT | 0.8015 | 0.1985 | 1.0822 | −0.0822 | 0.2000 | 0.0000 |
| ViT+GPT2 | 0.8187 | 0.1813 | 1.0841 | −0.0841 | 0.1765 | 0.0000 |
| Swin+BERT | 0.8762 | 0.1238 | 1.1497 | −0.1497 | 0.1282 | −0.1429 |

### Paired bootstrap 95 % CI (10k iters, seed=42)

| arch | char_acc [95% CI] | word_acc [95% CI] |
|---|---|---|
| ViT+RoBERTa | 93.89 [93.07, 94.63] | 73.48 [70.84, 76.08] |
| Swin+xlm-RoBERTa | 27.37 [26.51, 28.23] | −3.79 [−5.85, −1.82] |
| Swin+GPT2 | 20.29 [19.53, 21.06] | −4.03 [−5.51, −2.60] |
| ViT+BERT | 19.85 [19.00, 20.71] | −8.23 [−10.32, −6.27] |
| ViT+GPT2 | 18.13 [17.20, 19.08] | −8.41 [−10.11, −6.81] |
| Swin+BERT | 12.38 [10.84, 13.86] | −14.98 [−17.77, −12.38] |

### Ink-bleed p90 (char_acc, clean n=270 / bleed n=29)

| arch | clean (p90=F) [95% CI] | bleed (p90=T) [95% CI] | Δ |
|---|---|---|---|
| ViT+RoBERTa | 93.99 [93.15, 94.74] | 93.01 [90.57, 95.26] | −0.98 |
| Swin+xlm-RoBERTa | 27.34 [26.47, 28.20] | 27.56 [24.38, 31.15] | +0.22 |
| Swin+GPT2 | 20.13 [19.34, 20.93] | 21.78 [18.67, 24.66] | +1.65 |
| ViT+BERT | 19.77 [18.86, 20.69] | 20.63 [18.03, 22.93] | +0.86 |
| ViT+GPT2 | 18.09 [17.15, 19.03] | 18.50 [14.80, 22.00] | +0.41 |
| Swin+BERT | 12.42 [10.84, 13.89] | 11.81 [5.31, 17.64] | −0.61 |

## Reading

- **Only ViT+RoBERTa does real OCR** (~0.934–0.939 char_acc). The other five
  are **near-random** (0.12–0.28 char_acc, WER > 1 ⇒ negative word_acc — more
  word errors than reference words). Confirms cross-attention pretraining, not
  decoder identity, is the deciding factor (§6.3.6).
- **Corpus choice (COMETA vs medical) barely moves ViT+RoBERTa** at 1000
  external: medical 0.9389 vs COMETA 0.9345 (overlapping CIs) — the corrected-GT
  collapse of the old medical>COMETA finding (§6.4). Scaling the external slot to
  4000 is what actually helps (§6.5.13), not the corpus label at 1000.
- **Ink-bleed Δ is only meaningful for ViT+RoBERTa** (−2.42 A″ / −0.98 B″,
  moderately robust). For the near-random archs the "Δ" is noise — several are
  *positive* (bleed ≥ clean) precisely because the model isn't reading glyphs,
  so bleed can't hurt what was never being decoded. Do not interpret their
  bleed robustness.
- Among from-scratch archs, **Swin+xlm-RoBERTa is nominally top** (~0.27–0.28)
  but still non-functional; ranking within the floor is not practically
  meaningful.
