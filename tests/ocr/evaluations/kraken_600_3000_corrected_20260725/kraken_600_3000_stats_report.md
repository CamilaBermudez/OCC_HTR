# Kraken matched-pool stats on corrected 300-val (2026-07-25)

Same statistics pipeline as catmus/Medusa (§6.5.11) and the TrOCR tracks:
corpus + per-line median, paired bootstrap 95 % CI (10k iters, seed=42),
ink-bleed p90 stratification.

Models (leak-fixed, matched aug pool — kraken catmus-medieval base fine-tuned):
- **kraken 600+3000** = 600 real + 3000 anno re-renders, no external corpus
  (`kraken_matched_nomedical_leakfixed_val300_20260722`)
- **kraken 600+3000+Medical(1000)** = same + 1000 medical-corpus re-renders
  (`kraken_matched_medical_leakfixed_val300_20260722`)

GT: `data/processed/annotated_samples/OCR/validation` (299 non-empty lines).

## Corpus + per-line median

| model | CER | char_acc | WER | word_acc | char_acc median | word_acc median |
|---|---|---|---|---|---|---|
| kraken 600+3000 | 0.0982 | **0.9018** | 0.4439 | 0.5561 | 0.9189 | 0.5714 |
| kraken 600+3000+Medical(1000) | 0.1006 | 0.8994 | 0.4589 | 0.5411 | 0.9167 | 0.5714 |

## Paired bootstrap 95 % CI (full 299 lines)

| model | char_acc [95% CI] | word_acc [95% CI] |
|---|---|---|
| kraken 600+3000 | 90.18% [89.15, 91.14] | 55.60% [52.24, 58.86] |
| kraken 600+3000+Medical(1000) | 89.94% [88.97, 90.89] | 54.10% [50.72, 57.49] |
| **Δ (no-med − medical)** | **+0.23% [−0.24, +0.65] ns** | **+1.50% [−0.14, +3.13] ns** |

P(no-med > medical char_acc) = 0.842. **Adding the 1000 medical re-renders
neither helps nor hurts kraken** on the corrected GT — 0 lies inside both
difference CIs. (This is the corrected-GT collapse of the earlier "medical
hurts kraken" finding: on old GT it looked significant; it is not.)

## Ink-bleed p90 stratification (270 clean / 29 bleed lines)

| model | char_acc clean (p90=F) | char_acc bleed (p90=T) | Δ (bleed − clean) |
|---|---|---|---|
| kraken 600+3000 | 91.24% | 80.18% | **−11.06 pp** |
| kraken 600+3000+Medical(1000) | 90.87% | 81.20% | **−9.67 pp** |

Kraken is the **least ink-bleed-robust family in the program** (Δ −9.7 to
−11.1 pp), far worse than Medusa (−0.37), catmus (−3.04) and the TrOCR tracks
(−2 to −5). A CTC model with no language prior has nothing to fall back on
when glyphs bleed. On the 29 heavy-bleed lines the medical version is nominally
better (81.20 vs 80.18), reversing the full-set order — but the CIs overlap
almost entirely at n=29, so this is not significant.
