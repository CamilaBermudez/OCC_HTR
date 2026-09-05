# OCC-HTR — user guide

How to actually do things: prepare data, fine-tune, transcribe, align,
compare, and which analysis scripts exist. Conventions and layout are in
[`docs/project_organization.md`](project_organization.md).

## 0. Setup

```bash
uv sync                    # create .venv from pyproject.toml
make setup-precommit       # hooks (ruff etc.)
```

Everything below is a make target; override any variable inline:
`make <target> VAR=value`. Long runs log to `logs/<area>/`.

## 1. From page photos to line crops (preprocessing)

Run once per manuscript (already done for AlbucE; the products are in
`data/processed/`):

```
make create_masks          # YOLO layout masks
make segment_images        # kraken line segmentation inside the masks
make crop_segments         # per-line polygon crops
make binarize_image        # otsu_gaussian binarization (for filtering)
make filter_images         # drop noise/marginalia -> the "kept" crops
make detect_ink_bleed      # per-line ink-bleed score (feeds analyses + manifest)
```

The **kept crops** (`data/processed/filtered_images/<stamp>/original/kept/<page>/`)
are the canonical inputs for every transcription and training step.

## 2. Ground-truth samples: how to store them

One flat folder per sample set; each line is a **pair of files with the same
stem**:

```
<page>_line_<N>.png       # the line crop (copied from the kept crops)
<page>_line_<N>.gt.txt    # UTF-8, one line: the diplomatic transcription
```

Rules:
- Diplomatic style: transcribe what the scribe wrote (u/v, i/j, long-s as
  base letters; keep ⁊, ꝑ, tilde abbreviations as characters).
- Train pool and validation live in **different folders** and must stay
  disjoint: `data/processed/annotated_samples/OCR/full_annotated/` (600,
  training) vs `.../validation/` (300, **report-only — never train/tune on it**).
- To draw a fresh annotation batch without touching existing sets:
  `make sample_annotation_batch SAMPLE_SEED=<next>` (excludes previously
  annotated stems automatically; seed 100 is reserved for the validation draw).
  Annotate the OCR-seeded `.gt.txt` files it produces, then move the finished
  pairs into the training folder.
- The frontend's *Review & correct* tab appends corrections to a JSONL log —
  a future source of additional training pairs.

## 3. Fine-tuning

**kraken (CTC, recommended path):**
```bash
make kraken_finetune NO_SYNTH_TRAIN=1 FINETUNE_AUGMENT=1 FINETUNE_DEVICE=mps
```
Real-only + ketos augmentation is the settled best recipe (synthetic renders
do NOT help kraken — spec §6.5). Output:
`models/ocr/finetuned/finetune_<stamp>/model_best.mlmodel`.

**TrOCR (ViT+RoBERTa seq2seq):**
```bash
make trocr_finetune TROCR_PRETRAINED_MODEL_ID=microsoft/trocr-base-handwritten
```
Output: `models/ocr/finetuned/trocr_<stamp>/best_model/` (a HF checkpoint dir;
`resize_mode.txt` inside records the image-prep mode — keep it with the model).
Large mixed-pool runs go on the cluster (`scripts/cluster/*.sbatch`).

**Evaluate a model** on the 300-val:
`run_evaluate_ocr.py` (per-line CSV + corpus metrics), then
`bootstrap_ocr_ci.py` for 95% CIs. Consolidated results:
`docs/model_results.md` (`plot_model_results.py` regenerates the figures).

## 4. Transcribing the full manuscript

```bash
make run_transcription                      # frozen catmus baseline
make kraken_lm_transcribe                   # deployed leader: CTC + char-LM (λ=0.2)
make trocr_transcribe_conf                  # TrOCR with per-token confidence
make conf_to_txt                            # conf JSON -> per-page txt layout
```

All produce `data/processed/transcription/<run>/<page>/<page>_line_<N>.txt`
(+ `<page>_full.txt`), the layout every downstream tool reads.
`make trocr_transcribe` transcribes an arbitrary folder of crops (e.g. the
300-val) with any TrOCR checkpoint.

## 5. Aligning the scholarly edition (one-off per edition)

The scholarly edition is a continuous text; the manuscript's line breaks come
from an **auxiliary OCR transcription** (any decent full-corpus run):

```bash
make align_scholarly_edition \
     SCHOLARLY_REFERENCE_TXT=./data/raw/AlbucE.txt \
     SCHOLARLY_ALIGN_OCR_DIR=./data/processed/transcription/<aux_run>
```

Two-pass anchored alignment (`src/ocr/scholarly_alignment.py`): per-page
n-gram anchor + word-level DP → one "hi" endpoint per OCR line → lossless
partition of the reference (the output re-concatenates to the edition exactly;
verified, non-zero exit on violation). Output:
`tests/ocr/AlbucE_aligned_<stamp>.txt` — freeze it and set `SCHOLARLY_TXT` to
it for all subsequent steps.

## 6. Comparing a transcription against the edition

For any model's full-corpus run (`MODEL_TRANSCRIPTION_DIR`):

```bash
make align_transcriptions   # content-based line alignment -> line_alignment.json
make diff_transcriptions    # classified diffs (6 categories + TEI) -> line_diff.json
make build_line_compare     # Model-compare tab JSON (3 models + confidence)
make frontend               # inspect everything at http://localhost:8000
```

`align_transcriptions` pairs model lines with scholarly lines by fuzzy content
(Needleman–Wunsch, monotonic). `diff_transcriptions` runs the anchored banded
word-level diff and classifies every difference as
abbreviation / orthographic / punctuation / addition / deletion / substitution,
grouped substantive / editorial / scramble, each with its TEI encoding.
`discrepancy_table.py` exports the same diffs as a flat table.

## 7. Analysis-script catalog (`scripts/ocr/`)

Run once per question; results + verdicts are written to
`tests/ocr/evaluations/<name>/` and summarized in `spec.md`. Grouped by theme:

**Error structure**
- `error_distribution.py` — where errors concentrate (per-line CER, per-page maps, % perfect lines).
- `error_map_from_csv.py` — per-page error map for the deployed leaders.
- `error_vs_length.py` — does line length drive error? (No.)
- `lm_error_analysis.py` — char-level error breakdown + **substitution confusion
  tables** (the most common character confusions, e.g. the minim family).
- `kraken_topk_recall.py` — **top-k recall** (is the correct char in the model's
  top-3/5/10 candidates? → the headroom a reranker can recover).
- `assess_line_errors.py`, `assess_line_errors_buckets.py`,
  `assess_pagelevel_diff.py` — manual-sample root-cause bucketing of diffs.

**Image-quality features**
- `run_ink_bleed_detection.py` (+ `merge_ink_bleed_to_manifest.py`) — per-line
  ink-bleed score; bleed is the one robust image feature (hurts CTC most).
- `hard_case_features.py` — what makes a line hard, and for which model
  (bleed / minim run-length / special glyphs; hard-case overlap between models).

**Confidence & calibration** ("does the model know when it's wrong?")
- `confidence_analysis.py` — char-level calibration for kraken/catmus/TrOCR
  (AUROC, ECE, reliability plots). NB: kraken confidence must be the
  peak-frame posterior, not the reported conf field (see spec — decoder bug).
- `catmus_transcribe_conf.py`, `vit_transcribe_conf.py`,
  `medusa_confidence_dump.py` + `confidence_from_dump.py` — per-model
  confidence dumps (CTC per char, seq2seq per token, VLM per token).
- `longtail_confidence.py` — confidence on the worst-decile lines.
- `temperature_scale_kraken.py` — calibration polish (T*≈1.3).
- `manuscript_confidence.py` — whole-manuscript confidence distribution
  (drives the review-queue ranking).

**LM rescoring & post-correction**
- `kraken_lm_rescore.py` / `kraken_lm_tune.py` — per-position char-LM rescorer
  (+ honest λ tuning) — the deployed +0.33pp lever.
- `kraken_ctc_lm.py` — CTC prefix-beam + LM shallow fusion (validated, not needed).
- `kraken_lm_corpus_sweep.py` — external LM corpora (negative).
- `trocr_lm_rescore.py` — N-best rescoring for the seq2seq track.
- `minim_variant_rescore.py` — LM-only minim re-partition (negative — needs
  visual evidence).
- `lexicon_postcorrect.py` — dictionary-based post-correction;
  `run_dictionary_evaluation.py` — DOM-lexicon coverage evaluation.

**Ensembles & routing** (all settled negative — see spec before revisiting)
- `ensemble_oracle.py` — oracle headroom of a per-line 2-model ensemble.
- `confidence_router.py`, `feature_router.py` — confidence-gated / supervised
  routing (don't beat always-kraken).

**Baselines & misc**
- `run_medusa_transcribe.py` + `run_clean_medusa_output.py` — Medusa 9B VLM.
- `paddleocr_recognize.py`, `paddleocr_seg_eval.py` — PaddleOCR baseline.
- `analyze_tokenizer_floor.py` — CER floor of each TrOCR tokenizer.
- `plot_model_results.py` — the consolidated 6-model results figure/tables.
