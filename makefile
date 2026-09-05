# Makefile
LOGS_DIR ?= ./logs
#======= Image preprocessing ========
LAYOUT_YOLO_MODEL_PATH=./models/layout/y8_YALTAi_50epochs_best_+9annotated_fix50.pt
IMAGES_TEST_SET_DIR=./data/processed/annotated_samples/retrain/images
ANNOT_TEST_SET_DIR=./data/processed/annotated_samples/retrain/annotations
ORIGINAL_IMAGES_PATH=./data/raw/original_manuscript/reproduction14453_100
MASKS_DIR=./data/processed/img_layout
IMAGES_SEGMENTS = ./data/processed/segmented_images
MASKS_PATH = ./data/processed/img_layout/masks/20260515_092830
FONT_SIZE?=50
PLOTTED_BOUNDS_DIR=./data/processed/plotted_bounds
IMAGES_SEGMENTS_PATH=./data/processed/segmented_images/segmentation_20260618_111517

EXTRACTED_LINES_DIR=./data/processed/extracted_lines
CROP_TYPE?=polygon
EXTRACTED_LINES_PATH=./data/processed/extracted_lines/extraction_20260618_154440
BINARIZED_IMAGES_DIR=./data/processed/binarized_images
BINARIZED_METHOD?=otsu_gaussian
BINARIZED_IMAGES_PATH=./data/processed/binarized_images/20260618_155706
FILTERED_IMAGES_DIR=./data/processed/filtered_images
FILTERED_ORIGINAL_LINES_PATH=./data/processed/filtered_images/20260618_160948/original/kept
RESIZED_IMAGES_DIR=./data/processed/resized_samples
RESIZING_TARGET_SIZE?=224
#======= Ink-bleed detection ========
INK_BLEED_OUTPUT_DIR=./data/processed/filtered_images/20260618_160948
INK_BLEED_PERCENTILE?=95
INK_BLEED_W_BG_STD?=0.6
INK_BLEED_W_INTERMEDIATE?=0.4
#======= Tokenizer ========
RAW_CORPORA_DIR=./data/raw/COMETA_medieval_corpus
TOKENIZER_CORPORA_DIR=./data/processed/tokenizer_corpora
TOKENIZER_DIR=./data/processed/tokenizer
TOKENIZER_TYPE?=byte
VOCAB_SIZE?=100
#========= transcription ========
TRANSCRIPTION_DIR=./data/processed/transcription
IMAGE_INVENTORY=./data/processed/filtered_images/20260618_160948/filter_tracking.csv
TRANSCRIPTION_MODEL=./models/ocr/catmus-medieval.mlmodel
#========= dictionary evaluation ========
DICT_PATH=./data/raw/DOM_lemma_variants.json
TRANSCRIPTION_RUN=./data/processed/transcription/ocr_kept_20260515_104644
DICT_EVAL_OUTPUT_DIR=./data/processed/dictionary_eval
#========= synthetic data augmentation ========
COMETA_CORPUS_DIR=./data/raw/COMETA_medieval_corpus
CATEGORIZED_SAMPLES_DIR=./data/processed/synthetic_seeds
CATEGORIZED_SAMPLES_JSON=./data/processed/synthetic_seeds/categorize_20260613_214958/cometa_categorized.json
WORD_PATTERNS?=am,ma
SUBSTRING_PATTERNS?=
# corpus_categorization knobs (override on the command line as needed):
#   CORPUS_DIR        - source directory of *.txt files
#   CUT_TO_LINES=1    - cut each file into OCR-shaped pseudo-lines
#   KEEP_ALL=1        - skip pattern filtering, keep every line
# Output filename is derived from CORPUS_DIR's basename, e.g.
# COMETA_medieval_corpus -> COMETA_medieval_corpus_categorized.json,
# medical_texts          -> medical_texts_categorized.json.
CORPUS_DIR?=$(COMETA_CORPUS_DIR)
OCR_LINE_LENGTHS_JSON=./tests/ocr/ocr_line_lengths_20260625_120230.json
MEDIEVAL_TEXT_DIR=./data/processed/synthetic_text
MEDIEVAL_TEXT_RUN_PATH=./data/processed/synthetic_text/medieval_text_20260613_215219
RENDERING_FONT_PATH=./fonts/merged_font_code_cmpl2.ttf
# When set, the generator scans this directory for *.ttf/*.otf files and
# picks one at random per line, rewriting long-s / rotunda-r in the
# label when the chosen font lacks that glyph. Leave empty to use the
# single RENDERING_FONT_PATH font (the prior behaviour).
RENDERING_FONTS_DIR?=
FONT_RENDER_SIZE?=60
P_LONG_S_BEGIN?=0.95
P_LONG_S_MIDDLE?=0.80
P_ROTUNDA_R?=0.70
P_TIRONIAN_ET?=0.30
ET_STAMP_DIR?=./glyphs/et
C_STAMP_DIR?=./glyphs/C_capitol
P_CAPITAL_E?=0.40
E_STAMP_DIR?=./glyphs/E_capitol
P_ABBREVIATION?=0.10
ABBREV_BASE_DIR?=./glyphs
MAX_ABBREVIATION_PER_LINE?=3
ENABLE_PATTERN_STAMPS?=1
# End-of-line decoration: purely visual, no label contribution.
P_END_DECOR?=0.3
PARCHMENT_PAGES_DIR=./data/raw/original_manuscript/reproduction14453_100
PARCHMENT_CROPS_DIR=./data/processed/synthetic_samples/parchment_crops
PARCHMENT_CROPS_PATH=./data/processed/synthetic_samples/parchment_crops/parchments_20260608_082718
PARCHMENT_MIN_BRIGHTNESS?=100
# Max fraction of saturated-blue pixels allowed in a parchment crop.
# Filters out illuminated-initial frames (blue pigment) that Canny doesn't
# catch. 0.002 = 0.2%; bump higher if you want to be more permissive.
PARCHMENT_MAX_BLUE_FRACTION?=0.002
AUGMENTED_IMAGES_DIR=./data/processed/synthetic_samples/augmented_images
AUGMENTED_RUN_PATH=./data/processed/synthetic_samples/augmented_images/aug_20260613_220436
CORRECTED_LABELS_DIR=./data/processed/synthetic_samples/img_labels
MEDIEVAL_TEXT_LABELS=$(MEDIEVAL_TEXT_RUN_PATH)/labels.json
LABEL_TEXT_FIELD?=original_text
LABEL_SUBSTITUTIONS?=
N_AUGMENTATIONS?=5
BASE_SEED?=42
SAMPLE?=
SAMPLE_SIZE?=5
#========= OCR fine-tuning ========
FINETUNE_BASE_MODEL=./models/ocr/catmus-medieval.mlmodel
FINETUNE_OUTPUT_DIR=./models/ocr/finetuned
FINETUNE_LABELS_JSON=$(CORRECTED_LABELS_DIR)/labels_20260613_220436/labels.json
FINETUNE_VAL_FRACTION?=0.3
FINETUNE_SEED?=42
FINETUNE_LRATE?=1e-5
FINETUNE_BATCH_SIZE?=30
FINETUNE_LAG?=5
FINETUNE_DEVICE?=cpu
FINETUNE_EPOCHS?=-1
# Real-manuscript mix-in for the fine-tune. The "real corrected" folder
# was built from the hand-corrected subset of the validation sample;
# split into n_train + n_val so the model anchors to real data AND its
# val_accuracy reflects real-manuscript performance.
FINETUNE_REAL_FOLDER?=./data/processed/annotated_samples/OCR/full_annotated
FINETUNE_REAL_TRAIN_FRAC?=0.6
FINETUNE_REAL_VAL_FRAC?=0.4
# When set to 1, drop synthetic train entirely — the real folder is the
# whole train + val pool. Use this for Phase 1 (catmus already understands
# generic medieval, so we just anchor it to the specific manuscript hand).
NO_SYNTH_TRAIN?=
# When set to 1, ketos applies per-batch random transforms during training.
# Recommended when the training pool is small (real-only) so the model
# sees more visual variation without needing more annotated lines.
FINETUNE_AUGMENT?=
# When set to 1, keep every per-epoch model_*.mlmodel checkpoint in the
# run dir instead of pruning all but model_best.mlmodel. Useful when you
# want to inspect each epoch's actual val_accuracy (e.g. to verify the
# best-model selection or rebuild a per-epoch curve).
FINETUNE_KEEP_ALL_CHECKPOINTS?=
SMOKE?=
SMOKE_SIZE?=50
SMOKE_EPOCHS?=2
#========= TrOCR (Swin + BERT) fine-tuning ========
# TrOCR-style VisionEncoderDecoderModel: Swin image encoder + mBERT text
# decoder. Trained end-to-end via HuggingFace Seq2SeqTrainer on the same
# real-manuscript pool as the kraken fine-tune so the two runs are
# directly comparable when evaluated against the permanent val set.
TROCR_REAL_FOLDER?=./data/processed/annotated_samples/OCR/full_annotated
# Optional pretrained TrOCR checkpoint. When set, the from-scratch
# Swin+BERT build is skipped and this checkpoint (image processor +
# tokenizer + model) is loaded via from_pretrained. Recommended:
#   microsoft/trocr-base-handwritten  — ViT+RoBERTa; cross-attention
#     pretrained on 34M synthetic + IAM handwriting; roughly matches
#     the classic TrOCR paper.
#   microsoft/trocr-base-stage1       — same architecture, no IAM
#     fine-tune baked in.
# Leave empty to use TROCR_ENCODER_ID + TROCR_DECODER_ID (Swin+BERT
# from-scratch), which underperformed by a large margin — see
# spec.md §6.3.
TROCR_PRETRAINED_MODEL_ID?=
# Optional augmented pool — same one the kraken fine-tune uses so the
# TrOCR and kraken numbers stay apples-to-apples. Set both to empty to
# train on the real folder only:
#   make trocr_finetune TROCR_AUGMENTED_FOLDER= TROCR_LABELS_JSON=
TROCR_AUGMENTED_FOLDER?=$(AUGMENTED_RUN_PATH)
TROCR_LABELS_JSON?=$(FINETUNE_LABELS_JSON)
# Cap on the augmented pool. The kraken aug pool has ~266k pairs — using
# all of them on MPS with swin-base + mBERT would take days per epoch.
# 5000 keeps the training set at ~5600 pairs total (real 600 + 5k aug),
# which matches an effective "×5-ish" augmentation ratio like the kraken
# runs and stays feasible on Apple Silicon. Set to empty to lift the cap.
TROCR_MAX_AUG_SAMPLES?=5000
TROCR_OUTPUT_DIR?=./models/ocr/finetuned
TROCR_ENCODER_ID?=microsoft/swin-base-patch4-window7-224
TROCR_DECODER_ID?=bert-base-multilingual-cased
TROCR_VAL_FRACTION?=0.2
TROCR_SEED?=42
TROCR_EPOCHS?=20
TROCR_LRATE?=5e-5
TROCR_BATCH_SIZE?=8
TROCR_EVAL_BATCH_SIZE?=8
TROCR_MAX_TARGET_LENGTH?=128
TROCR_NUM_BEAMS?=4
TROCR_NO_REPEAT_NGRAM_SIZE?=3
TROCR_LENGTH_PENALTY?=1.0
TROCR_EARLY_STOPPING_PATIENCE?=5
TROCR_DATALOADER_NUM_WORKERS?=0
TROCR_DEVICE?=auto
# TrOCR inference — point at a best_model/ folder from a finished
# fine-tune and a folder of line PNGs to transcribe.
TROCR_MODEL_DIR?=
TROCR_INPUT_DIR?=./data/processed/annotated_samples/OCR/validation
TROCR_TRANSCRIBE_OUTPUT_DIR?=./data/processed/transcription
TROCR_RUN_NAME?=
TROCR_TRANSCRIBE_BATCH_SIZE?=8
TROCR_MAX_NEW_TOKENS?=128
#========= post-training pipeline: alignment, diffs, viewer data ========
# The steps that turn a finished recogniser into viewer-ready data:
#   1. align_scholarly_edition  scholarly edition -> manuscript lineation (one-off)
#   2. kraken_lm_transcribe / trocr_transcribe_conf   full-corpus transcription
#   3. conf_to_txt              per-token conf JSON -> per-line txt layout
#   4. align_transcriptions     model lines <-> scholarly lines (content-based)
#   5. diff_transcriptions      classified discrepancies (six categories + TEI)
#   6. build_line_compare       Model-compare tab JSON (3 models + confidence)
# Raw scholarly edition + the auxiliary OCR that lends it the manuscript's
# line breaks (the one-off §6.4 alignment; re-run only if either changes).
SCHOLARLY_REFERENCE_TXT=./data/raw/AlbucE.txt
SCHOLARLY_ALIGN_OCR_DIR=./data/processed/transcription/ocr_kept_20260622_120413
SCHOLARLY_ALIGN_OUTPUT_ROOT=./tests/ocr
# The frozen aligned edition every downstream step consumes.
SCHOLARLY_TXT=./tests/ocr/AlbucE_aligned_20260628_142959.txt
# Which model's full-corpus transcription to align/diff (per-page dirs of
# <page>_line_<N>.txt). line_alignment.json / line_diff.json are written
# INTO this dir — the viewer picks them up from there.
MODEL_TRANSCRIPTION_DIR?=./data/processed/transcription/vitlightreal_full_corpus
DIFF_METHOD?=banded
# kraken deployed pipeline (CTC + per-position char-LM) full-corpus run.
KRAKEN_LM_MODEL?=./models/ocr/finetuned/finetune_20260806_123435/model_best.mlmodel
KRAKEN_LM_DIR?=./data/processed/annotated_samples/OCR/full_annotated
KRAKEN_LM_RUN_NAME?=krakenLM_full_corpus
KRAKEN_LM_LAMBDA?=0.2
# TrOCR per-token-confidence full-corpus run (feeds tab 4 + conf_to_txt).
TROCR_CONF_MODEL_DIR?=./models/vit_lightreal_med4k/trocr_20260823_073535/best_model
TROCR_CONF_OUT_DIR?=./data/processed/transcription/vitlightreal_conf_fullms
TROCR_CONF_DEVICE?=mps
TROCR_CONF_BATCH_SIZE?=16
# Model-compare tab inputs (per-model confidence dumps) + output.
CATMUS_CONF_DIR?=./data/processed/transcription/catmus_conf_fullms
KRAKEN_CONF_DIR?=./data/processed/transcription/krakenleader_conf_fullms
LINE_COMPARE_OUT_DIR?=./data/processed/line_compare
#========= annotation batch sampling ========
# Source of line PNGs — the manually-filtered/corrected crops the OCR
# pipeline actually ran on (NOT the raw extraction folder, which still
# has un-fixed double-column crops etc.).
SAMPLE_SOURCE_LINES?=./data/processed/filtered_images/20260618_160948/original/kept
SAMPLE_OCR_SEED?=./data/processed/transcription/ocr_kept_20260622_120413
# Space-separated list of folders to exclude — every <stem>.gt.txt across
# all listed folders is dropped from the eligible pool. Non-existent
# folders are skipped with a warning (so this works during bootstrap
# when e.g. the validation set hasn't been created yet).
SAMPLE_EXCLUDES?=./data/processed/annotated_samples/OCR/full_annotated ./data/processed/annotated_samples/OCR/validation
SAMPLE_OUTPUT_ROOT?=./tests/ocr
# 'real_val_sample' matches historic training batches; override to
# 'validation' when sampling the permanent held-out benchmark set.
SAMPLE_OUTPUT_PREFIX?=real_val_sample
SAMPLE_N_TARGET?=100
# Increment for each new batch — prior batches used 42-48. Seed=100 is
# reserved for the initial validation set draw so it can never collide
# with a training-batch seed.
SAMPLE_SEED?=49
# Optional regex to target specific characters. Empty = every non-empty
# OCR seed is eligible. Example for capital C or E (word-initial):
#   SAMPLE_PATTERN='(?<![A-Za-z])[CE]' SAMPLE_PATTERN_LABEL='capital C or E targeted'
SAMPLE_PATTERN?=
SAMPLE_PATTERN_LABEL?=

PYTHON=uv run python

.PHONY: all setup-precommit evaluate_yolo_performance create_masks segment_images plot_bounds crop_segments binarize_image filter_images resize_images detect_ink_bleed unify_corpora run_tokenizer run_transcription run_dictionary_eval corpus_categorization medieval_text_generation extract_parchment_crops augmentation_techniques correct_labels kraken_finetune finetune_ocr trocr_finetune trocr_transcribe align_scholarly_edition kraken_lm_transcribe trocr_transcribe_conf conf_to_txt align_transcriptions diff_transcriptions build_line_compare sample_annotation_batch frontend clean

all: evaluate_yolo_performance

setup-precommit:
	uv sync
	uv run pre-commit install
	uv run pre-commit run --all-files || true

evaluate_yolo_performance:
	$(PYTHON) scripts/data_preprocessing/run_yolo_eval_test_set.py \
		--model-path $(LAYOUT_YOLO_MODEL_PATH) \
		--images-dir $(IMAGES_TEST_SET_DIR) \
		--annotations-dir $(ANNOT_TEST_SET_DIR)

create_masks:
	$(PYTHON) scripts/data_preprocessing/run_yolo_masks.py \
		--model-path $(LAYOUT_YOLO_MODEL_PATH) \
		--images-path $(ORIGINAL_IMAGES_PATH) \
		--output-path $(MASKS_DIR)

segment_images:
	$(PYTHON) scripts/data_preprocessing/run_image_segmentation.py \
		--input-folder $(ORIGINAL_IMAGES_PATH) \
		--output-folder $(IMAGES_SEGMENTS) \
		--masks-folder $(MASKS_PATH)

plot_bounds:
	$(PYTHON) scripts/data_preprocessing/run_plot_bounds.py \
		--input-dir $(ORIGINAL_IMAGES_PATH) \
		--kraken-output-path $(IMAGES_SEGMENTS_PATH) \
		--output-dir $(PLOTTED_BOUNDS_DIR) \
		--font-size $(FONT_SIZE)


crop_segments:
	$(PYTHON) scripts/data_preprocessing/run_crop_image_segments.py \
		--input-folder $(ORIGINAL_IMAGES_PATH) \
		--output-kraken-path $(IMAGES_SEGMENTS_PATH) \
		--output-folder $(EXTRACTED_LINES_DIR)  \
		--crop-type $(CROP_TYPE)


binarize_image:
	$(PYTHON) scripts/data_preprocessing/run_binarize_images.py \
		--input-path $(EXTRACTED_LINES_PATH) \
		--output-base-dir $(BINARIZED_IMAGES_DIR) \
		--method $(BINARIZED_METHOD)
		#make binarize_image BINARIZED_METHOD=otsu


filter_images:
	$(PYTHON) scripts/data_preprocessing/run_filtering_noisy_images.py \
		--binarized-src $(BINARIZED_IMAGES_PATH) \
		--extracted-src $(EXTRACTED_LINES_PATH) \
		--dst-base-dir $(FILTERED_IMAGES_DIR) \
		--kraken-json-dir $(IMAGES_SEGMENTS_PATH)

resize_images:
	$(PYTHON) scripts/data_preprocessing/run_resize_image.py \
			--input-folder $(FILTERED_ORIGINAL_LINES_PATH) \
			--output-folder $(RESIZED_IMAGES_DIR) \
			--target-size $(RESIZING_TARGET_SIZE)


# make detect_ink_bleed                              # default p75 (top 25% flagged)
# make detect_ink_bleed INK_BLEED_PERCENTILE=90     # stricter (only top 10%)
# make detect_ink_bleed INK_BLEED_W_BG_STD=0.5 INK_BLEED_W_INTERMEDIATE=0.5
detect_ink_bleed:
	$(PYTHON) scripts/data_preprocessing/run_ink_bleed_detection.py \
			--input-folder $(FILTERED_ORIGINAL_LINES_PATH) \
			--output-base-dir $(INK_BLEED_OUTPUT_DIR) \
			--bleed-percentile $(INK_BLEED_PERCENTILE) \
			--w-bg-std $(INK_BLEED_W_BG_STD) \
			--w-intermediate $(INK_BLEED_W_INTERMEDIATE)


unify_corpora:
	$(PYTHON) scripts/tokenizer/run_unified_corpus.py \
			--input_dir $(RAW_CORPORA_DIR) \
			--output_dir $(TOKENIZER_CORPORA_DIR) \
			--run_name $(RUN_NAME_CORPORA)


run_tokenizer:
	$(PYTHON) scripts/tokenizer/run_BPE_tokenizer.py \
			--input_path $(TOKENIZER_CORPORA_DIR) \
			--output_path $(TOKENIZER_DIR) \
			--type $(TOKENIZER_TYPE) \
			--vocab_size $(VOCAB_SIZE)


run_transcription:
	$(PYTHON) scripts/ocr/run_transcribe_img.py \
			--seg-path $(IMAGES_SEGMENTS_PATH) \
			--input-img-dir $(FILTERED_ORIGINAL_LINES_PATH) \
			--output-dir $(TRANSCRIPTION_DIR) \
			--img-inventory $(IMAGE_INVENTORY)\
			--model-path $(TRANSCRIPTION_MODEL)

run_dictionary_eval:
	$(PYTHON) scripts/ocr/run_dictionary_evaluation.py \
			--transcription-dir $(TRANSCRIPTION_RUN) \
			--dictionary-path $(DICT_PATH) \
			--output-dir $(DICT_EVAL_OUTPUT_DIR)


# Usage examples:
#   make corpus_categorization
#       COMETA, manuscript-style line breaks, pattern filtering on (default).
#   make corpus_categorization CORPUS_DIR=./data/raw/medical_texts CUT_TO_LINES=1 KEEP_ALL=1
#       Paragraph corpus, cut into OCR-shaped pseudo-lines, keep every line.
#   make corpus_categorization CORPUS_DIR=./data/raw/medical_texts CUT_TO_LINES=1
#       Paragraph corpus, cut into pseudo-lines, then keep only pattern matches.
corpus_categorization:
	$(PYTHON) scripts/data_augmentation/run_corpus_categorization.py \
			--corpus-dir $(CORPUS_DIR) \
			--output-dir $(CATEGORIZED_SAMPLES_DIR) \
			--word-patterns $(WORD_PATTERNS) \
			--substring-patterns "$(SUBSTRING_PATTERNS)" \
			--output-filename "$(notdir $(patsubst %/,%,$(CORPUS_DIR)))_categorized.json" \
			$(if $(CUT_TO_LINES),--cut-to-lines --line-lengths-json $(OCR_LINE_LENGTHS_JSON)) \
			$(if $(KEEP_ALL),--keep-all)


medieval_text_generation:
	$(PYTHON) scripts/data_augmentation/run_medieval_text_generation.py \
			--input-json $(CATEGORIZED_SAMPLES_JSON) \
			--output-dir $(MEDIEVAL_TEXT_DIR) \
			--font-path $(RENDERING_FONT_PATH) \
			$(if $(RENDERING_FONTS_DIR),--fonts-dir $(RENDERING_FONTS_DIR)) \
			--font-size $(FONT_RENDER_SIZE) \
			--p-long-s-begin $(P_LONG_S_BEGIN) \
			--p-long-s-middle $(P_LONG_S_MIDDLE) \
			--p-rotunda-r $(P_ROTUNDA_R) \
			--p-tironian-et $(P_TIRONIAN_ET) \
			--et-stamp-dir $(ET_STAMP_DIR) \
			--c-stamp-dir $(C_STAMP_DIR) \
			--p-capital-e $(P_CAPITAL_E) \
			--e-stamp-dir $(E_STAMP_DIR) \
			--p-abbreviation $(P_ABBREVIATION) \
			--abbrev-base-dir $(ABBREV_BASE_DIR) \
			--max-abbreviation-per-line $(MAX_ABBREVIATION_PER_LINE) \
			$(if $(ENABLE_PATTERN_STAMPS),--enable-pattern-stamps) \
			--p-end-decor $(P_END_DECOR) \
			--base-seed $(BASE_SEED)


extract_parchment_crops:
	$(PYTHON) scripts/data_augmentation/run_augmentation_techniques.py \
			--input-folder $(PARCHMENT_PAGES_DIR) \
			--output-folder $(PARCHMENT_CROPS_DIR) \
			--min-brightness $(PARCHMENT_MIN_BRIGHTNESS) \
			--max-blue-fraction $(PARCHMENT_MAX_BLUE_FRACTION)


# Full run (default — same as before)
# make augmentation_techniques
# Quick 5-image preview to sanity-check
# make augmentation_techniques SAMPLE=1
# Custom preview size
# make augmentation_techniques SAMPLE=1 SAMPLE_SIZE=3

augmentation_techniques:
	$(PYTHON) scripts/data_augmentation/run_augment_images.py \
			--input-folder $(MEDIEVAL_TEXT_RUN_PATH) \
			--output-folder $(AUGMENTED_IMAGES_DIR) \
			--parchment-folder $(PARCHMENT_CROPS_PATH) \
			--n-augmentations $(N_AUGMENTATIONS) \
			--seed $(BASE_SEED) \
			$(if $(SAMPLE),--sample --sample-size $(SAMPLE_SIZE))



# make correct_labels                                    # uses defaults
# make correct_labels LABEL_SUBSTITUTIONS=v:u,j:i        # extend later
# make correct_labels LABEL_TEXT_FIELD=medieval_text     # keep long-s

correct_labels:
	$(PYTHON) scripts/data_augmentation/run_label_correction.py \
			--input-json $(MEDIEVAL_TEXT_LABELS) \
			--augmented-folder $(AUGMENTED_RUN_PATH) \
			--output-base-dir $(CORRECTED_LABELS_DIR) \
			--text-field $(LABEL_TEXT_FIELD) \
			$(if $(LABEL_SUBSTITUTIONS),--substitutions "$(LABEL_SUBSTITUTIONS)")


# Fine-tune the KRAKEN (ketos/CTC) recogniser — this target is kraken-only;
# the seq2seq counterpart is `trocr_finetune` below.
# make kraken_finetune                                # full run, early stopping
# make kraken_finetune SMOKE=1                        # smoke test: 50 lines, 2 epochs
# make kraken_finetune SMOKE=1 SMOKE_SIZE=20 SMOKE_EPOCHS=1
# make kraken_finetune FINETUNE_DEVICE=cuda:0         # use GPU if available

#KETOS_EARLY_STOP_MIN_DELTA=0.001 make kraken_finetune FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps
# Stricter — requires 0.1pp improvement per epoch

#KETOS_EARLY_STOP_MIN_DELTA=0.0 make kraken_finetune FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps
# Reverts to the old Lightning default (any positive change counts)


#PYTORCH_ENABLE_MPS_FALLBACK=1 make kraken_finetune FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps

kraken_finetune:
	$(PYTHON) scripts/ocr/run_finetune_ocr.py \
			$(if $(NO_SYNTH_TRAIN),,--augmented-folder $(AUGMENTED_RUN_PATH)) \
			$(if $(NO_SYNTH_TRAIN),,--labels-json $(FINETUNE_LABELS_JSON)) \
			$(if $(NO_SYNTH_TRAIN),--no-synth-train) \
			$(if $(FINETUNE_AUGMENT),--augment) \
			$(if $(FINETUNE_KEEP_ALL_CHECKPOINTS),--keep-all-checkpoints) \
			--base-model $(FINETUNE_BASE_MODEL) \
			--output-base-dir $(FINETUNE_OUTPUT_DIR) \
			--val-fraction $(FINETUNE_VAL_FRACTION) \
			--seed $(FINETUNE_SEED) \
			--lrate $(FINETUNE_LRATE) \
			--batch-size $(FINETUNE_BATCH_SIZE) \
			--lag $(FINETUNE_LAG) \
			--device $(FINETUNE_DEVICE) \
			--epochs $(FINETUNE_EPOCHS) \
			--real-folder $(FINETUNE_REAL_FOLDER) \
			--real-train-frac $(FINETUNE_REAL_TRAIN_FRAC) \
			--real-val-frac $(FINETUNE_REAL_VAL_FRAC) \
			$(if $(SMOKE),--smoke --smoke-size $(SMOKE_SIZE) --smoke-epochs $(SMOKE_EPOCHS))

# Deprecated alias — kept so old notes/scripts keep working. Use kraken_finetune.
finetune_ocr: kraken_finetune


# Fine-tune a Swin+BERT VisionEncoderDecoderModel on the real
# annotated-manuscript pool. Examples:
#   make trocr_finetune                                # defaults: swin-base + mBERT, 20 epochs, MPS auto
#   make trocr_finetune TROCR_EPOCHS=40 TROCR_BATCH_SIZE=4
#   PYTORCH_ENABLE_MPS_FALLBACK=1 make trocr_finetune  # required if any op falls back to CPU on MPS
trocr_finetune:
	$(PYTHON) scripts/ocr/run_trocr_finetune.py \
			--real-folder $(TROCR_REAL_FOLDER) \
			$(if $(TROCR_PRETRAINED_MODEL_ID),--pretrained-model-id $(TROCR_PRETRAINED_MODEL_ID)) \
			$(if $(TROCR_AUGMENTED_FOLDER),--augmented-folder $(TROCR_AUGMENTED_FOLDER)) \
			$(if $(TROCR_LABELS_JSON),--labels-json $(TROCR_LABELS_JSON)) \
			$(if $(TROCR_MAX_AUG_SAMPLES),--max-aug-samples $(TROCR_MAX_AUG_SAMPLES)) \
			--output-base-dir $(TROCR_OUTPUT_DIR) \
			--encoder-id $(TROCR_ENCODER_ID) \
			--decoder-id $(TROCR_DECODER_ID) \
			--val-fraction $(TROCR_VAL_FRACTION) \
			--seed $(TROCR_SEED) \
			--epochs $(TROCR_EPOCHS) \
			--learning-rate $(TROCR_LRATE) \
			--batch-size $(TROCR_BATCH_SIZE) \
			--eval-batch-size $(TROCR_EVAL_BATCH_SIZE) \
			--max-target-length $(TROCR_MAX_TARGET_LENGTH) \
			--num-beams $(TROCR_NUM_BEAMS) \
			--no-repeat-ngram-size $(TROCR_NO_REPEAT_NGRAM_SIZE) \
			--length-penalty $(TROCR_LENGTH_PENALTY) \
			--early-stopping-patience $(TROCR_EARLY_STOPPING_PATIENCE) \
			--dataloader-num-workers $(TROCR_DATALOADER_NUM_WORKERS) \
			--device $(TROCR_DEVICE)


# Transcribe a folder of line PNGs with a fine-tuned TrOCR checkpoint.
# Point TROCR_MODEL_DIR at the best_model/ subfolder from a completed
# trocr_finetune run. Example:
#   make trocr_transcribe \
#        TROCR_MODEL_DIR=./models/ocr/finetuned/trocr_20260710_183045/best_model \
#        TROCR_INPUT_DIR=./data/processed/annotated_samples/OCR/validation \
#        TROCR_RUN_NAME=trocr_vs_validation_300
trocr_transcribe:
	$(PYTHON) scripts/ocr/run_trocr_transcribe.py \
			--model-dir $(TROCR_MODEL_DIR) \
			--input-dir $(TROCR_INPUT_DIR) \
			--output-dir $(TROCR_TRANSCRIBE_OUTPUT_DIR) \
			$(if $(TROCR_RUN_NAME),--run-name $(TROCR_RUN_NAME)) \
			--device $(TROCR_DEVICE) \
			--batch-size $(TROCR_TRANSCRIBE_BATCH_SIZE) \
			--max-new-tokens $(TROCR_MAX_NEW_TOKENS) \
			--num-beams $(TROCR_NUM_BEAMS)


# ======= Post-training pipeline: alignment, diffs, viewer data =======

# One-off §6.4 alignment: break the continuous scholarly edition at the
# manuscript's line boundaries, guided by an auxiliary OCR transcription.
# Lossless (output re-concatenates to the reference exactly); verified.
#   make align_scholarly_edition
#   make align_scholarly_edition SCHOLARLY_ALIGN_OCR_DIR=./data/processed/transcription/<other>
align_scholarly_edition:
	$(PYTHON) scripts/ocr/run_scholarly_alignment.py \
			--reference-txt $(SCHOLARLY_REFERENCE_TXT) \
			--ocr-dir $(SCHOLARLY_ALIGN_OCR_DIR) \
			--output-root $(SCHOLARLY_ALIGN_OUTPUT_ROOT)


# Full-corpus transcription with the DEPLOYED kraken pipeline
# (CTC + per-position char-LM rescore, λ=0.2 — spec §6.13 P1).
#   make kraken_lm_transcribe
kraken_lm_transcribe:
	$(PYTHON) scripts/ocr/kraken_lm_transcribe.py \
			--input-dir $(FILTERED_ORIGINAL_LINES_PATH) \
			--model-path $(KRAKEN_LM_MODEL) \
			--lm-dir $(KRAKEN_LM_DIR) \
			--run-name $(KRAKEN_LM_RUN_NAME) \
			--output-dir $(TRANSCRIPTION_DIR) \
			--lam $(KRAKEN_LM_LAMBDA)


# Full-corpus TrOCR transcription WITH per-token confidence (greedy) —
# feeds the Model-compare tab and, via conf_to_txt, tabs 1-2.
#   make trocr_transcribe_conf
trocr_transcribe_conf:
	$(PYTHON) scripts/ocr/vit_transcribe_conf.py \
			--input-dir $(FILTERED_ORIGINAL_LINES_PATH) \
			--model-dir $(TROCR_CONF_MODEL_DIR) \
			--out-dir $(TROCR_CONF_OUT_DIR) \
			--device $(TROCR_CONF_DEVICE) \
			--batch-size $(TROCR_CONF_BATCH_SIZE) \
			--num-beams 1


# Derive the per-page txt layout (<page>/<page>_line_<N>.txt + _full.txt)
# from a confidence-JSON dump — one decode pass serves both consumers.
#   make conf_to_txt
conf_to_txt:
	$(PYTHON) scripts/ocr/conf_json_to_txt.py \
			--conf-dir $(TROCR_CONF_OUT_DIR) \
			--out-dir $(MODEL_TRANSCRIPTION_DIR)


# Content-based line alignment: model lines <-> scholarly lines (§6.6).
# Writes line_alignment.json INTO the model dir (viewer + banded diff read it).
#   make align_transcriptions MODEL_TRANSCRIPTION_DIR=./data/processed/transcription/<model>
align_transcriptions:
	$(PYTHON) scripts/ocr/align_transcriptions.py \
			--model-dir $(MODEL_TRANSCRIPTION_DIR) \
			--scholarly-txt $(SCHOLARLY_TXT) \
			--output $(MODEL_TRANSCRIPTION_DIR)/line_alignment.json


# Classified model-vs-scholarly discrepancies (§6.7: six categories + TEI),
# anchored banded word-NW by default. Writes line_diff.json INTO the model dir.
# Requires line_alignment.json (run align_transcriptions first).
#   make diff_transcriptions MODEL_TRANSCRIPTION_DIR=./data/processed/transcription/<model>
diff_transcriptions:
	$(PYTHON) scripts/ocr/diff_transcriptions.py \
			--model-dir $(MODEL_TRANSCRIPTION_DIR) \
			--scholarly-txt $(SCHOLARLY_TXT) \
			--output $(MODEL_TRANSCRIPTION_DIR)/line_diff.json \
			--method $(DIFF_METHOD)


# Per-page comparison JSON for the Model-compare tab (§7.4.1): scholarly +
# catmus + kraken-leader + TrOCR-leader with per-char/token confidence.
#   make build_line_compare
build_line_compare:
	$(PYTHON) scripts/ocr/build_line_compare.py \
			--catmus-dir $(CATMUS_CONF_DIR) \
			--kraken-dir $(KRAKEN_CONF_DIR) \
			--vit-dir $(TROCR_CONF_OUT_DIR) \
			--scholarly-txt $(SCHOLARLY_TXT) \
			--out-dir $(LINE_COMPARE_OUT_DIR)


# Sample a fresh annotation batch. Examples:
#   make sample_annotation_batch                                # 100 lines, seed 49, no content filter
#   make sample_annotation_batch SAMPLE_SEED=50 SAMPLE_N_TARGET=50
#   make sample_annotation_batch \
#        SAMPLE_PATTERN='(?<![A-Za-z])[CE]' \
#        SAMPLE_PATTERN_LABEL='capital C or E targeted' \
#        SAMPLE_SEED=50                                         # 100 lines with a word-initial C or E
#   make sample_annotation_batch \
#        SAMPLE_N_TARGET=300 SAMPLE_SEED=100 \
#        SAMPLE_OUTPUT_PREFIX=validation                        # permanent held-out validation set
sample_annotation_batch:
	$(PYTHON) scripts/data_preprocessing/run_sample_annotation_batch.py \
			--source-lines-dir $(SAMPLE_SOURCE_LINES) \
			--ocr-seed-dir $(SAMPLE_OCR_SEED) \
			$(foreach f,$(SAMPLE_EXCLUDES),--exclude-folder $(f)) \
			--output-root $(SAMPLE_OUTPUT_ROOT) \
			--output-subfolder-prefix $(SAMPLE_OUTPUT_PREFIX) \
			--n-target $(SAMPLE_N_TARGET) \
			--seed $(SAMPLE_SEED) \
			$(if $(SAMPLE_PATTERN),--pattern "$(SAMPLE_PATTERN)") \
			$(if $(SAMPLE_PATTERN_LABEL),--pattern-label "$(SAMPLE_PATTERN_LABEL)")


# Launch the AlbucE manuscript viewer (FastAPI + static HTML/JS).
# Serves at http://localhost:$(FRONTEND_PORT). Override paths via env:
#   VIEWER_MODEL_TRANSCRIPTION=./data/processed/transcription/<new> make frontend
FRONTEND_HOST?=127.0.0.1
FRONTEND_PORT?=8000
frontend:
	PROJECT_ROOT=. $(PYTHON) -m uvicorn frontend.app:app \
			--host $(FRONTEND_HOST) --port $(FRONTEND_PORT) --reload


clean:
	rm -rf $(LOGS_DIR)
