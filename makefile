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
# medical_texts_categorization: paragraph corpus cut into OCR-shaped pseudo-lines
MEDICAL_TEXTS_CORPUS_DIR=./data/raw/medical_texts
OCR_LINE_LENGTHS_JSON=./tests/ocr/ocr_line_lengths_20260625_120230.json
MEDIEVAL_TEXT_DIR=./data/processed/synthetic_text
MEDIEVAL_TEXT_RUN_PATH=./data/processed/synthetic_text/medieval_text_20260613_215219
RENDERING_FONT_PATH=./fonts/merged_font_code_cmpl2.ttf
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
SMOKE?=
SMOKE_SIZE?=50
SMOKE_EPOCHS?=2

PYTHON=uv run python

.PHONY: all setup-precommit evaluate_yolo_performance create_masks segment_images plot_bounds crop_segments binarize_image filter_images resize_images detect_ink_bleed unify_corpora run_tokenizer run_transcription run_dictionary_eval corpus_categorization medical_texts_categorization medieval_text_generation extract_parchment_crops augmentation_techniques correct_labels finetune_ocr clean

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


corpus_categorization:
	$(PYTHON) scripts/data_augmentation/run_corpus_categorization.py \
			--corpus-dir $(COMETA_CORPUS_DIR) \
			--output-dir $(CATEGORIZED_SAMPLES_DIR) \
			--word-patterns $(WORD_PATTERNS) \
			--substring-patterns "$(SUBSTRING_PATTERNS)"


# Paragraph-style medical_texts corpus: cut into OCR-shaped pseudo-lines
# (length sampled from the empirical OCR per-line distribution) and keep
# every line — no pattern filtering.
medical_texts_categorization:
	$(PYTHON) scripts/data_augmentation/run_corpus_categorization.py \
			--corpus-dir $(MEDICAL_TEXTS_CORPUS_DIR) \
			--output-dir $(CATEGORIZED_SAMPLES_DIR) \
			--cut-to-lines \
			--line-lengths-json $(OCR_LINE_LENGTHS_JSON) \
			--keep-all \
			--output-filename medical_texts_categorized.json


medieval_text_generation:
	$(PYTHON) scripts/data_augmentation/run_medieval_text_generation.py \
			--input-json $(CATEGORIZED_SAMPLES_JSON) \
			--output-dir $(MEDIEVAL_TEXT_DIR) \
			--font-path $(RENDERING_FONT_PATH) \
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


# make finetune_ocr                                # full run, early stopping
# make finetune_ocr SMOKE=1                        # smoke test: 50 lines, 2 epochs
# make finetune_ocr SMOKE=1 SMOKE_SIZE=20 SMOKE_EPOCHS=1
# make finetune_ocr FINETUNE_DEVICE=cuda:0         # use GPU if available

#KETOS_EARLY_STOP_MIN_DELTA=0.001 make finetune_ocr FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps
# Stricter — requires 0.1pp improvement per epoch

#KETOS_EARLY_STOP_MIN_DELTA=0.0 make finetune_ocr FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps
# Reverts to the old Lightning default (any positive change counts)


#PYTORCH_ENABLE_MPS_FALLBACK=1 make finetune_ocr FINETUNE_EPOCHS=150 FINETUNE_DEVICE=mps

finetune_ocr:
	$(PYTHON) scripts/ocr/run_finetune_ocr.py \
			--augmented-folder $(AUGMENTED_RUN_PATH) \
			--labels-json $(FINETUNE_LABELS_JSON) \
			--base-model $(FINETUNE_BASE_MODEL) \
			--output-base-dir $(FINETUNE_OUTPUT_DIR) \
			--val-fraction $(FINETUNE_VAL_FRACTION) \
			--seed $(FINETUNE_SEED) \
			--lrate $(FINETUNE_LRATE) \
			--batch-size $(FINETUNE_BATCH_SIZE) \
			--lag $(FINETUNE_LAG) \
			--device $(FINETUNE_DEVICE) \
			--epochs $(FINETUNE_EPOCHS) \
			$(if $(SMOKE),--smoke --smoke-size $(SMOKE_SIZE) --smoke-epochs $(SMOKE_EPOCHS))


clean:
	rm -rf $(LOGS_DIR)
