# Makefile
LOGS_DIR ?= ./logs 
#======= Image preprocessing ========
LAYOUT_YOLO_MODEL_PATH=./models/layout/y8_YALTAi_50epochs_best_+9annotated_fix50.pt
IMAGES_TEST_SET_DIR=./data/processed/annotated_samples/retrain/images
ANNOT_TEST_SET_DIR=./data/processed/annotated_samples/retrain/annotations
ORIGINAL_IMAGES_PATH=./data/raw/original_manuscript/reproduction14453_100
MASKS_DIR=./data/processed/img_layout
IMAGES_SEGMENTS = ./data/processed/segmented_images
MASKS_PATH = ./data/processed/img_layout/masks/20260430_115917
PLOTTED_BOUNDS_PATH=./data/processed/plotted_bounds
IMAGES_SEGMENTS_PATH=./data/processed/segmented_images/segmentation_20260430_123217
EXTRACTED_LINES_DIR=./data/processed/extracted_lines
EXTRACTED_LINES_PATH=./data/processed/extracted_lines/extraction_20260430_190006
BINARIZED_IMAGES_DIR=./data/processed/binarized_images
BINARIZED_METHOD?=otsu_gaussian
BINARIZED_IMAGES_PATH=./data/processed/binarized_images/20260430_192414
FILTERED_IMAGES_DIR=./data/processed/filtered_images
FILTERED_ORIGINAL_LINES_PATH=./data/processed/filtered_images/20260430_224958/original/kept
RESIZED_IMAGES_DIR=./data/processed/resized_samples
RESIZING_TARGET_SIZE?=224
#======= Tokenizer ========
RAW_CORPORA_DIR=./data/raw/COMETA_medieval_corpus
TOKENIZER_CORPORA_DIR=./data/processed/tokenizer_corpora
TOKENIZER_DIR=./data/processed/tokenizer
VOCAB_SIZE?=100


PYTHON=python

.PHONY: all evaluate_yolo_performance create_masks segment_images plot_bounds crop_segments binarize_image filter_images resize_images unify_corpora run_tokenizer clean

all: evaluate_yolo_performance

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
		--output-dir $(PLOTTED_BOUNDS_PATH)


crop_segments:
	$(PYTHON) scripts/data_preprocessing/run_crop_image_segments.py \
		--input-folder $(ORIGINAL_IMAGES_PATH) \
		--output-kraken-path $(IMAGES_SEGMENTS_PATH) \
		--output-folder $(EXTRACTED_LINES_DIR)


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
		--dst-base-dir $(FILTERED_IMAGES_DIR)

resize_images:
	$(PYTHON) scripts/data_preprocessing/run_resize_image.py \
			--input-folder $(FILTERED_ORIGINAL_LINES_PATH) \
			--output-folder $(RESIZED_IMAGES_DIR) \
			--target-size $(RESIZING_TARGET_SIZE)


unify_corpora:
	$(PYTHON) scripts/tokenizer/run_unified_corpus.py \
			--input_dir $(RAW_CORPORA_DIR) \
			--output_dir $(TOKENIZER_CORPORA_DIR) \
			--run_name $(RUN_NAME_CORPORA)


run_tokenizer:
	$(PYTHON) scripts/tokenizer/run_BPE_tokenizer.py \
			--input_path $(TOKENIZER_CORPORA_DIR) \
			--output_path $(TOKENIZER_DIR) \
			--vocab_size $(VOCAB_SIZE)


clean:
	rm -rf $(LOGS_DIR)
