# Makefile
LAYOUT_YOLO_MODEL_PATH=./models/layout/y8_YALTAi_50epochs_best_+9annotated_fix50.pt
IMAGES_TEST_SET_DIR=./data/processed/annotated_samples/retrain/images
ANNOT_TEST_SET_DIR=./data/processed/annotated_samples/retrain/annotations
ORIGINAL_IMAGES_PATH=./data/raw/original_manuscript/reproduction14453_100
MASKS_PATH=./data/processed/img_layout

PYTHON=python

.PHONY: all evaluate_yolo_performance create_masks clean

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
		--output-path $(MASKS_PATH) 

clean:
	rm -rf $(LOGS_DIR)