# Makefile
# Variables
LAYOUT_YOLO_MODEL_PATH= ./models/layout/y8_YALTAi_50epochs_best_+9annotated_fix50.pt
IMAGES_TEST_SET_DIR= ./data/processed/annotated_samples/retrain/images
ANNOT_TEST_SET_DIR= ./data/processed/annotated_samples/retrain/annotations
PYTHON=python

.PHONY: all evaluate_yolo_performance clean

all: evaluate_yolo_performance

evaluate_yolo_performance:
	$(PYTHON) scripts/data_preprocessing/run_yolo_eval_test_set.py \
		--model-path $(LAYOUT_YOLO_MODEL_PATH) \
		--images-dir $(IMAGES_TEST_SET_DIR) \
		--annotations-dir $(ANNOT_TEST_SET_DIR) 

clean:
	rm -rf $(LOGS_DIR)



