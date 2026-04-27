import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import os
import ast
from dotenv import load_dotenv

import json
import logging
from datetime import datetime

def setup_evaluation_logging(logs_dir, run_name=None):
    """
    Setup logging directory and file for evaluation run
    Returns: path to log file
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = Path(logs_dir) / f"{run_name}_evaluation.log"
    metrics_file = Path(logs_dir) / f"{run_name}_metrics.json"
    
    # Configure file logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return str(log_file), str(metrics_file)

def parse_alto_xml_boxes_only(xml_path):
    """
    Parse ALTO XML and extract bounding boxes
    Returns: List of [x_min, y_min, x_max, y_max]
    """
    NS = {'alto': 'http://www.loc.gov/standards/alto/ns-v4#'}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    textblocks = root.findall('.//alto:TextBlock', NS)
    
    for tb in textblocks:
        hpos = int(tb.get('HPOS'))
        vpos = int(tb.get('VPOS'))
        width = int(tb.get('WIDTH'))
        height = int(tb.get('HEIGHT'))
        
        x_min = hpos
        y_min = vpos
        x_max = hpos + width
        y_max = vpos + height
        
        boxes.append([x_min, y_min, x_max, y_max])
    
    return boxes


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold):
    """
    Match predictions to ground truth using IoU (ignoring classes)
    Returns: (tp_count, fp_count, fn_count)
    """
    tp = 0
    fp = 0
    fn = 0
    
    gt_matched = [False] * len(gt_boxes)
    
    # Sort predictions by confidence (descending)
    pred_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    for pred in pred_sorted:
        pred_box = pred[:4]
        pred_conf = pred[4]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching GT box (any class)
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match meets threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    # Count unmatched GT boxes as false negatives
    fn = sum(1 for m in gt_matched if not m)
    
    return tp, fp, fn


def calculate_precision_at_iou(gt_boxes, pred_boxes, iou_threshold):
    """Calculate precision at a specific IoU threshold"""
    tp, fp, fn = match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, tp, fp, fn


def evaluate_detection_only(
    model_path,
    test_images_dir,
    test_annotations_dir,
    iou_thresholds=[0.5],
    conf_threshold=0.25,
    logs_dir=None,
    run_name=None 
):
    """
    Evaluate YOLO model for region detection
    Returns: Dictionary with mPA (mean Precision at IoU) 
    """
    if logs_dir:
        log_file, metrics_file = setup_evaluation_logging(logs_dir, run_name)
        logger = logging.getLogger(__name__)
        logger.info(f"=== Evaluation Run Started ===")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Metrics file: {metrics_file}")
    else:
        logger = None
    
    # Log model and evaluation arguments
    eval_config = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "test_images_dir": test_images_dir,
        "test_annotations_dir": test_annotations_dir,
        "iou_thresholds": iou_thresholds,
        "conf_threshold": conf_threshold,
        "imgsz": 640,
        "environment": {
            "SEGMONTO_LIST": os.environ.get("SEGMONTO_LIST"),
            "PROJECT_ROOT": os.environ.get("PROJECT_ROOT")}
    }
    
    if logger:
        logger.info(f"Evaluation Configuration: {json.dumps(eval_config, indent=2)}")
    

    model = YOLO(model_path)
    results_per_iou = {iou: {'tp': 0, 'fp': 0, 'fn': 0} for iou in iou_thresholds}
    
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [
        p for p in Path(test_images_dir).iterdir() 
        if p.suffix.lower() in image_extensions
    ]
    if logger:
        logger.info(f"Evaluating {len(test_images)} images")
    print(f"Evaluating {len(test_images)} images")
    
    for img_path in tqdm(test_images, desc="Processing"):
        
        xml_path = Path(test_annotations_dir) / img_path.with_suffix('.xml').name
        if not xml_path.exists():
            if logger:
                logger.debug(f"Skipping {img_path.name}: no annotation XML")
            continue
        
        gt_boxes = parse_alto_xml_boxes_only(xml_path)
        
        
        results = model.predict(
            str(img_path),
            imgsz=640,
            conf=conf_threshold,
            save=True,
            show=False,
            verbose=False
        )
        
        load_dotenv()
        SEGMONTO_LIST = ast.literal_eval(os.environ.get("SEGMONTO_LIST"))

        pred_boxes = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()  
            for i in range(len(boxes)):
                if classes[i] == SEGMONTO_LIST.index("MainZone"):
                    pred_boxes.append([
                        boxes[i][0], boxes[i][1], 
                        boxes[i][2], boxes[i][3], 
                        confs[i]  # confidence 
                    ])
        
        
        for iou_thresh in iou_thresholds:
            tp, fp, fn = match_predictions_to_gt(gt_boxes, pred_boxes, iou_thresh)
            results_per_iou[iou_thresh]['tp'] += tp
            results_per_iou[iou_thresh]['fp'] += fp
            results_per_iou[iou_thresh]['fn'] += fn
    
    if logger:
        logger.info("DETECTION-ONLY EVALUATION RESULTS (Class-Agnostic)")
    
    print("DETECTION-ONLY EVALUATION RESULTS (Class-Agnostic)")
    metrics = {}
    
    for iou_thresh in iou_thresholds:
        tp = results_per_iou[iou_thresh]['tp']
        fp = results_per_iou[iou_thresh]['fp']
        fn = results_per_iou[iou_thresh]['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # mPA = mean Precision at this IoU (same as precision here since single "class")
        mPA = precision
        
        metrics[f'mPA@{iou_thresh}'] = mPA
        metrics[f'Precision@{iou_thresh}'] = precision
        metrics[f'Recall@{iou_thresh}'] = recall
        metrics[f'F1@{iou_thresh}'] = f1

        result_str = (
            f"IoU Threshold: {iou_thresh} | "
            f"mPA@{iou_thresh}: {mPA:.4f} | "
            f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f} | "
            f"TP:{tp} FP:{fp} FN:{fn}"
        )
        
        if logger:
            logger.info(result_str)
        print(result_str)

    if logger and metrics_file:
        metrics["evaluation_config"] = eval_config
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
    
    if logger:
        logger.info(f"=== Evaluation Run Completed ===")   
        

    return metrics


if __name__ == "__main__":
    load_dotenv()
    PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "layout", "y8_YALTAi_5epochs_best_+9annotated_fix50.pt")
    TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "annotated_samples", "retrain","images")
    TEST_ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "annotated_samples", "retrain","annotations")

    LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "evaluation")
    RUN_NAME = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Evaluate at common IoU thresholds
    metrics = evaluate_detection_only(
        model_path=MODEL_PATH,
        test_images_dir=TEST_IMAGES_DIR,
        test_annotations_dir=TEST_ANNOTATIONS_DIR,
        iou_thresholds=[0.3, 0.5, 0.7],  # Relaxed to strict
        conf_threshold=0.25,
        logs_dir=LOGS_DIR,
        run_name=RUN_NAME
    )
    
    print(f"mPA@0.5: {metrics.get('mPA@0.5', 0):.4f}")
    
    mpa_05 = metrics.get('mPA@0.5', 0)
    if mpa_05 >= 0.8:
        print("Model is excellent for region detection - ready for production!")
    elif mpa_05 >= 0.6:
        print("Model is good - consider minor fine-tuning for edge cases")
    else:
        print("Model needs more training data or hyperparameter tuning")