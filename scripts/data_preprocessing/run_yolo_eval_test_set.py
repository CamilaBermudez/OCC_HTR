import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.data_preprocessing.yolo_eval_test_set import evaluate_detection_only


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--conf-threshold", type=float, default=0.25)

    args = parser.parse_args()

    project_root = os.environ.get("PROJECT_ROOT") or str(Path(__file__).parent)
    logs_dir = args.logs_dir or str(Path(project_root) / "logs" / "evaluation")

    metrics = evaluate_detection_only(
        model_path=args.model_path,
        test_images_dir=args.images_dir,
        test_annotations_dir=args.annotations_dir,
        logs_dir=logs_dir,
        run_name=args.run_name,
        iou_thresholds=args.iou_thresholds,
        conf_threshold=args.conf_threshold,
    )

    print(f"mPA@0.5: {metrics.get('mPA@0.5', 0):.4f}")


if __name__ == "__main__":
    main()
