"""
Train YOLOv8 for document field detection.

This lets you train a real object detection model to locate
field regions (name, DOB, pan_number, etc.) on ID cards.

Usage:
    python train_yolo.py --data_dir ./yolo_dataset --epochs 50

Expected dataset structure (YOLO format):
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   │   ├── pan_001.jpg
    │   │   └── ...
    │   └── val/
    │       ├── pan_101.jpg
    │       └── ...
    ├── labels/
    │   ├── train/
    │   │   ├── pan_001.txt    # YOLO format: class x_center y_center w h
    │   │   └── ...
    │   └── val/
    │       ├── pan_101.txt
    │       └── ...
    └── dataset.yaml

dataset.yaml example:
    train: images/train
    val: images/val
    nc: 7
    names: ['name', 'dob', 'pan_number', 'aadhaar_number', 'father_name', 'gender', 'address']
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 for ID field detection")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to YOLO dataset directory (contains dataset.yaml)")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="Base model: yolov8n.pt, yolov8s.pt, yolov8m.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--output", type=str, default="models/weights/yolo_field_detector.pt")
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(args.data_dir) / "dataset.yaml"

    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found.")
        print("Create a dataset.yaml in your data_dir. See script docstring for format.")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=str(Path(args.output).parent),
        name=Path(args.output).stem,
    )

    # Export best model to expected path
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        import shutil
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best), args.output)
        print(f"\n✅ Best model saved to {args.output}")
    else:
        print(f"\nTraining complete. Check {results.save_dir} for weights.")


if __name__ == "__main__":
    main()