from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


DEFAULT_DATASET_DIR = Path(r"O:\yolo_tea\datasets\datasets")
DEFAULT_MODEL = "yolov8n.pt"


def build_data_yaml(dataset_dir: Path) -> Path:
    """Create an absolute-path YOLO data yaml from the dataset folder."""
    dataset_dir = dataset_dir.resolve()
    source_yaml = dataset_dir / "data.yaml"

    if not source_yaml.exists():
        raise FileNotFoundError(f"Cannot find data.yaml: {source_yaml}")

    with source_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "valid" / "images"
    test_images = dataset_dir / "test" / "images"

    for path in (train_images, val_images, test_images):
        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

    fixed_data = {
        "train": str(train_images),
        "val": str(val_images),
        "test": str(test_images),
        "nc": int(data["nc"]),
        "names": data["names"],
    }

    output_yaml = dataset_dir / "data_abs.yaml"
    with output_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(fixed_data, f, allow_unicode=True, sort_keys=False)

    return output_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on tea disease dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Dataset root folder.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLOv8 model, e.g. yolov8n.pt/yolov8s.pt.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device id, or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--project", default="runs/train", help="Output project directory.")
    parser.add_argument("--name", default="tea_disease_yolov8", help="Experiment name.")
    parser.add_argument("--resume", action="store_true", help="Resume last interrupted training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = build_data_yaml(args.dataset)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="auto",
        patience=50,
        save=True,
        save_period=-1,
        cache=False,
        seed=42,
        deterministic=True,
        resume=args.resume,
    )

    model.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
