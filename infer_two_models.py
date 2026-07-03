from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    ROOT / "yolo26n.pt",
    ROOT / "yolov8n.pt",
]
DEFAULT_SOURCE = Path(r"O:\yolo_tea\基于YOLOv8的茶叶病害识别项目(1)\基于YOLOv8的茶叶病害识别项目\test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection with two YOLOv8 models.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Image, video, or folder to detect.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more model weight files. Defaults to yolo26n.pt and yolov8n.pt.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "infer_two_models")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0", help="CUDA device id, or cpu.")
    return parser.parse_args()


def iter_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return []


def draw_image(model: YOLO, image_path: Path, save_path: Path, imgsz: int, conf: float, iou: float, device: str) -> None:
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]

    annotated = result.plot()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotated)


def draw_video(model: YOLO, video_path: Path, save_path: Path, imgsz: int, conf: float, iou: float, device: str) -> None:
    results = model.predict(
        source=str(video_path),
        stream=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for result in results:
            writer.write(result.plot())
    finally:
        writer.release()


def detect_with_model(model_path: Path, source: Path, output: Path, imgsz: int, conf: float, iou: float, device: str) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = YOLO(str(model_path))
    model_output = output / model_path.stem

    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        draw_video(
            model=model,
            video_path=source,
            save_path=model_output / f"{source.stem}_det.mp4",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
        )
        return

    image_paths = iter_images(source)
    if not image_paths:
        raise FileNotFoundError(f"No supported images or video found in: {source}")

    source_root = source if source.is_dir() else source.parent
    for image_path in image_paths:
        relative_path = image_path.relative_to(source_root)
        save_path = model_output / relative_path.with_name(f"{relative_path.stem}_det{relative_path.suffix}")
        draw_image(
            model=model,
            image_path=image_path,
            save_path=save_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    output.mkdir(parents=True, exist_ok=True)

    for model_path in args.model:
        detect_with_model(
            model_path=model_path.resolve(),
            source=source,
            output=output,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

    print(f"Detection results saved to: {output}")


if __name__ == "__main__":
    main()
