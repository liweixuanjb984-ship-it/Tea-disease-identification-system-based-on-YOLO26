from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

# Default trained model and default detection source.
DEFAULT_MODEL = ROOT / "runs" / "detect" / "runs" / "train" / "tea_disease_yolov8" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "test"
DEFAULT_OUTPUT = ROOT / "runs" / "predict_complete"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


@dataclass
class Detection:
    source: str
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete YOLOv8 inference script.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to YOLOv8 .pt model.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image, folder, video, or camera id.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold.")
    parser.add_argument("--device", default="0", help="CUDA device id, or cpu.")
    parser.add_argument("--show", action="store_true", help="Show realtime window for video/camera.")
    return parser.parse_args()


def is_camera_source(source: str) -> bool:
    return source.isdigit()


def get_image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]

    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)

    return []


def make_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (255, 56, 203),
        (200, 149, 255),
        (199, 55, 255),
    )
    return palette[class_id % len(palette)]


def draw_detections(image: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    output = image.copy()

    for det in detections:
        color = make_color(det.class_id)
        label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        label_y1 = max(det.y1 - text_h - baseline - 6, 0)
        label_y2 = label_y1 + text_h + baseline + 6
        label_x2 = min(det.x1 + text_w + 8, output.shape[1] - 1)

        cv2.rectangle(output, (det.x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            output,
            label,
            (det.x1 + 4, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def collect_detections(result, source_name: str) -> list[Detection]:
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box.astype(int).tolist()
        detections.append(
            Detection(
                source=source_name,
                class_id=class_id,
                class_name=str(names[class_id]),
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return detections


def save_csv(csv_path: Path, rows: list[Detection]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for row in rows:
            writer.writerow(
                [
                    row.source,
                    row.class_id,
                    row.class_name,
                    f"{row.confidence:.6f}",
                    row.x1,
                    row.y1,
                    row.x2,
                    row.y2,
                ]
            )


def save_txt_label(txt_path: Path, image_shape: tuple[int, int, int], detections: list[Detection]) -> None:
    height, width = image_shape[:2]
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with txt_path.open("w", encoding="utf-8") as f:
        for det in detections:
            x_center = ((det.x1 + det.x2) / 2) / width
            y_center = ((det.y1 + det.y2) / 2) / height
            box_width = (det.x2 - det.x1) / width
            box_height = (det.y2 - det.y1) / height
            f.write(
                f"{det.class_id} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f} {det.confidence:.6f} {det.class_name}\n"
            )


def predict_image(model: YOLO, image_path: Path, source_root: Path, output: Path, args: argparse.Namespace) -> list[Detection]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    result = model.predict(
        source=image,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    relative_path = image_path.relative_to(source_root)
    detections = collect_detections(result, str(relative_path))
    annotated = draw_detections(image, detections)

    image_save_path = output / "images" / relative_path.with_name(f"{relative_path.stem}_det{relative_path.suffix}")
    label_save_path = output / "labels" / relative_path.with_suffix(".txt")

    image_save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_save_path), annotated)
    save_txt_label(label_save_path, image.shape, detections)

    print(f"[IMAGE] {image_path} -> {image_save_path} ({len(detections)} detections)")
    return detections


def predict_images(model: YOLO, source: Path, output: Path, args: argparse.Namespace) -> list[Detection]:
    image_paths = get_image_paths(source)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found: {source}")

    source_root = source if source.is_dir() else source.parent
    all_detections: list[Detection] = []

    for image_path in image_paths:
        all_detections.extend(predict_image(model, image_path, source_root, output, args))

    return all_detections


def predict_video(model: YOLO, source: str, output: Path, args: argparse.Namespace) -> list[Detection]:
    camera = is_camera_source(source)
    capture_source = int(source) if camera else source
    cap = cv2.VideoCapture(capture_source)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    video_name = f"camera_{source}" if camera else Path(source).stem
    video_save_path = output / "videos" / f"{video_name}_det.mp4"
    video_save_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_save_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    all_detections: list[Detection] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]

            source_name = f"{video_name}:frame_{frame_index:06d}"
            detections = collect_detections(result, source_name)
            all_detections.extend(detections)

            annotated = draw_detections(frame, detections)
            writer.write(annotated)

            if args.show:
                cv2.imshow("YOLOv8 Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"[VIDEO] {source} -> {video_save_path} ({len(all_detections)} detections)")
    return all_detections


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    output = args.output.resolve()
    source_text = str(args.source)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    start_time = time.time()

    if is_camera_source(source_text):
        detections = predict_video(model, source_text, output, args)
    else:
        source_path = Path(source_text).resolve()
        if source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
            detections = predict_video(model, str(source_path), output, args)
        else:
            detections = predict_images(model, source_path, output, args)

    save_csv(output / "detections.csv", detections)

    elapsed = time.time() - start_time
    print(f"Done. Total detections: {len(detections)}")
    print(f"Output folder: {output}")
    print(f"CSV file: {output / 'detections.csv'}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
