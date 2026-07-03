from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
import os
from pathlib import Path
from typing import Iterable

# Avoid CUDA DLL initialization failures on Windows when the installed
# GPU PyTorch build does not match the local driver/runtime environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "detect" / "runs" / "train" / "tea_disease_yolov8" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "client_results"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


@dataclass
class DetectionItem:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["confidence"] = round(self.confidence, 6)
        return data


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(str(self.model_path))

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[DetectionItem], float]:
        start = time.time()
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        elapsed = time.time() - start
        detections = self._collect_detections(result)
        annotated = draw_detections(frame, detections)
        return annotated, detections, elapsed

    def predict_image_file(self, image_path: str | Path, output_dir: str | Path = DEFAULT_OUTPUT) -> dict:
        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        annotated, detections, elapsed = self.predict_frame(image)

        save_dir = output_dir / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{image_path.stem}_{uuid.uuid4().hex[:8]}_det{image_path.suffix}"
        cv2.imwrite(str(save_path), annotated)

        return {
            "type": "image",
            "source": str(image_path),
            "output": str(save_path),
            "elapsed": elapsed,
            "count": len(detections),
            "detections": [item.to_dict() for item in detections],
        }

    def predict_video_file(self, video_path: str | Path, output_dir: str | Path = DEFAULT_OUTPUT) -> dict:
        video_path = Path(video_path).resolve()
        output_dir = Path(output_dir).resolve()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_dir = output_dir / "videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_path.stem}_{uuid.uuid4().hex[:8]}_det.mp4"

        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_index = 0
        total_count = 0
        first_detections: list[dict] = []
        start = time.time()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                annotated, detections, _ = self.predict_frame(frame)
                writer.write(annotated)

                if frame_index == 0:
                    first_detections = [item.to_dict() for item in detections]
                total_count += len(detections)
                frame_index += 1
        finally:
            cap.release()
            writer.release()

        return {
            "type": "video",
            "source": str(video_path),
            "output": str(save_path),
            "elapsed": time.time() - start,
            "frames": frame_index,
            "count": total_count,
            "first_frame_detections": first_detections,
        }

    def _collect_detections(self, result) -> list[DetectionItem]:
        detections: list[DetectionItem] = []
        if result.boxes is None:
            return detections

        names = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int).tolist()
            detections.append(
                DetectionItem(
                    class_id=int(class_id),
                    class_name=str(names[int(class_id)]),
                    confidence=float(confidence),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def draw_detections(image: np.ndarray, detections: Iterable[DetectionItem]) -> np.ndarray:
    output = image.copy()

    for item in detections:
        color = make_color(item.class_id)
        label = f"{item.class_name} {item.confidence:.2f}"

        cv2.rectangle(output, (item.x1, item.y1), (item.x2, item.y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        label_y1 = max(item.y1 - text_h - baseline - 6, 0)
        label_y2 = label_y1 + text_h + baseline + 6
        label_x2 = min(item.x1 + text_w + 8, output.shape[1] - 1)

        cv2.rectangle(output, (item.x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            output,
            label,
            (item.x1 + 4, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def make_color(class_id: int) -> tuple[int, int, int]:
    colors = (
        (56, 56, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (134, 219, 61),
        (255, 194, 0),
        (255, 115, 100),
        (255, 56, 132),
        (199, 55, 255),
    )
    return colors[class_id % len(colors)]
