from __future__ import annotations

import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_file, send_from_directory


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.onnx"
UPLOAD_DIR = ROOT / "runtime" / "uploads"
RESULT_DIR = ROOT / "runtime" / "results"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
CLASS_NAMES = [
    "Black rot of tea",
    "Brown blight of tea",
    "Leaf rust of tea",
    "Red Spider infested tea leaf",
    "Tea Mosquito bug infested leaf",
    "Tea leaf",
    "White spot of tea",
    "disease",
]

app = Flask(__name__)
detector: "OnnxYoloDetector | None" = None


@dataclass
class Detection:
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


class OnnxYoloDetector:
    def __init__(self, model_path: Path, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_image_file(self, image_path: Path) -> dict:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        start = time.time()
        annotated, detections = self.predict_frame(image)
        elapsed = time.time() - start

        save_dir = RESULT_DIR / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{image_path.stem}_{uuid.uuid4().hex[:8]}_det.jpg"
        cv2.imwrite(str(save_path), annotated)

        return {
            "type": "image",
            "source": str(image_path),
            "output": str(save_path),
            "elapsed": elapsed,
            "count": len(detections),
            "detections": [item.to_dict() for item in detections],
        }

    def predict_video_file(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        save_dir = RESULT_DIR / "videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{video_path.stem}_{uuid.uuid4().hex[:8]}_det.mp4"
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        start = time.time()
        frame_index = 0
        total_count = 0
        first_frame_detections: list[dict] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                annotated, detections = self.predict_frame(frame)
                writer.write(annotated)

                if frame_index == 0:
                    first_frame_detections = [item.to_dict() for item in detections]
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
            "first_frame_detections": first_frame_detections,
        }

    def predict_frame(self, image: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        input_tensor, ratio, pad_x, pad_y = self._preprocess(image)
        output = self.session.run(None, {self.input_name: input_tensor})[0]
        detections = self._postprocess(output, image.shape, ratio, pad_x, pad_y)
        return draw_detections(image, detections), detections

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        height, width = image.shape[:2]
        ratio = min(self.imgsz / width, self.imgsz / height)
        new_width = int(round(width * ratio))
        new_height = int(round(height * ratio))
        pad_x = (self.imgsz - new_width) / 2
        pad_y = (self.imgsz - new_height) / 2

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top : top + new_height, left : left + new_width] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, ratio, left, top

    def _postprocess(
        self,
        output: np.ndarray,
        image_shape: tuple[int, int, int],
        ratio: float,
        pad_x: float,
        pad_y: float,
    ) -> list[Detection]:
        predictions = np.squeeze(output).T

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        image_h, image_w = image_shape[:2]

        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < self.conf:
                continue

            x, y, w, h = pred[:4]
            x1 = int((x - w / 2 - pad_x) / ratio)
            y1 = int((y - h / 2 - pad_y) / ratio)
            x2 = int((x + w / 2 - pad_x) / ratio)
            y2 = int((y + h / 2 - pad_y) / ratio)

            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(0, min(x2, image_w - 1))
            y2 = max(0, min(y2, image_h - 1))
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
        if len(keep) == 0:
            return []

        detections: list[Detection] = []
        for index in np.array(keep).flatten():
            x, y, w, h = boxes[index]
            class_id = class_ids[index]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    confidence=scores[index],
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                )
            )
        return detections


def get_detector() -> OnnxYoloDetector:
    global detector
    if detector is None:
        detector = OnnxYoloDetector(
            model_path=MODEL_PATH,
            conf=float(os.environ.get("YOLO_CONF", "0.25")),
            iou=float(os.environ.get("YOLO_IOU", "0.45")),
            imgsz=int(os.environ.get("YOLO_IMGSZ", "640")),
        )
    return detector


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


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


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
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


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response


@app.route("/detect", methods=["OPTIONS"])
def detect_options():
    return ("", 204)


@app.get("/")
def index():
    return send_file(ROOT / "web" / "index.html")


@app.get("/web/<path:filename>")
def web_file(filename: str):
    return send_from_directory(ROOT / "web", filename)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "tea-yolov8-onnx-server",
            "model_exists": MODEL_PATH.exists(),
            "model": str(MODEL_PATH),
        }
    )


@app.post("/detect")
def detect():
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "missing upload file field: file"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"ok": False, "message": "empty filename"}), 400

    suffix = Path(file.filename).suffix.lower()
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(str(upload_path))

    try:
        yolo = get_detector()
        if is_image(upload_path):
            result = yolo.predict_image_file(upload_path)
        elif is_video(upload_path):
            result = yolo.predict_video_file(upload_path)
        else:
            return jsonify({"ok": False, "message": f"unsupported file type: {suffix}"}), 400

        output_path = Path(result["output"]).resolve()
        result["output_url"] = f"/result/{output_path.relative_to(ROOT).as_posix()}"
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.get("/result/<path:filename>")
def result_file(filename: str):
    return send_from_directory(ROOT, filename)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
