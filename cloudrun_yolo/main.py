from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from detection_core import YoloDetector, is_image, is_video


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"
UPLOAD_DIR = ROOT / "runtime" / "uploads"
RESULT_DIR = ROOT / "runtime" / "results"

app = Flask(__name__)
detector: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global detector
    if detector is None:
        detector = YoloDetector(
            model_path=MODEL_PATH,
            conf=float(os.environ.get("YOLO_CONF", "0.25")),
            iou=float(os.environ.get("YOLO_IOU", "0.45")),
            imgsz=int(os.environ.get("YOLO_IMGSZ", "640")),
            device=os.environ.get("YOLO_DEVICE", "cpu"),
        )
    return detector


@app.get("/")
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "tea-yolov8", "model": MODEL_PATH.name})


@app.post("/detect")
def detect():
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "missing upload file"}), 400

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
            result = yolo.predict_image_file(upload_path, RESULT_DIR)
        elif is_video(upload_path):
            result = yolo.predict_video_file(upload_path, RESULT_DIR)
        else:
            return jsonify({"ok": False, "message": "unsupported file type"}), 400

        output_path = Path(result["output"]).resolve()
        result["output_url"] = f"/result/{output_path.relative_to(ROOT).as_posix()}"
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.get("/result/<path:filename>")
def result_file(filename: str):
    return send_from_directory(ROOT, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "80")), debug=False)
