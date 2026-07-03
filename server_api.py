from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from detection_core import DEFAULT_MODEL, DEFAULT_OUTPUT, YoloDetector, is_image, is_video


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "runs" / "wechat_uploads"
RESULT_DIR = DEFAULT_OUTPUT

app = Flask(__name__)
detector: YoloDetector | None = None


def get_detector() -> YoloDetector:
    global detector
    if detector is None:
        detector = YoloDetector(DEFAULT_MODEL)
    return detector


@app.get("/health")
def health():
    return jsonify({"ok": True, "model": str(DEFAULT_MODEL)})


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
    app.run(host="0.0.0.0", port=5000, debug=False)
