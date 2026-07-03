from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from detection_core import DEFAULT_MODEL, DEFAULT_OUTPUT, YoloDetector, is_image, is_video

import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent


class DetectWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, model_path: Path, source_path: Path, output_dir: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self.source_path = source_path
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            detector = YoloDetector(self.model_path)
            if is_image(self.source_path):
                result = detector.predict_image_file(self.source_path, self.output_dir)
            elif is_video(self.source_path):
                result = detector.predict_video_file(self.source_path, self.output_dir)
            else:
                raise RuntimeError("请选择支持的图片或视频文件")
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("茶叶病害 YOLOv8 检测客户端")
        self.resize(1200, 760)

        self.model_path = DEFAULT_MODEL
        self.source_path: Path | None = None
        self.worker: DetectWorker | None = None

        self.model_input = QLineEdit(str(self.model_path))
        self.model_input.setReadOnly(True)

        self.source_input = QLineEdit()
        self.source_input.setReadOnly(True)

        self.input_preview = QLabel("输入预览")
        self.output_preview = QLabel("检测结果")
        for label in (self.input_preview, self.output_preview):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(480, 360)
            label.setStyleSheet("QLabel { border: 1px solid #c8cdd2; background: #f6f7f9; color: #667085; }")

        self.status_label = QLabel("请选择图片或视频后开始检测")

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["类别ID", "类别名称", "置信度", "x1", "y1", "x2/y2"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self._build_ui()

    def _build_ui(self) -> None:
        choose_model_btn = QPushButton("选择模型")
        choose_model_btn.clicked.connect(self.choose_model)

        choose_image_btn = QPushButton("选择图片")
        choose_image_btn.clicked.connect(self.choose_image)

        choose_video_btn = QPushButton("选择视频")
        choose_video_btn.clicked.connect(self.choose_video)

        detect_btn = QPushButton("开始检测")
        detect_btn.clicked.connect(self.start_detection)
        self.detect_btn = detect_btn

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型："))
        model_row.addWidget(self.model_input)
        model_row.addWidget(choose_model_btn)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("文件："))
        source_row.addWidget(self.source_input)
        source_row.addWidget(choose_image_btn)
        source_row.addWidget(choose_video_btn)
        source_row.addWidget(detect_btn)

        preview_row = QHBoxLayout()
        preview_row.addWidget(self.input_preview)
        preview_row.addWidget(self.output_preview)

        layout = QVBoxLayout()
        layout.addLayout(model_row)
        layout.addLayout(source_row)
        layout.addLayout(preview_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", str(ROOT), "PyTorch Model (*.pt)")
        if path:
            self.model_path = Path(path)
            self.model_input.setText(path)

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            str(ROOT),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if path:
            self.set_source(Path(path))

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            str(ROOT),
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.flv)",
        )
        if path:
            self.set_source(Path(path))

    def set_source(self, path: Path) -> None:
        self.source_path = path
        self.source_input.setText(str(path))
        self.status_label.setText("文件已选择，点击开始检测")
        self.clear_table()

        if is_image(path):
            self.show_image(self.input_preview, path)
            self.output_preview.setText("检测结果")
        else:
            self.show_video_first_frame(self.input_preview, path)
            self.output_preview.setText("检测完成后保存视频文件")

    def start_detection(self) -> None:
        if self.source_path is None:
            QMessageBox.warning(self, "提示", "请先选择图片或视频")
            return

        self.detect_btn.setEnabled(False)
        self.status_label.setText("正在加载模型并检测，请稍候...")
        self.clear_table()

        self.worker = DetectWorker(Path(self.model_input.text()), self.source_path, DEFAULT_OUTPUT)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.failed.connect(self.on_detection_failed)
        self.worker.start()

    def on_detection_finished(self, result: dict) -> None:
        self.detect_btn.setEnabled(True)
        output_path = Path(result["output"])

        if result["type"] == "image":
            self.show_image(self.output_preview, output_path)
            self.fill_table(result["detections"])
        else:
            self.show_video_first_frame(self.output_preview, output_path)
            self.fill_table(result.get("first_frame_detections", []))

        self.status_label.setText(
            f"检测完成：共 {result['count']} 个目标，耗时 {result['elapsed']:.2f}s，结果保存到 {output_path}"
        )

    def on_detection_failed(self, message: str) -> None:
        self.detect_btn.setEnabled(True)
        self.status_label.setText("检测失败")
        QMessageBox.critical(self, "错误", message)

    def show_image(self, label: QLabel, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText("图片读取失败")
            return
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_video_first_frame(self, label: QLabel, path: Path) -> None:
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()

        if not ok:
            label.setText("视频读取失败")
            return

        temp_path = DEFAULT_OUTPUT / "preview.jpg"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(temp_path), frame)
        self.show_image(label, temp_path)

    def fill_table(self, detections: list[dict]) -> None:
        self.table.setRowCount(len(detections))
        for row, item in enumerate(detections):
            values = [
                item["class_id"],
                item["class_name"],
                f"{item['confidence']:.3f}",
                item["x1"],
                item["y1"],
                f"{item['x2']}, {item['y2']}",
            ]
            for col, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                table_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, table_item)

    def clear_table(self) -> None:
        self.table.setRowCount(0)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
