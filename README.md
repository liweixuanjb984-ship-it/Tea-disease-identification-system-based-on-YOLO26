# Tea Disease Identification System Based on YOLO26

基于 YOLO26 的茶叶病害识别系统，包含训练脚本、检测推理代码、Flask 后端、Web 前端、PyQt5 客户端、微信小程序示例、模型权重、训练结果和项目数据文件。

## 项目简介

本项目面向茶叶病害目标检测场景，使用自建数据集进行标注和训练，并对 YOLOv8 与 YOLO26 模型效果进行对比。早期 YOLOv8 模型在无关图片上存在误检问题，后续使用 YOLO26 训练后，模型抗干扰能力明显提升。

## 项目内容

- `train_yolov8.py`：训练入口脚本。
- `detection_core.py`：本地推理核心逻辑。
- `server_api.py`：本地 Flask API。
- `pyqt5_detect_client.py`：PyQt5 桌面检测客户端。
- `infer_two_models.py`：双模型对比推理脚本。
- `main/`：训练相关代码、数据集与模型工程文件。
- `runs/`：训练与检测输出结果。
- `yolo_cloud_server/`：云端 Flask 检测服务。
- `yolo_onnx_server/`：ONNX 推理服务示例。
- `yolo_web_frontend/`：Web 前端。
- `wechat_miniprogram/`：微信小程序端。
- `cloudrun_yolo/`：Cloud Run 部署示例。

## 快速运行

安装依赖：

```bash
pip install -r yolo_cloud_server/requirements.txt
```

启动 Flask 后端：

```bash
cd yolo_cloud_server
python app.py
```

访问健康检查：

```text
http://127.0.0.1:5000/health
```

## 说明

本仓库按完整项目上传，包含训练数据、训练输出和模型权重等文件。由于这类文件体积较大，首次克隆仓库可能需要较长时间。
