# 微信云托管 YOLOv8 服务

部署前需要把项目训练好的模型复制到本目录：

```text
cloudrun_yolo/best.pt
```

本服务提供接口：

```text
GET  /health
POST /detect
GET  /result/<file>
```

`/detect` 使用 multipart/form-data 上传文件，字段名为 `file`。
