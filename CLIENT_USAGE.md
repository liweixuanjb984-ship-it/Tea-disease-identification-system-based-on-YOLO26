# YOLOv8 客户端使用说明

## 1. PyQt5 电脑端

安装依赖：

```bash
pip install pyqt5 flask ultralytics opencv-python
```

启动电脑端：

```bash
python pyqt5_detect_client.py
```

默认模型：

```text
runs\detect\runs\train\tea_disease_yolov8\weights\best.pt
```

界面支持选择图片或视频，检测后会显示带框结果。结果保存到：

```text
runs\client_results
```

## 2. 微信小程序端

小程序不能直接运行 YOLOv8 模型，需要先启动本地 Python 检测服务：

```bash
python server_api.py
```

服务地址：

```text
http://127.0.0.1:5000
```

电脑模拟器可以使用 `127.0.0.1`。手机真机调试不能使用 `127.0.0.1`，需要把小程序首页的检测服务地址改成电脑的局域网 IP，例如：

```text
http://10.40.93.44:5000
```

电脑和手机必须连接到同一个网络，并且 Windows 防火墙需要允许 Python/端口 5000 访问。

然后使用微信开发者工具打开目录：

```text
wechat_miniprogram
```

如果开发者工具提示域名校验问题，勾选：

```text
不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书
```

小程序上传图片或视频后，会调用：

```text
http://127.0.0.1:5000/detect
```

检测结果同样保存到：

```text
runs\client_results
```
