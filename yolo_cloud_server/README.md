# YOLOv8 云服务器后端部署包

这个文件夹可以直接发送到你的云服务器，用来运行茶叶病害检测后端服务。

## 目录结构

```text
yolo_cloud_server
├─ app.py
├─ detection_core.py
├─ requirements.txt
├─ Dockerfile
├─ models
│  └─ best.pt
├─ start_linux.sh
└─ start_windows.bat
```

## 方式一：直接 Python 启动

Linux：

```bash
cd yolo_cloud_server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows：

```bat
cd yolo_cloud_server
python -m venv .venv
.venv\Scripts\activate
install_cpu_torch_windows.bat
python app.py
```

如果已经装过 CUDA 版 torch，必须先卸载再安装 CPU 版，否则可能出现 `c10.dll` 初始化失败。

默认监听：

```text
0.0.0.0:5000
```

## 方式二：Docker 启动

```bash
cd yolo_cloud_server
docker build -t tea-yolov8-server .
docker run -d --name tea-yolov8-server -p 5000:5000 tea-yolov8-server
```

## 测试接口

浏览器访问：

```text
http://你的服务器IP:5000/health
```

返回 `ok: true` 说明服务启动成功。

## 小程序连接地址

小程序首页的检测服务地址填写：

```text
http://你的服务器IP:5000
```

如果你绑定了域名和 HTTPS，则填写：

```text
https://你的域名
```

正式版小程序必须使用 HTTPS，并在微信公众平台配置 request/uploadFile/downloadFile 合法域名。开发者工具或真机调试阶段可以先关闭域名校验。

## 服务器安全组

云服务器需要放行 TCP 端口：

```text
5000
```

如果使用 Nginx + HTTPS 反向代理，服务器还需要放行：

```text
80
443
```
