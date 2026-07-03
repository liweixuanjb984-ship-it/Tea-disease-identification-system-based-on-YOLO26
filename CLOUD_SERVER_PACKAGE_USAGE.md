# 自有云服务器部署说明

后端部署包目录：

```text
yolo_cloud_server
```

把这个整个文件夹上传到你的云服务器即可。

## 1. 云服务器要求

建议环境：

```text
Python 3.10 - 3.12
内存 2GB 以上
磁盘 5GB 以上
```

CPU 可以运行，但推理速度较慢。视频检测会更慢。

## 2. 放行端口

云服务器安全组、防火墙需要放行：

```text
TCP 5000
```

如果后续配置 HTTPS/Nginx，再放行：

```text
TCP 80
TCP 443
```

## 3. 启动服务

Linux：

```bash
cd yolo_cloud_server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows Server：

```bat
cd yolo_cloud_server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

服务默认监听：

```text
0.0.0.0:5000
```

## 4. 测试服务

在浏览器打开：

```text
http://你的服务器IP:5000/health
```

看到 `ok: true` 表示服务正常。

## 5. 小程序连接服务器

打开：

```text
wechat_miniprogram/app.js
```

把：

```js
apiBase: "http://你的服务器IP:5000"
```

替换成你的服务器地址，例如：

```js
apiBase: "http://1.2.3.4:5000"
```

开发者工具里也可以直接在首页输入框填写服务器地址，然后点击「测试」。

## 6. 正式发布注意

开发调试阶段可以用 HTTP + IP。

正式上线小程序时，微信要求合法域名通常必须是 HTTPS 域名，不能直接用裸 IP。你需要：

1. 给服务器绑定域名。
2. 配置 HTTPS 证书。
3. 使用 Nginx 反向代理到 `127.0.0.1:5000`。
4. 在微信公众平台配置 request、uploadFile、downloadFile 合法域名。

## 7. Docker 方式

服务器如果有 Docker，可以运行：

```bash
cd yolo_cloud_server
docker build -t tea-yolov8-server .
docker run -d --name tea-yolov8-server -p 5000:5000 tea-yolov8-server
```
