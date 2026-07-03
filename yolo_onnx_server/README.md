# YOLOv8 ONNX 云服务器后端

这个版本不依赖 PyTorch、Ultralytics，也不会导入 `torch`，适合云服务器上 `torch c10.dll` 加载失败的情况。

## 启动

Windows Server：

```bat
cd yolo_onnx_server
python -m pip install -r requirements.txt
python app.py
```

如果 `import onnxruntime` 报 `DLL load failed` 或 `找不到指定的模块`，先安装 Windows C++ 运行库：

```bat
install_windows_runtime.bat
```

如果脚本执行后仍失败，重启 Windows Server，再测试：

```bat
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Linux：

```bash
cd yolo_onnx_server
python3 -m pip install -r requirements.txt
python3 app.py
```

## 测试

```text
http://你的服务器IP:5000/health
```

## 网页前端

启动 Flask 和 Caddy 后，浏览器访问：

```text
https://ottoordersystem.site
```

即可打开网页检测界面。网页和微信小程序共用同一个后端接口：

```text
POST /detect
GET  /result/<file>
```

## 小程序地址

小程序首页填写：

```text
https://ottoordersystem.site
```

## HTTPS 反向代理

小程序预览和正式环境应使用 HTTPS。Flask 本身是 HTTP 服务，不要直接让 `https://域名` 指向 Flask 端口。

Windows Server 推荐使用 Caddy 自动申请 HTTPS 证书：

```powershell
cd C:\Users\yolo_onnx_server
.\start_caddy_windows.ps1
```

需要确保服务器安全组放行：

```text
80
443
5000
```

Flask 继续运行：

```powershell
python .\app.py
```

Flask 默认只监听本机：

```text
127.0.0.1:5000
```

Caddy 负责把：

```text
https://ottoordersystem.site
```

转发到：

```text
http://127.0.0.1:5000
```

配置好 Caddy 后，云服务器安全组可以关闭公网 `5000` 端口，只保留 `80` 和 `443` 对外开放。
