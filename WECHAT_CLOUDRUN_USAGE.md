# 微信云托管部署说明

YOLOv8/PyTorch 不能直接放在普通小程序前端执行，也不适合普通云函数。这里使用微信开发者工具里的「云开发」付费云端能力：云托管。

## 1. 开通云开发

1. 打开微信开发者工具。
2. 导入本项目的小程序目录：

```text
wechat_miniprogram
```

3. 点击顶部「云开发」。
4. 如果没有环境，按提示开通一个云开发环境。
5. 进入「云托管」。
6. 选择付费套餐或按量计费。

## 2. 部署后端服务

本项目已经准备好云托管服务目录：

```text
cloudrun_yolo
```

目录里包含：

```text
Dockerfile
main.py
detection_core.py
best.pt
requirements.txt
```

在云托管里创建服务，例如：

```text
tea-yolov8
```

部署方式选择「代码上传」或「本地目录部署」，目录选择：

```text
cloudrun_yolo
```

端口使用：

```text
80
```

启动命令使用 Dockerfile 默认命令即可。

## 3. 获取服务 HTTPS 地址

部署成功后，在云托管服务详情里找到公网访问地址或默认访问地址，格式类似：

```text
https://xxx.service.tcloudbase.com
```

打开浏览器访问：

```text
https://xxx.service.tcloudbase.com/health
```

看到下面结果说明服务可用：

```json
{"ok": true}
```

## 4. 修改小程序服务地址

打开：

```text
wechat_miniprogram/app.js
```

把：

```js
apiBase: "https://你的云托管服务域名"
```

改成你的云托管 HTTPS 地址，例如：

```js
apiBase: "https://xxx.service.tcloudbase.com"
```

也可以在小程序首页输入框里临时填写这个地址，然后点击「测试」。

## 5. 真机调试

1. 重新编译小程序。
2. 点击「预览」或「真机调试」。
3. 首页点击「测试」。
4. 选择图片或视频。
5. 点击「开始检测」。

## 6. 注意事项

- 云托管首次部署会比较慢，因为需要安装 PyTorch、Ultralytics、OpenCV。
- 如果内存太小，PyTorch 可能启动失败，建议云托管实例内存至少 2GB，视频检测建议更高。
- CPU 推理速度较慢，图片可用；视频检测会比较慢。
- 如果要使用 GPU，需要云端实例支持 GPU，并修改环境变量 `YOLO_DEVICE`。
- 小程序真机不能访问电脑的 `127.0.0.1`，使用云托管后应该填写 HTTPS 云服务地址。
