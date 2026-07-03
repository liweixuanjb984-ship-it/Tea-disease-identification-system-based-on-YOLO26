# 茶叶病害检测纯前端

这个目录只包含静态前端文件，不包含模型和后端代码。

## 文件

```text
index.html
styles.css
app.js
config.js
```

## 后端地址

默认调用：

```text
https://ottoordersystem.site
```

如需修改，编辑：

```text
config.js
```

```js
window.YOLO_API_BASE = "https://你的后端域名"
```

## 部署

把整个 `yolo_web_frontend` 文件夹上传到任意静态网站服务器即可。

如果直接本地预览，可以双击打开 `index.html`，但部分浏览器会限制本地文件跨域请求。推荐放到服务器或用本地静态服务打开。
