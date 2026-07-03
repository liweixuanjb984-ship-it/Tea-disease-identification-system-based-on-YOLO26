const API_BASE = (window.YOLO_API_BASE || "").replace(/\/+$/, "")

const state = {
  file: null,
  inputUrl: "",
  outputUrl: "",
}

const els = {
  healthPulse: document.getElementById("healthPulse"),
  healthText: document.getElementById("healthText"),
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  fileName: document.getElementById("fileName"),
  detectBtn: document.getElementById("detectBtn"),
  clearBtn: document.getElementById("clearBtn"),
  progress: document.getElementById("progress"),
  inputPreview: document.getElementById("inputPreview"),
  outputPreview: document.getElementById("outputPreview"),
  downloadLink: document.getElementById("downloadLink"),
  countMetric: document.getElementById("countMetric"),
  timeMetric: document.getElementById("timeMetric"),
  typeMetric: document.getElementById("typeMetric"),
  resultRows: document.getElementById("resultRows"),
  toast: document.getElementById("toast"),
}

function apiUrl(path) {
  return `${API_BASE}${path}`
}

async function checkHealth() {
  try {
    const response = await fetch(apiUrl("/health"))
    const data = await response.json()
    if (response.ok && data.ok) {
      els.healthPulse.className = "pulse ok"
      els.healthText.textContent = "运行中"
      return
    }
    throw new Error("服务异常")
  } catch (error) {
    els.healthPulse.className = "pulse fail"
    els.healthText.textContent = "连接失败"
  }
}

function setFile(file) {
  clearOutput()
  state.file = file
  state.inputUrl = URL.createObjectURL(file)
  els.fileName.textContent = `${file.name} · ${formatBytes(file.size)}`
  els.detectBtn.disabled = false
  els.clearBtn.disabled = false
  els.typeMetric.textContent = file.type.startsWith("video") ? "视频" : "图片"
  renderPreview(els.inputPreview, state.inputUrl, file.type)
}

function renderPreview(container, url, mimeType) {
  container.innerHTML = ""
  if (mimeType.startsWith("video")) {
    const video = document.createElement("video")
    video.src = url
    video.controls = true
    container.appendChild(video)
    return
  }

  const image = document.createElement("img")
  image.src = url
  image.alt = "preview"
  container.appendChild(image)
}

async function detect() {
  if (!state.file) {
    showToast("请先选择文件")
    return
  }

  els.detectBtn.disabled = true
  els.progress.classList.add("active")
  clearOutput()

  const form = new FormData()
  form.append("file", state.file)

  try {
    const response = await fetch(apiUrl("/detect"), {
      method: "POST",
      body: form,
    })
    const payload = await response.json()

    if (!response.ok || !payload.ok) {
      throw new Error(payload.message || `检测失败：${response.status}`)
    }

    renderResult(payload.data)
    showToast("检测完成")
  } catch (error) {
    showToast(error.message || "检测失败")
  } finally {
    els.detectBtn.disabled = false
    els.progress.classList.remove("active")
  }
}

function renderResult(data) {
  const outputUrl = apiUrl(data.output_url)
  const detections = data.detections || data.first_frame_detections || []

  state.outputUrl = outputUrl
  els.countMetric.textContent = data.count || 0
  els.timeMetric.textContent = `${Number(data.elapsed || 0).toFixed(2)}s`
  els.typeMetric.textContent = data.type === "video" ? "视频" : "图片"
  els.downloadLink.href = outputUrl
  els.downloadLink.classList.remove("hidden")

  renderPreview(els.outputPreview, outputUrl, data.type === "video" ? "video/mp4" : "image/jpeg")
  renderRows(detections)
}

function renderRows(items) {
  els.resultRows.innerHTML = ""
  if (!items.length) {
    els.resultRows.innerHTML = '<tr><td colspan="3" class="empty">未检测到目标</td></tr>'
    return
  }

  const fragment = document.createDocumentFragment()
  items.forEach((item) => {
    const row = document.createElement("tr")
    row.innerHTML = `
      <td>${escapeHtml(item.class_name)}</td>
      <td>${Number(item.confidence || 0).toFixed(3)}</td>
      <td>${item.x1}, ${item.y1}, ${item.x2}, ${item.y2}</td>
    `
    fragment.appendChild(row)
  })
  els.resultRows.appendChild(fragment)
}

function clearAll() {
  if (state.inputUrl) {
    URL.revokeObjectURL(state.inputUrl)
  }
  state.file = null
  state.inputUrl = ""
  els.fileInput.value = ""
  els.fileName.textContent = "尚未选择文件"
  els.detectBtn.disabled = true
  els.clearBtn.disabled = true
  els.inputPreview.innerHTML = "<span>暂无输入</span>"
  els.typeMetric.textContent = "-"
  clearOutput()
}

function clearOutput() {
  els.outputPreview.innerHTML = "<span>暂无结果</span>"
  els.downloadLink.classList.add("hidden")
  els.countMetric.textContent = "0"
  els.timeMetric.textContent = "0.00s"
  els.resultRows.innerHTML = '<tr><td colspan="3" class="empty">暂无检测数据</td></tr>'
}

function showToast(message) {
  els.toast.textContent = message
  els.toast.classList.add("show")
  window.clearTimeout(showToast.timer)
  showToast.timer = window.setTimeout(() => {
    els.toast.classList.remove("show")
  }, 2600)
}

function formatBytes(bytes) {
  if (!bytes) return "0 B"
  const units = ["B", "KB", "MB", "GB"]
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;")
}

els.fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0]
  if (file) {
    setFile(file)
  }
})

els.detectBtn.addEventListener("click", detect)
els.clearBtn.addEventListener("click", clearAll)

;["dragenter", "dragover"].forEach((name) => {
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault()
    els.dropZone.classList.add("dragging")
  })
})

;["dragleave", "drop"].forEach((name) => {
  els.dropZone.addEventListener(name, (event) => {
    event.preventDefault()
    els.dropZone.classList.remove("dragging")
  })
})

els.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0]
  if (file) {
    setFile(file)
  }
})

checkHealth()
