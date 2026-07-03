const app = getApp()

Page({
  data: {
    serverAddress: app.globalData.apiBase,
    filePath: "",
    previewPath: "",
    fileType: "",
    detecting: false,
    resultUrl: "",
    resultType: "",
    count: 0,
    elapsed: "0.00",
    detections: []
  },

  onLoad() {
    wx.removeStorageSync("serverAddress")
    this.setData({ serverAddress: app.globalData.apiBase })
  },

  onServerAddressInput(event) {
    const value = event.detail.value.trim()
    this.setData({ serverAddress: value })
    if (value) {
      app.globalData.apiBase = value
    }
  },

  testServer() {
    const base = this.normalizeServerAddress(this.data.serverAddress)
    this.setData({ serverAddress: base })
    console.log("test server:", `${base}/health`)
    wx.request({
      url: `${base}/health`,
      timeout: 5000,
      success: (res) => {
        console.log("health response:", res)
        if (res.statusCode === 200 && res.data && res.data.ok) {
          wx.showToast({ title: "服务连接成功", icon: "success" })
        } else {
          wx.showToast({ title: "服务返回异常", icon: "none" })
        }
      },
      fail: (error) => {
        console.error("health failed:", error)
        wx.showToast({ title: error.errMsg || "无法连接服务", icon: "none" })
      }
    })
  },

  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["album", "camera"],
      success: (res) => {
        const filePath = res.tempFiles[0].tempFilePath
        this.setData({
          filePath,
          previewPath: filePath,
          fileType: "image",
          resultUrl: "",
          detections: []
        })
      }
    })
  },

  chooseVideo() {
    wx.chooseMedia({
      count: 1,
      mediaType: ["video"],
      sourceType: ["album", "camera"],
      success: (res) => {
        const filePath = res.tempFiles[0].tempFilePath
        this.setData({
          filePath,
          previewPath: filePath,
          fileType: "video",
          resultUrl: "",
          detections: []
        })
      }
    })
  },

  startDetect() {
    if (!this.data.filePath) {
      wx.showToast({ title: "请先选择文件", icon: "none" })
      return
    }

    const base = this.normalizeServerAddress(this.data.serverAddress)
    app.globalData.apiBase = base

    this.setData({ detecting: true, resultUrl: "", detections: [] })
    console.log("upload to:", `${base}/detect`)
    wx.uploadFile({
      url: `${base}/detect`,
      filePath: this.data.filePath,
      name: "file",
      timeout: 120000,
      success: (res) => {
        console.log("upload response:", res)
        if (res.statusCode !== 200) {
          wx.showToast({ title: `服务错误：${res.statusCode}`, icon: "none" })
          return
        }

        let payload
        try {
          payload = JSON.parse(res.data)
        } catch (error) {
          wx.showToast({ title: "服务返回格式错误", icon: "none" })
          return
        }

        if (!payload.ok) {
          wx.showToast({ title: payload.message || "检测失败", icon: "none" })
          return
        }

        this.handleDetectResult(base, payload.data)
      },
      fail: (error) => {
        console.error("upload failed:", error)
        wx.showToast({ title: error.errMsg || "无法连接检测服务", icon: "none" })
      },
      complete: () => {
        this.setData({ detecting: false })
      }
    })
  },

  handleDetectResult(base, data) {
    const remoteUrl = `${base}${data.output_url}`
    const detections = this.normalizeDetections(data.detections || data.first_frame_detections || [])
    console.log("download result:", remoteUrl)

    this.setData({
      resultType: data.type,
      count: data.count || 0,
      elapsed: Number(data.elapsed || 0).toFixed(2),
      detections
    })

    wx.downloadFile({
      url: remoteUrl,
      timeout: 120000,
      success: (res) => {
        console.log("download response:", res)
        if (res.statusCode === 200) {
          this.setData({ resultUrl: res.tempFilePath })
        } else {
          wx.showToast({ title: "结果文件下载失败", icon: "none" })
        }
      },
      fail: (error) => {
        console.error("download failed:", error)
        wx.showToast({ title: error.errMsg || "结果文件无法下载", icon: "none" })
      }
    })
  },

  normalizeServerAddress(value) {
    let address = (value || "").trim()
    if (!address) {
      address = app.globalData.apiBase
    }
    if (!/^https?:\/\//i.test(address)) {
      address = `http://${address}`
    }
    return address.replace(/\/+$/, "")
  },

  normalizeDetections(items) {
    return items.map((item) => ({
      ...item,
      confidenceText: Number(item.confidence || 0).toFixed(3),
      boxText: `${item.x1},${item.y1},${item.x2},${item.y2}`
    }))
  }
})
