const { contextBridge, ipcRenderer } = require("electron");

window.electronAPI = {
  setAlwaysOnTop: (val) => ipcRenderer.send("set-always-on-top", val),
  show: () => ipcRenderer.send("show-window"),
  hide: () => ipcRenderer.send("hide-window")
};

console.log("[PRELOAD] electronAPI exposed");
