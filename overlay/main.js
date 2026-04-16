const { app, BrowserWindow, globalShortcut, screen } = require("electron");
const path = require("path");
const { execSync } = require("child_process");

try {
  console.log("Cleaning up orphaned processes...");
  const currentPid = process.pid;
  execSync(`powershell -Command "Get-Process electron -ErrorAction SilentlyContinue | Where-Object { $_.Id -ne ${currentPid} } | Stop-Process -Force"`, { stdio: 'ignore' });
} catch (e) {
  // Safe to ignore
}

let mainWindow = null;

const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

function createWindow() {
  if (mainWindow) {
    mainWindow.destroy();
  }
  
  console.log("Creating window...");
  
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.workAreaSize;
  
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 600,
    x: Math.floor((width - 1000) / 2),
    y: Math.floor((height - 600) / 2),
    title: "Corebrum HUD",
    show: true,
    frame: false,
    transparent: true,
    resizable: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, "preload.js")
    }
  });

  mainWindow.loadFile(path.join(__dirname, "index.html"));
  
  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    console.log(`[RENDERER] ${message}`);
  });

  mainWindow.show();
  console.log("Window created and shown");

  globalShortcut.register("CommandOrControl+Shift+X", () => {
    if (mainWindow) {
      if (mainWindow.isVisible()) {
        mainWindow.hide();
      } else {
        mainWindow.show();
        mainWindow.focus();
      }
    }
  });
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => { 
  globalShortcut.unregisterAll(); 
  app.quit(); 
});
