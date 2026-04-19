$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$ollamaExe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
$serveHudScript = Join-Path $PSScriptRoot "serve-hud.ps1"

if (-not (Test-Path $venvPython)) {
  throw "Project virtual environment not found at $venvPython"
}

if (-not (Test-Path $ollamaExe)) {
  throw "Ollama executable not found at $ollamaExe"
}

if (-not (Test-Path $serveHudScript)) {
  throw "HUD server script not found at $serveHudScript"
}

Write-Host "Starting Ollama, Corebrum backend, and HUD server..."

Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
  "-NoExit",
  "-Command",
  "`$env:OLLAMA_ORIGINS='*'; & '$ollamaExe' serve"
)

Start-Sleep -Seconds 2

Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
  "-NoExit",
  "-Command",
  "& '$venvPython' 'main.py'"
)

Start-Sleep -Seconds 2

Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy",
  "Bypass",
  "-File",
  $serveHudScript,
  "8080"
)

Write-Host ""
Write-Host "Launched services in separate PowerShell windows."
Write-Host "HUD URL: http://127.0.0.1:8080/interview-copilot-v3.html"
Write-Host "Backend health: http://127.0.0.1:8001/health"
Write-Host "Ollama tags: http://127.0.0.1:11434/api/tags"
