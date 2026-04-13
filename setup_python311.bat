@echo off
setlocal

set "PY311_CMD="

where py >nul 2>nul
if not errorlevel 1 (
  py -3.11 --version >nul 2>nul
  if not errorlevel 1 (
    set "PY311_CMD=py -3.11"
  )
)

if not defined PY311_CMD (
  if exist "C:\Users\arvbo\AppData\Local\Programs\Python\Python311\python.exe" (
    set "PY311_CMD=C:\Users\arvbo\AppData\Local\Programs\Python\Python311\python.exe"
  )
)

if not defined PY311_CMD (
  echo Python 3.11 is not installed.
  echo Install Python 3.11 from python.org, make sure the launcher is enabled, then rerun this script.
  exit /b 1
)

if not exist .venv (
  echo Creating .venv with Python 3.11...
  %PY311_CMD% -m venv .venv
)

echo Activating .venv...
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo Failed to activate .venv.
  exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing project requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Requirement install failed. If PyAudio is the blocker, install PortAudio or a compatible PyAudio wheel for Python 3.11 and rerun this script.
  exit /b 1
)

echo.
echo Python 3.11 environment is ready.
echo Activate it later with: .venv\Scripts\activate
endlocal
