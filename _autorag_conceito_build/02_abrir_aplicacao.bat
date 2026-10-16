@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Execute 01_configurar_ambiente.bat primeiro.
  pause
  exit /b 1
)
if exist "%~dp0config\ollama-local.json" powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Iniciar-Ollama.ps1"
"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
if errorlevel 1 pause
