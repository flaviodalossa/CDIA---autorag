@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Execute 01_configurar_ambiente.bat primeiro.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0diagnostico_modelos.py"
if errorlevel 1 (pause & exit /b 1)
"%~dp0.venv\Scripts\python.exe" "%~dp0tests\integracao_ollama.py"
if errorlevel 1 (pause & exit /b 1)
pause
