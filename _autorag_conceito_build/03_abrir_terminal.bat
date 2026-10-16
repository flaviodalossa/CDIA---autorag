@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "%~dp0.venv\Scripts\activate.bat" (
  echo Execute 01_configurar_ambiente.bat primeiro.
  pause
  exit /b 1
)
call "%~dp0.venv\Scripts\activate.bat"
python autorag_cli.py --help
cmd /k
