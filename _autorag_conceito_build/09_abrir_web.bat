@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  set "AUTORAG_WEB_PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist ..\.venv\Scripts\python.exe (
  set "AUTORAG_WEB_PYTHON=%~dp0..\.venv\Scripts\python.exe"
) else (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
  set "AUTORAG_WEB_PYTHON=%~dp0.venv\Scripts\python.exe"
)
"%AUTORAG_WEB_PYTHON%" -m pip install -r requirements-web.txt
if errorlevel 1 exit /b 1
if not defined AUTORAG_PORT set "AUTORAG_PORT=8766"
echo Aplicacao disponivel em http://127.0.0.1:%AUTORAG_PORT%
"%AUTORAG_WEB_PYTHON%" -m uvicorn api:app --host 127.0.0.1 --port %AUTORAG_PORT% --workers 1
