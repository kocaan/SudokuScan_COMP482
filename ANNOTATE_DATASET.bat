@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
"%PYTHON%" annotate_dataset.py %*
if errorlevel 1 (
  echo.
  echo Annotation tool exited with an error.
  pause
)
endlocal
