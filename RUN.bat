@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title SudokuScan - One Click Launcher

if not exist ".venv\Scripts\python.exe" (
    echo Project environment not found. Running the installer first...
    call "%~dp0INSTALL.bat"
    if errorlevel 1 exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv could not be created. Run INSTALL.bat and review its output.
    pause
    exit /b 1
)

if not "%~1"=="" goto HAS_ARGUMENT
".venv\Scripts\python.exe" launcher.py
goto LAUNCH_FINISHED

:HAS_ARGUMENT
".venv\Scripts\python.exe" launcher.py "%~1"

:LAUNCH_FINISHED
set "LAUNCH_EXIT=%ERRORLEVEL%"
echo.
if not "%LAUNCH_EXIT%"=="0" echo SudokuScan ended with exit code %LAUNCH_EXIT%.
pause
exit /b %LAUNCH_EXIT%
