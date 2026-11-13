@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title SudokuScan - Personal Dataset Diagnosis

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Running INSTALL.bat first...
    call "%~dp0INSTALL.bat"
    if errorlevel 1 exit /b 1
)

set "DIAG_FLAGS="
echo.
echo Optional extended studies:
set /p "RUN_PSM=Run the five-mode Tesseract PSM comparison? [y/N]: "
if /i "%RUN_PSM%"=="Y" set "DIAG_FLAGS=%DIAG_FLAGS% --psm-sweep"
set /p "RUN_HARRIS=Run the 36-setting Harris study (requires at least 6 annotated photos)? [y/N]: "
if /i "%RUN_HARRIS%"=="Y" set "DIAG_FLAGS=%DIAG_FLAGS% --parameter-study --parameter-limit 6"

echo.
echo Select your original photos, matching JSON annotations, and a separate output folder.
".venv\Scripts\python.exe" diagnose_dataset.py --select %DIAG_FLAGS%
set "DIAG_EXIT=%ERRORLEVEL%"
echo.
if "%DIAG_EXIT%"=="0" (
    echo Diagnosis finished. Open REPORT_SUMMARY.md in the selected output folder.
) else (
    echo Diagnosis ended with exit code %DIAG_EXIT%. Check the command output and log files.
)
echo.
pause
exit /b %DIAG_EXIT%
