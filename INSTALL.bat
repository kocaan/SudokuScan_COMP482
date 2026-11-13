@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title SudokuScan - One Click Installer

echo ============================================================
echo  SudokuScan - COMP 482 Project 1
echo One-click environment installer
echo ============================================================
echo.

set "PY_RUN=py -3"
%PY_RUN% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if errorlevel 1 (
    set "PY_RUN=python"
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if errorlevel 1 goto NO_PYTHON
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating the project virtual environment...
    %PY_RUN% -m venv .venv
    if errorlevel 1 goto INSTALL_FAILED
) else (
    echo [1/5] Existing .venv found.
)

echo.
echo [2/5] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto INSTALL_FAILED

echo.
echo [3/5] Installing Python dependencies and test tools...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto INSTALL_FAILED

echo.
echo [4/5] Checking Tesseract OCR...
where tesseract >nul 2>&1
if not errorlevel 1 goto TESSERACT_READY
if exist "%ProgramFiles%\Tesseract-OCR\tesseract.exe" goto TESSERACT_READY
if exist "%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe" goto TESSERACT_READY
if defined LOCALAPPDATA if exist "%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe" goto TESSERACT_READY

where winget >nul 2>&1
if errorlevel 1 goto TESSERACT_MANUAL

echo Tesseract was not found. Attempting installation with Windows Package Manager...
winget install --id UB-Mannheim.TesseractOCR -e --accept-source-agreements --accept-package-agreements
if errorlevel 1 goto TESSERACT_MANUAL

echo.
:TESSERACT_READY
".venv\Scripts\python.exe" -c "from sudokuscan.recognition import _tesseract_executable; import sys; p=_tesseract_executable(); print('Tesseract:', p or 'NOT FOUND'); sys.exit(0 if p else 1)"
if errorlevel 1 goto TESSERACT_MANUAL

echo.
echo [5/5] Running the automated test suite...
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto TESTS_FAILED

echo.
echo ============================================================
echo  Installation completed successfully.
echo  Double-click RUN.bat to scan a Sudoku photo or folder.
echo ============================================================
echo.
pause
exit /b 0

:TESSERACT_MANUAL
echo.
echo WARNING: The Python environment is installed, but Tesseract OCR is still missing.
echo Install the Windows Tesseract OCR application, then run INSTALL.bat again.
echo Common install location: C:\Program Files\Tesseract-OCR\tesseract.exe
echo The scanner can detect/rectify puzzles without Tesseract, but digit OCR will be blank.
echo.
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto TESTS_FAILED
pause
exit /b 0

:NO_PYTHON
echo ERROR: Python 3.10 or newer was not found.
echo Install 64-bit Python 3.10+ and enable the Python Launcher or add python.exe to PATH.
echo Then run INSTALL.bat again.
pause
exit /b 1

:INSTALL_FAILED
echo.
echo ERROR: Python environment setup failed. Check your internet connection and pip output.
pause
exit /b 1

:TESTS_FAILED
echo.
echo ERROR: One or more tests failed. Review the output above before using the scanner.
pause
exit /b 1
