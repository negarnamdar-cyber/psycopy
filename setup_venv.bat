@echo off
REM =============================================================================
REM Setup script (venv + pip only)
REM =============================================================================
setlocal enabledelayedexpansion

echo ==========================================
echo Speech Gating Experiment - Setup (Windows)
echo ==========================================
echo.

set "PYTHON_CMD="

where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3.10 --version >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON_CMD=py -3.10"
    if not defined PYTHON_CMD (
        py -3.11 --version >nul 2>&1
        if !errorlevel! equ 0 set "PYTHON_CMD=py -3.11"
    )
)

if not defined PYTHON_CMD (
    where python >nul 2>&1
    if !errorlevel! equ 0 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    echo Error: Python not found.
    echo Install Python 3.10+ and try again.
    pause
    exit /b 1
)

echo Using Python: !PYTHON_CMD!
!PYTHON_CMD! --version
echo.

if exist venv (
    set /p remove_venv="Existing venv found. Remove and recreate it? [Y/n]: "
    if /I not "!remove_venv!"=="n" (
        echo Removing existing venv...
        rmdir /s /q venv
    )
)

if not exist venv (
    echo Creating virtual environment...
    !PYTHON_CMD! -m venv venv
)

echo Activating venv...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate venv.
    pause
    exit /b 1
)

echo Upgrading pip/setuptools/wheel...
python -m pip install -U pip setuptools wheel
if errorlevel 1 (
    echo pip upgrade failed.
    pause
    exit /b 1
)

echo Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo requirements install failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Run:
echo   run_experiment.bat
pause
exit /b 0

