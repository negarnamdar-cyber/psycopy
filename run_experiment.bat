@echo off
REM Run the Speech Gating Experiment (venv only)
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist venv (
    echo Using venv environment
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo Failed to activate venv.
        pause
        exit /b 1
    )
    echo Starting Speech Gating Experiment...
    python main.py
    set "EXIT_CODE=%ERRORLEVEL%"
    deactivate
    if not "%EXIT_CODE%"=="0" (
        echo.
        echo Experiment failed with exit code %EXIT_CODE%.
        pause
    )
    exit /b %EXIT_CODE%
)

echo Error: No Python environment found
echo.
echo Please run setup first:
echo   setup_venv.bat
echo.
echo Or manually create an environment:
echo   python -m venv venv ^&^& venv\Scripts\activate.bat ^&^& pip install -r requirements.txt ^&^& python main.py
pause
exit /b 1

