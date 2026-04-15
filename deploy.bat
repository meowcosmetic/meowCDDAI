@echo off
REM AI-Enhanced Gaze Tracking System — Windows Deployment Script
REM Usage: deploy.bat [--dev | --prod]

setlocal

set MODE=%1
if "%MODE%"=="" set MODE=--dev

echo ============================================================
echo  AI-Enhanced Gaze Tracking System Deployment
echo  Mode: %MODE%
echo ============================================================

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 or 3.12.
    exit /b 1
)

REM Activate virtual environment if present
if exist "venv312\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv312\Scripts\activate.bat
) else (
    echo WARNING: No virtual environment found at venv312\
    echo Run: python -m venv venv312 ^&^& venv312\Scripts\activate
)

REM Install / update dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

REM Copy config template if no config exists
if not exist "config.yaml" (
    echo Creating default config.yaml from template...
    copy config_template.yaml config.yaml
)

REM Run tests before starting
if "%MODE%"=="--prod" (
    echo Running test suite...
    python -m pytest ai_enhanced_gaze_tracking\tests\ -q --tb=short
    if errorlevel 1 (
        echo ERROR: Tests failed. Aborting deployment.
        exit /b 1
    )
    echo All tests passed.
)

REM Start the server
echo Starting server...
if "%MODE%"=="--prod" (
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
) else (
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
)

endlocal
