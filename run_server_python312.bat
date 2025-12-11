@echo off
echo ========================================
echo Chạy API Server với Python 3.12
echo ========================================
echo.

REM Activate Python 3.12 venv
echo [1/3] Activating Python 3.12 virtual environment...
call venv312\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Không thể activate venv312!
    echo Vui lòng chạy setup_python312.bat trước.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Verify librosa is installed
echo [2/3] Verifying librosa installation...
python -c "import librosa; print('✅ librosa:', librosa.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] librosa chưa được cài đặt. Đang cài đặt...
    python -m pip install librosa soundfile moviepy
    if %errorlevel% neq 0 (
        echo [ERROR] Không thể cài đặt librosa!
        pause
        exit /b 1
    )
    echo [OK] Đã cài đặt librosa, soundfile, moviepy
) else (
    echo [OK] librosa đã được cài đặt
)
echo.

REM Run server
echo [3/3] Starting API server...
echo.
python main.py
pause

