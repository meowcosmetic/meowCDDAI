@echo off
echo ========================================
echo Chạy Gaze API Test với Python 3.12
echo ========================================
echo.

REM Kiểm tra virtual environment
if not exist venv312 (
    echo [ERROR] Virtual environment chưa được tạo!
    echo Chạy: setup_python312.bat
    pause
    exit /b 1
)

REM Activate và chạy test
call venv312\Scripts\activate.bat
python test_gaze_api.py
pause

