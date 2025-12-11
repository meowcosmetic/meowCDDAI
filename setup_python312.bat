@echo off
echo ========================================
echo Setup Python 3.12 cho Gaze API
echo ========================================
echo.

REM Kiểm tra Python 3.12
py -3.12 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.12 chưa được cài đặt!
    echo.
    echo Vui lòng cài Python 3.12 từ:
    echo https://www.python.org/downloads/release/python-3120/
    echo.
    echo Hoặc sử dụng winget:
    echo winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

echo [OK] Python 3.12 đã được cài đặt
py -3.12 --version
echo.

REM Tạo virtual environment
echo [1/3] Tạo virtual environment...
if exist venv312 (
    echo Virtual environment đã tồn tại, bỏ qua...
) else (
    py -3.12 -m venv venv312
    echo [OK] Đã tạo virtual environment
)
echo.

REM Activate và cài dependencies
echo [2/3] Cài đặt dependencies...
call venv312\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

REM Cài MediaPipe (quan trọng!)
echo [3/3] Cài đặt MediaPipe...
pip install mediapipe
echo.

echo ========================================
echo Setup hoàn tất!
echo ========================================
echo.
echo Để sử dụng:
echo   1. Activate venv: venv312\Scripts\activate
echo   2. Chạy server: python main.py
echo   3. Test API: python test_gaze_api.py
echo.
pause

