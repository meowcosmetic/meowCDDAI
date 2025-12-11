@echo off
echo ========================================
echo Cài đặt Gaze Tracking Dependencies
echo ========================================
echo.

REM Activate Python 3.12 venv nếu có
if exist venv312\Scripts\activate.bat (
    echo [1/3] Activating Python 3.12 virtual environment...
    call venv312\Scripts\activate.bat
    echo [OK] Virtual environment activated
    echo.
) else (
    echo [WARNING] venv312 không tìm thấy, sử dụng Python global
    echo.
)

REM Install dependencies
echo [2/3] Installing dependencies...
echo Đang cài đặt deep-sort-realtime và scipy...
python -m pip install deep-sort-realtime scipy --no-cache-dir
if %errorlevel% neq 0 (
    echo [ERROR] Không thể cài đặt dependencies!
    pause
    exit /b 1
)
echo.

REM Verify installation
echo [3/3] Verifying installation...
python -c "import deep_sort_realtime; import scipy; print('✅ deep-sort-realtime: installed'); print('✅ scipy:', scipy.__version__)" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Không thể verify, nhưng có thể đã cài đặt thành công.
    echo Hãy thử import thủ công: python -c "import deep_sort_realtime; import scipy"
) else (
    echo [OK] Tất cả dependencies đã được cài đặt thành công!
)
echo.

echo ========================================
echo Hoàn tất!
echo ========================================
echo.
echo Các thư viện đã cài đặt:
echo   - deep-sort-realtime: Object tracking
echo   - scipy: Scientific computing (cho 3D gaze estimation)
echo.
echo Bây giờ bạn có thể:
echo   - Chạy test: python test_gaze_api.py
echo   - Hoặc chạy server: python main.py
echo.
pause

