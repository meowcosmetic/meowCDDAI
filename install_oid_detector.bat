@echo off
echo ========================================
echo Cài đặt OID Detector (Open Images Dataset)
echo ========================================
echo.

REM Activate Python 3.12 venv nếu có
if exist venv312\Scripts\activate.bat (
    echo [1/2] Activating Python 3.12 virtual environment...
    call venv312\Scripts\activate.bat
    echo [OK] Virtual environment activated
    echo.
) else (
    echo [WARNING] venv312 không tìm thấy, sử dụng Python global
    echo.
)

REM Install ultralytics
echo [2/2] Installing ultralytics (YOLOv8)...
python -m pip install ultralytics>=8.0.0 --no-cache-dir
if %errorlevel% neq 0 (
    echo [ERROR] Không thể cài đặt ultralytics!
    pause
    exit /b 1
)
echo.

REM Verify installation
echo [3/3] Verifying installation...
python -c "from ultralytics import YOLO; print('✅ ultralytics:', YOLO.__version__ if hasattr(YOLO, '__version__') else 'installed')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Không thể verify, nhưng có thể đã cài đặt thành công.
    echo Hãy thử import thủ công: python -c "from ultralytics import YOLO"
) else (
    echo [OK] Ultralytics đã được cài đặt thành công!
)
echo.

echo ========================================
echo Hoàn tất!
echo ========================================
echo.
echo Lưu ý:
echo   - YOLOv8 OID model sẽ được download tự động lần đầu tiên sử dụng
echo   - Model size: 'n' (nano) - nhanh nhất, 's', 'm', 'l', 'x' - chính xác hơn
echo   - OID có 600 classes, bao gồm 'pen' và 'pencil' ✅
echo.
echo Cấu hình trong config:
echo   USE_OID_DATASET = True  # Bật OID detector
echo   OID_MODEL_SIZE = 'n'    # 'n', 's', 'm', 'l', 'x'
echo.
pause





