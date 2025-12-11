@echo off
echo ========================================
echo Download YOLOv8 OID Model
echo ========================================
echo.

REM Activate venv312
if exist venv312\Scripts\activate.bat (
    echo [1/3] Activating venv312...
    call venv312\Scripts\activate.bat
    echo [OK] Virtual environment activated
    echo.
) else (
    echo [ERROR] venv312 không tìm thấy!
    echo Vui lòng tạo venv312 trước: python -m venv venv312
    pause
    exit /b 1
)

REM Check ultralytics
echo [2/3] Checking ultralytics...
python -c "from ultralytics import YOLO" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Ultralytics chưa được cài đặt!
    echo [INFO] Đang cài đặt ultralytics...
    python -m pip install ultralytics>=8.0.0 --no-cache-dir
    if %errorlevel% neq 0 (
        echo [ERROR] Không thể cài đặt ultralytics!
        pause
        exit /b 1
    )
    echo [OK] Ultralytics đã được cài đặt
) else (
    echo [OK] Ultralytics đã được cài đặt
)
echo.

REM Download model
echo [3/3] Downloading YOLOv8 OID model (yolov8m-oidv7.pt)...
echo [INFO] Model sẽ được download tự động lần đầu tiên...
echo [INFO] Kích thước: ~52MB
echo.
python -c "from ultralytics import YOLO; print('Đang download model...'); model = YOLO('yolov8m-oidv7.pt'); print('✅ Model đã được download và load thành công!')"
if %errorlevel% neq 0 (
    echo [ERROR] Không thể download model!
    echo [INFO] Có thể do:
    echo   - Không có kết nối internet
    echo   - Firewall chặn download
    echo   - Ultralytics chưa được cài đặt đúng
    pause
    exit /b 1
)
echo.

echo ========================================
echo Hoàn tất!
echo ========================================
echo.
echo Model đã được download vào:
echo   %USERPROFILE%\.ultralytics\weights\yolov8m-oidv7.pt
echo.
pause

