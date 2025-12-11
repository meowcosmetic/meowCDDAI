@echo off
echo ========================================
echo Copy YOLOv8 OID Model to Ultralytics Folder
echo ========================================
echo.

REM Tạo thư mục nếu chưa có
set WEIGHTS_DIR=%USERPROFILE%\.ultralytics\weights
if not exist "%WEIGHTS_DIR%" (
    echo [1/2] Creating weights directory...
    mkdir "%WEIGHTS_DIR%"
    echo [OK] Directory created: %WEIGHTS_DIR%
) else (
    echo [1/2] Weights directory exists: %WEIGHTS_DIR%
)
echo.

REM Kiểm tra file model
echo [2/2] Checking model file...
if exist "yolov8m-oidv7.pt" (
    echo [INFO] Found model file in current directory
    echo [INFO] Copying to: %WEIGHTS_DIR%\yolov8m-oidv7.pt
    copy /Y "yolov8m-oidv7.pt" "%WEIGHTS_DIR%\yolov8m-oidv7.pt"
    if %errorlevel% equ 0 (
        echo [OK] Model file copied successfully!
    ) else (
        echo [ERROR] Failed to copy model file!
        pause
        exit /b 1
    )
) else (
    echo [WARNING] Model file 'yolov8m-oidv7.pt' not found in current directory
    echo [INFO] Please place the model file in this directory first
    echo [INFO] Or manually copy to: %WEIGHTS_DIR%\yolov8m-oidv7.pt
)
echo.

echo ========================================
echo Hoàn tất!
echo ========================================
echo.
echo Model location:
echo   %WEIGHTS_DIR%\yolov8m-oidv7.pt
echo.
echo Bây giờ bạn có thể sử dụng OID detector!
echo.
pause

