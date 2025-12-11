@echo off
REM Test camera gaze với venv312
echo ========================================
echo Test Camera Gaze Analysis
echo ========================================
echo.

REM Activate venv312 và chạy test
call venv312\Scripts\activate.bat
python test_camera_gaze.py

pause


