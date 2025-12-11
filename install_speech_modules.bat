@echo off
echo ========================================
echo Cài đặt Speech Analysis Modules
echo ========================================
echo.

REM Activate Python 3.12 venv
echo [1/4] Activating Python 3.12 virtual environment...
call venv312\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Không thể activate venv312!
    echo Vui lòng chạy setup_python312.bat trước.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [2/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install librosa, soundfile, moviepy
echo [3/4] Installing librosa, soundfile, moviepy...
echo Đang cài đặt (có thể mất vài phút)...
python -m pip install librosa soundfile moviepy --no-cache-dir
if %errorlevel% neq 0 (
    echo [ERROR] Không thể cài đặt modules!
    echo.
    echo Thử cài từng cái một:
    python -m pip install librosa
    python -m pip install soundfile
    python -m pip install moviepy
    pause
    exit /b 1
)
echo.

REM Verify installation
echo [4/4] Verifying installation...
python -c "import librosa; import soundfile; import moviepy; print('✅ librosa:', librosa.__version__); print('✅ soundfile:', soundfile.__version__); print('✅ moviepy:', moviepy.__version__); print(''); print('🎉 Tất cả modules đã được cài đặt thành công!')" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Không thể verify, nhưng có thể đã cài đặt thành công.
    echo Hãy thử import thủ công: python -c "import librosa"
    pause
)
echo.

echo ========================================
echo Hoàn tất!
echo ========================================
echo.
echo Bây giờ bạn có thể chạy:
echo   python test_speech_api.py
echo.
pause

