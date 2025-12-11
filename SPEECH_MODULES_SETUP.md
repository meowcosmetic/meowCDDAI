# Hướng dẫn cài đặt Speech Analysis Modules

## Vấn đề
Nếu bạn gặp lỗi: `librosa không được cài đặt. Vui lòng cài: pip install librosa soundfile`

## Giải pháp

### Cách 1: Sử dụng script tự động (Khuyến nghị)

1. **Chạy script cài đặt:**
   ```bash
   install_speech_modules.bat
   ```

2. **Chạy server với Python 3.12:**
   ```bash
   run_server_python312.bat
   ```

### Cách 2: Cài đặt thủ công

1. **Activate Python 3.12 virtual environment:**
   ```bash
   venv312\Scripts\activate
   ```

2. **Cài đặt các modules:**
   ```bash
   pip install librosa soundfile moviepy
   ```

3. **Kiểm tra cài đặt:**
   ```bash
   python test_librosa_install.py
   ```

4. **Chạy server:**
   ```bash
   python main.py
   ```

### Cách 3: Cài đặt từ requirements.txt

1. **Activate venv:**
   ```bash
   venv312\Scripts\activate
   ```

2. **Cài đặt tất cả dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Kiểm tra

Sau khi cài đặt, chạy:
```bash
python test_librosa_install.py
```

Nếu thấy:
```
✅ librosa: 0.x.x
✅ soundfile: 0.x.x
✅ moviepy: x.x.x
🎉 Tất cả modules đã được cài đặt thành công!
```

Thì bạn đã cài đặt thành công!

## Lưu ý quan trọng

⚠️ **BẮT BUỘC phải dùng Python 3.12 venv!**

- Server phải chạy với `venv312` (Python 3.12)
- Không dùng Python 3.13 vì MediaPipe không hỗ trợ
- Luôn activate venv trước khi chạy server hoặc test

## Troubleshooting

### Lỗi: "librosa không được cài đặt"
- Đảm bảo đã activate `venv312`
- Kiểm tra: `python --version` phải là Python 3.12.x
- Chạy lại: `pip install librosa soundfile moviepy`

### Lỗi khi cài đặt librosa
- Thử cài từng cái một:
  ```bash
  pip install librosa
  pip install soundfile
  pip install moviepy
  ```

### Server vẫn báo lỗi
- Đảm bảo server đang chạy với Python 3.12 venv
- Restart server sau khi cài đặt modules
- Kiểm tra: `python -c "import librosa"` trong venv312

