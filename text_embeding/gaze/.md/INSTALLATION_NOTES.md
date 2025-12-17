# 📦 Hướng dẫn cài đặt dependencies

## ✅ Đã cài đặt thành công

- ✅ `opencv-python` - Xử lý video/image
- ✅ `librosa` - Xử lý audio
- ✅ `soundfile` - Đọc/ghi audio files

## ⚠️ Vấn đề với MediaPipe

**MediaPipe chưa hỗ trợ Python 3.13**

Hiện tại bạn đang dùng Python 3.13, nhưng MediaPipe chỉ hỗ trợ Python 3.8-3.12.

### Giải pháp:

#### Option 1: Downgrade Python (Khuyến nghị)
```bash
# Cài Python 3.12 từ python.org
# Sau đó tạo virtual environment:
python3.12 -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Option 2: Chờ MediaPipe hỗ trợ Python 3.13
Theo dõi: https://github.com/google/mediapipe/issues

#### Option 3: Sử dụng Docker
```dockerfile
FROM python:3.12-slim
# ... rest of Dockerfile
```

### Kiểm tra Python version:
```bash
python --version
```

### Cài MediaPipe (sau khi có Python 3.11 hoặc 3.12):
```bash
pip install mediapipe
```

## 📝 Lưu ý

Code đã được cập nhật để hiển thị lỗi rõ ràng nếu MediaPipe chưa được cài đặt. API sẽ trả về HTTP 503 với thông báo hướng dẫn cài đặt.

## 🔧 Test cài đặt

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import librosa; print('Librosa:', librosa.__version__)"
python -c "import mediapipe; print('MediaPipe:', mediapipe.__version__)"  # Sẽ lỗi nếu chưa cài
```

