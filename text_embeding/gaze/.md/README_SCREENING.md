# Hướng dẫn sử dụng Screening APIs

## 🚀 Khởi động Server

```bash
# Cài đặt dependencies (nếu chưa có)
pip install -r requirements.txt

# Chạy server
python main.py
```

Server sẽ chạy tại: `http://localhost:8102`

## 📡 API Endpoints

### 1. Gaze Tracking - `/screening/gaze/analyze`

Phân tích eye contact và gaze direction từ video.

#### Request:
```bash
curl -X POST "http://localhost:8102/screening/gaze/analyze" \
  -F "video=@test_video.mp4" \
  -F "target_type=camera"
```

#### Hoặc dùng Python:
```python
import requests

with open('test_video.mp4', 'rb') as f:
    files = {'video': f}
    data = {'target_type': 'camera'}
    response = requests.post(
        'http://localhost:8102/screening/gaze/analyze',
        files=files,
        data=data
    )
    print(response.json())
```

#### Response:
```json
{
  "eye_contact_percentage": 65.5,
  "gaze_direction_stats": {
    "left": 15.2,
    "right": 12.3,
    "center": 65.5,
    "up": 3.0,
    "down": 4.0
  },
  "total_frames": 300,
  "analyzed_duration": 10.0,
  "risk_score": 34.5
}
```

### 2. Facial Expression - `/screening/expression/analyze`

### 3. Pose & Movement - `/screening/pose/analyze`

### 4. Interaction Detection - `/screening/interaction/analyze`

### 5. Speech/Audio - `/screening/speech/analyze`

## 🧪 Test Script

Sử dụng script test có sẵn:

```bash
# Test Gaze API
python test_gaze_api.py path/to/video.mp4
```

## 📋 Yêu cầu Video/Audio

- **Video formats**: mp4, avi, mov, mkv
- **Audio formats**: wav, mp3, m4a
- **Khuyến nghị**: Video có độ phân giải tối thiểu 480p, có face rõ ràng

## 🔍 Kiểm tra API Documentation

Sau khi chạy server, truy cập:
- Swagger UI: `http://localhost:8102/docs`
- ReDoc: `http://localhost:8102/redoc`

## ⚠️ Lưu ý

- Xử lý video có thể mất thời gian tùy theo độ dài
- Đảm bảo có đủ RAM và CPU để xử lý MediaPipe
- Video files sẽ được lưu tạm trong quá trình xử lý

