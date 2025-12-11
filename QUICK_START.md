# 🚀 Quick Start - Gaze Tracking API

## Bước 1: Khởi động Server

```bash
python main.py
```

Server sẽ chạy tại: **http://localhost:8102**

## Bước 2: Test API

### Cách 1: Dùng script test (Dễ nhất)

```bash
python test_gaze_api.py path/to/your/video.mp4
```

### Cách 2: Dùng curl (Windows PowerShell)

```powershell
curl.exe -X POST "http://localhost:8102/screening/gaze/analyze" `
  -F "video=@test_video.mp4" `
  -F "target_type=camera"
```

### Cách 3: Dùng Python requests

Tạo file `test.py`:
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

Chạy:
```bash
python test.py
```

### Cách 4: Dùng Postman hoặc Insomnia

1. Method: **POST**
2. URL: `http://localhost:8102/screening/gaze/analyze`
3. Body: chọn **form-data**
4. Thêm:
   - Key: `video`, Type: **File**, Value: chọn file video
   - Key: `target_type`, Type: **Text**, Value: `camera`
5. Click **Send**

## Bước 3: Xem API Documentation

Mở browser và truy cập:
- **Swagger UI**: http://localhost:8102/docs
- **ReDoc**: http://localhost:8102/redoc

Tại đây bạn có thể test trực tiếp trên browser!

## 📊 Response mẫu

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

## ⚠️ Lưu ý

- Video cần có face rõ ràng
- Xử lý có thể mất vài phút tùy độ dài video
- Đảm bảo đã cài đủ dependencies: `pip install -r requirements.txt`

