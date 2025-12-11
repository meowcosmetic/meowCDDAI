# Tóm tắt: Config Application Status

## ✅ Đã được apply

### 1. **MAX_FRAME_WIDTH** ✅
- **Config value**: 640 pixels (có thể thay đổi trong `config.py`)
- **Đã apply**: 
  - Resize frame trước khi processing (cả MediaPipe và OpenCV fallback)
  - Resize frame trước khi hiển thị (`cv2.imshow`)
- **Vị trí**: 
  - Line ~646 (OpenCV fallback)
  - Line ~820 (MediaPipe)
  - Line ~770 (display - fallback)
  - Line ~1317 (display - MediaPipe)

### 2. **FPS_DEFAULT** ✅
- **Config value**: 30
- **Đã apply**: 
  - Dùng `config.FPS_DEFAULT` nếu video không có FPS metadata
- **Vị trí**: Line ~535

### 3. **MIN_FOCUSING_DURATION** ✅
- **Config value**: 5.0 giây
- **Đã apply**: 
  - Dùng `config.MIN_FOCUSING_DURATION` để tính focusing window
- **Vị trí**: Line ~531

### 4. **GAZE_STABILITY_THRESHOLD** ✅
- **Config value**: 0.1
- **Đã apply**: 
  - Dùng `config.GAZE_STABILITY_THRESHOLD` để kiểm tra gaze stability
- **Vị trí**: Line ~532, ~707, ~1142

### 5. **OID_MODEL_SIZE** ✅
- **Config value**: 'm' (medium)
- **Đã apply**: 
  - Dùng `config.OID_MODEL_SIZE` khi khởi tạo OID detector
- **Vị trí**: Line ~568, ~1509

### 6. **Các config khác** ✅
- `MIN_3D_GAZE_CONFIDENCE`: Line ~1064, ~1148
- `USE_3D_GAZE_CONFIDENCE`: Line ~1145
- `MIN_OBJECT_FOCUS_RATIO`: Line ~1154
- `ALLOW_CAMERA_FOCUS_WITH_ADULT`: Line ~1155
- `CAMERA_FOCUS_THRESHOLD`: Line ~1156

## 🔧 Cách thay đổi config

### Thay đổi kích thước hiển thị video:
```python
# Trong text_embeding/gaze_tracking/config.py
MAX_FRAME_WIDTH: int = 1280  # Thay đổi giá trị này
```

**Giá trị khuyến nghị:**
- 640: Nhỏ, phù hợp màn hình nhỏ
- 1280: Vừa phải (mặc định)
- 1920: Lớn, cho màn hình lớn

### Thay đổi thời gian focusing:
```python
MIN_FOCUSING_DURATION: float = 3.0  # 3 giây thay vì 5 giây
```

### Thay đổi model size:
```python
OID_MODEL_SIZE: str = 's'  # 'n', 's', 'm', 'l', 'x'
```

## 📝 Lưu ý

1. **MAX_FRAME_WIDTH**: 
   - Frame sẽ được resize CẢ khi processing VÀ khi hiển thị
   - Giúp tăng tốc độ xử lý và tránh màn hình quá lớn

2. **Config fallback**: 
   - Nếu `config = None` (modules không available), sẽ dùng hardcoded values
   - Fallback values: MAX_FRAME_WIDTH = 1280, FPS = 30

3. **Restart server**: 
   - Cần restart server sau khi thay đổi config để áp dụng




