# Implementation Guide - Option 3 + Gaze Wandering

## 📋 Tổng quan

Implement:
1. **Option 3**: 3D Gaze Confidence-based focusing
2. **Gaze Wandering Detection**: Phát hiện "nhìn vô định"

## 🔧 Các module đã tạo

### ✅ Đã hoàn thành:
- `gaze_estimation_3d.py` - 3D gaze estimation với confidence
- `gaze_wandering.py` - Gaze wandering detector
- `config.py` - Updated với configs mới
- `models.py` - Updated với fields mới

## 📝 Implementation Steps

### Step 1: Import modules trong routes_screening_gaze.py

```python
# Thêm vào đầu file
from gaze_tracking import (
    GazeConfig, GPUManager, GazeEstimator3D, 
    GazeWanderingDetector, FocusTimeline
)
from gaze_tracking.object_detector import ObjectDetector
from gaze_tracking.face_detector import create_face_detector
```

### Step 2: Initialize trong analyze_gaze()

```python
# Khởi tạo config và detectors
config = GazeConfig()
gpu_manager = GPUManager()
face_detector = create_face_detector(use_mediapipe=not use_fallback)
object_detector = ObjectDetector(config, gpu_manager, enable_tracking=True)

# 3D Gaze Estimator
gaze_estimator_3d = None
if not use_fallback and MEDIAPIPE_AVAILABLE:
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gaze_estimator_3d = GazeEstimator3D(image_width=w, image_height=h)

# Focus Timeline
focus_timeline = FocusTimeline(
    stability_threshold=config.GAZE_STABILITY_THRESHOLD,
    min_focus_duration=config.MIN_FOCUSING_DURATION
)

# Gaze Wandering Detector
wandering_detector = GazeWanderingDetector(config)
```

### Step 3: Update logic trong vòng lặp xử lý frame

**Vị trí: MediaPipe mode (line ~1030-1200)**

```python
# TRONG vòng lặp while cap.isOpened():

# 1. Detect faces và objects
faces_info = face_detector.detect(frame)
tracked_objects = object_detector.detect(frame, frame_count=frame_count) if use_object_detection else []

# 2. Estimate 3D gaze (nếu có MediaPipe)
gaze_3d_result = None
if gaze_estimator_3d and child_face_info and child_face_info.get('all_landmarks'):
    object_id, confidence = gaze_estimator_3d.estimate_3d_gaze(
        child_face_info['all_landmarks'],
        tracked_objects
    )
    if object_id and confidence > config.MIN_3D_GAZE_CONFIDENCE:
        gaze_3d_result = (object_id, confidence)

# 3. Update Focus Timeline (với 3D gaze result)
focus_timeline.update(
    frame_count=frame_count,
    current_time=current_time,
    gaze_pos=gaze_pos_2d,  # 2D fallback
    tracked_objects=tracked_objects,
    fps=fps,
    gaze_3d_result=gaze_3d_result  # ✅ 3D gaze result
)

# 4. Tính ratios cho wandering detection
looking_at_object_ratio = sum(1 for pos in gaze_positions_window if len(pos) > 3 and pos[3]) / len(gaze_positions_window) if len(gaze_positions_window) > 0 else 0
looking_at_adult_ratio = sum(1 for pos in gaze_positions_window if pos[2]) / len(gaze_positions_window) if len(gaze_positions_window) > 0 else 0

# 5. Update Wandering Detector
wandering_detector.update(
    frame_count=frame_count,
    current_time=current_time,
    is_stable=is_stable,
    looking_at_object_ratio=looking_at_object_ratio,
    looking_at_adult_ratio=looking_at_adult_ratio,
    adult_face_exists=(adult_face_info is not None),
    gaze_offset_x=eye_offset_x,
    gaze_offset_y=eye_offset_y,
    fps=fps,
    gaze_3d_result=gaze_3d_result
)

# 6. Cập nhật is_valid_focusing với Option 3 logic
if config.USE_3D_GAZE_CONFIDENCE and gaze_3d_result:
    # Option 3: Dùng 3D gaze confidence
    object_id, confidence = gaze_3d_result
    if confidence > config.MIN_3D_GAZE_CONFIDENCE:
        is_valid_focusing = is_stable
    else:
        is_valid_focusing = False
else:
    # Fallback: Smart Mode (Option 1)
    is_valid_focusing = is_stable and (
        (adult_face_info and (
            looking_at_adult_ratio > config.MIN_OBJECT_FOCUS_RATIO or
            (config.ALLOW_CAMERA_FOCUS_WITH_ADULT and 
             looking_at_object_ratio < 0.3 and 
             abs(eye_offset_x) < config.CAMERA_FOCUS_THRESHOLD and 
             abs(eye_offset_y) < config.CAMERA_FOCUS_THRESHOLD)
        )) or
        (looking_at_object_ratio > config.MIN_OBJECT_FOCUS_RATIO)
    )
```

### Step 4: Finalize và tính metrics

```python
# Sau vòng lặp, trước return:

# Finalize timeline và wandering
focus_timeline.finalize(total_frames, total_duration, fps)
wandering_detector.finalize(total_frames, total_duration, fps)

# Tính wandering metrics
wandering_score, wandering_percentage = wandering_detector.calculate_wandering_score(total_frames)
wandering_timeline = wandering_detector.get_wandering_timeline()

# Update risk score với wandering
if wandering_percentage > 20:  # Nếu > 20% thời gian nhìn vô định
    risk_score = min(100, risk_score + wandering_score * 0.3)  # Tăng risk
```

### Step 5: Update Response

```python
return GazeAnalysisResponse(
    # ... existing fields ...
    focus_timeline=focus_timeline.get_timeline(),
    object_focus_stats=focus_timeline.get_object_stats(),
    pattern_analysis=focus_timeline.get_pattern_analysis(),
    # NEW: Wandering metrics
    gaze_wandering_score=round(wandering_score, 2),
    gaze_wandering_percentage=round(wandering_percentage, 2),
    wandering_periods=wandering_timeline
)
```

## 🎯 Logic Flow

```
Frame → Detect Faces & Objects
  ↓
Estimate 3D Gaze (nếu có MediaPipe)
  ↓
Update Focus Timeline (với 3D gaze result)
  ↓
Update Wandering Detector
  ↓
Determine is_valid_focusing:
  - Nếu có 3D gaze với confidence > threshold → Focus
  - Nếu không → Fallback Smart Mode
  ↓
Track focusing periods
```

## 📊 Wandering Detection Logic

```python
is_wandering = (
    is_stable and                                # Mắt không di chuyển
    looking_at_object_ratio < 0.2 and            # Hầu như không nhìn object
    looking_at_adult_ratio < 0.2 and             # Hầu như không nhìn adult
    (adult_face is None or looking_at_adult_ratio < 0.2) and
    abs(offset_x) < 0.2 and abs(offset_y) < 0.2  # Nhìn "thẳng vô máy"
    not has_3d_gaze_target                       # 3D gaze không detect target
)
```

## 📍 MediaPipe Face Mesh Landmarks

### Eye Landmark Indices

MediaPipe Face Mesh cung cấp 468 landmarks trên khuôn mặt. Các landmarks quan trọng cho gaze tracking:

```python
# Left eye landmarks (16 điểm quanh mắt trái)
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Right eye landmarks (16 điểm quanh mắt phải)
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Iris centers (khi refine_landmarks=True)
LEFT_EYE_CENTER = 468   # Left iris center
RIGHT_EYE_CENTER = 473  # Right iris center
```

**Giải thích:**
- **LEFT_EYE_INDICES / RIGHT_EYE_INDICES**: Các điểm landmark quanh mắt (góc trong, góc ngoài, trên, dưới)
  - Dùng để tính tâm mắt (eye center) bằng cách lấy trung bình các điểm
  - Hoặc tính Eye Aspect Ratio (EAR) để detect blink
  - **Lưu ý**: Trong code hiện tại, các indices này được khai báo nhưng không được sử dụng trực tiếp
  - Thay vào đó, code dùng `LEFT_EYE_CENTER` và `RIGHT_EYE_CENTER` (iris centers) khi `refine_landmarks=True`

- **LEFT_EYE_CENTER / RIGHT_EYE_CENTER**: Điểm trung tâm của iris (con ngươi)
  - Chính xác hơn so với tính trung bình từ eye indices
  - Chỉ có khi MediaPipe được khởi tạo với `refine_landmarks=True`
  - Được dùng để tính gaze direction chính xác hơn

**Có sử dụng config không?**
- ❌ **KHÔNG** - Đây là các chỉ số cố định từ MediaPipe Face Mesh, không thể thay đổi
- MediaPipe định nghĩa các landmarks này, không phải config của chúng ta

**Hiển thị landmarks trên video:**
- ✅ **CÓ THỂ** - Các landmarks có thể được hiển thị trên video khi xử lý
- Trong hàm `draw_annotations()`, truyền `face_landmarks` và `show_landmarks=True`
- **Màu sắc hiển thị:**
  - **Left eye landmarks** (LEFT_EYE_INDICES): Màu xanh lá (green dots)
  - **Right eye landmarks** (RIGHT_EYE_INDICES): Màu xanh dương (blue dots)
  - **Left eye center** (LEFT_EYE_CENTER = 468): Màu vàng, lớn hơn, có label "L"
  - **Right eye center** (RIGHT_EYE_CENTER = 473): Màu vàng, lớn hơn, có label "R"
- **Eye outlines**: Vẽ đường viền quanh mắt bằng cách nối các landmarks

**Ví dụ sử dụng:**
```python
annotated_frame = draw_annotations(
    frame,
    child_face=child_face_vis,
    face_landmarks=face_landmarks,  # MediaPipe face landmarks
    show_landmarks=True,  # Bật hiển thị landmarks
    # ... các tham số khác
)
```

## 📊 Tracking Variables

Các biến được khởi tạo trong hàm `analyze_gaze()` để theo dõi quá trình phân tích:

```python
# Gaze direction statistics
gaze_directions = {"left": 0, "right": 0, "center": 0, "up": 0, "down": 0}

# Frame counting
frame_count = 0                    # Tổng số frame đã xử lý
face_detected_count = 0            # Tổng số frame có phát hiện khuôn mặt
child_face_detected_count = 0      # Tổng số frame có phát hiện khuôn mặt trẻ
```

**Giải thích:**

1. **`gaze_directions`**: Dictionary đếm số frame theo từng hướng nhìn
   - `"left"`: Nhìn sang trái
   - `"right"`: Nhìn sang phải
   - `"center"`: Nhìn thẳng (về camera hoặc center)
   - `"up"`: Nhìn lên trên
   - `"down"`: Nhìn xuống dưới
   - **Mục đích**: Tính phần trăm thời gian nhìn theo từng hướng
   - **Công thức**: `percentage = (count / total_frames) * 100`

2. **`frame_count`**: Bộ đếm frame tổng
   - Tăng mỗi khi xử lý một frame
   - Dùng để tính thời gian: `time = frame_count / fps`
   - Dùng để tính phần trăm: `percentage = (value / frame_count) * 100`

3. **`face_detected_count`**: Đếm số frame có phát hiện khuôn mặt
   - Tăng khi `results.multi_face_landmarks` không rỗng
   - Dùng để tính tỷ lệ phát hiện: `detection_rate = face_detected_count / frame_count`

4. **`child_face_detected_count`**: Đếm số frame có phát hiện khuôn mặt trẻ
   - Tăng khi phát hiện được face của trẻ (face nhỏ nhất hoặc ở giữa frame)
   - Dùng để tính attention percentages:
     - `attention_to_person_percentage = (attention_to_person_frames / child_face_detected_count) * 100`
     - `attention_to_objects_percentage = (attention_to_objects_frames / child_face_detected_count) * 100`

**Có sử dụng config không?**
- ❌ **KHÔNG** - Đây là các biến runtime (runtime variables)
- Khởi tạo = 0 và tăng dần trong quá trình xử lý video
- Không phải là tham số cấu hình có thể thay đổi

**Các giá trị KHÔNG dùng config:**
- `LEFT_EYE_INDICES`, `RIGHT_EYE_INDICES` - MediaPipe landmarks cố định
- `gaze_directions`, `frame_count`, `face_detected_count`, `child_face_detected_count` - Runtime variables

**Các giá trị CÓ dùng config (từ `GazeConfig`):**
- `MIN_FOCUSING_DURATION` - Thời gian tối thiểu để coi là focusing (mặc định: 5.0 giây)
- `GAZE_STABILITY_THRESHOLD` - Ngưỡng ổn định gaze (mặc định: 0.05)
- `GAZE_STABILITY_RMS_THRESHOLD` - RMS threshold cho improved stability (mặc định: 0.02)
- `MAX_FRAME_WIDTH`, `MAX_FRAME_HEIGHT` - Kích thước frame tối đa để hiển thị
- Và nhiều config khác trong `gaze_tracking/config.py`

## ⚠️ Lưu ý

1. **3D Gaze**: Chỉ hoạt động với MediaPipe (không có fallback)
2. **Wandering Detection**: Cần window size đủ lớn (30 frames)
3. **Performance**: 3D gaze estimation có thể chậm hơn, nên chỉ dùng khi cần độ chính xác cao
4. **Config**: Có thể toggle `USE_3D_GAZE_CONFIDENCE` và `ENABLE_WANDERING_DETECTION`
5. **MediaPipe Landmarks**: Các indices là cố định, không thể config. Chỉ có thể toggle `refine_landmarks=True/False` khi khởi tạo MediaPipe FaceMesh

## 🧪 Testing

Test cases:
1. Video có objects → 3D gaze detect đúng
2. Video không có objects → Wandering được detect
3. Video có adult kế camera → Focus được tính đúng
4. Video trẻ nhìn vô định → Wandering score cao

