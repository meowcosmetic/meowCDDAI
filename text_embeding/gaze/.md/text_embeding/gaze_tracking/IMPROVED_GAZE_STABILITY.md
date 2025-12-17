# Improved Gaze Stability Calculation

## 🎯 Tổng quan các cải thiện

Module `gaze_stability.py` cung cấp tính toán gaze stability được cải thiện với:

1. ✅ **Normalization by interocular distance** - Chuẩn hóa theo khoảng cách giữa 2 mắt
2. ✅ **Head motion compensation** - Bù trừ chuyển động đầu
3. ✅ **Outlier removal** - Loại bỏ giật mắt, blink, missing data
4. ✅ **Smoothing** - Làm mượt dữ liệu
5. ✅ **RMS distance metric** - Metric dễ hiểu hơn variance
6. ✅ **Adaptive threshold** - Threshold tự điều chỉnh (optional)
7. ✅ **Window size optimization** - 100-300ms (3-10 frames tại 30fps)

---

## 📋 So sánh với công thức cũ

### Công thức cũ (đơn giản):
```python
positions_x = [pos[0] for pos in gaze_positions_window]
positions_y = [pos[1] for pos in gaze_positions_window]
variance_x = np.var(positions_x)
variance_y = np.var(positions_y)
total_variance = variance_x + variance_y
is_stable = total_variance < threshold
```

**Vấn đề:**
- ❌ Đơn vị pixel - khác camera/độ phân giải
- ❌ Không chuẩn hóa theo kích thước khuôn mặt
- ❌ Không loại bỏ chuyển động đầu
- ❌ Không loại bỏ outliers
- ❌ Variance khó diễn giải

### Công thức mới (improved):
```python
# 1. Normalize by interocular distance
positions_x = [x * mean_iod for x in positions_x]
positions_y = [y * mean_iod for y in positions_y]

# 2. Head motion compensation
compensated = compensate_head_motion(gaze_positions, head_poses)

# 3. Outlier removal
positions_x = remove_outliers(positions_x, z_threshold=2.5)
positions_y = remove_outliers(positions_y, z_threshold=2.5)

# 4. Smoothing
positions_x = smooth_values(positions_x, window_size=3)
positions_y = smooth_values(positions_y, window_size=3)

# 5. Calculate RMS distance (dễ hiểu hơn)
rms_distance = calculate_rms_distance(positions_x, positions_y)

# 6. Check stability
is_stable = rms_distance < rms_threshold
```

**Cải thiện:**
- ✅ Normalized by interocular distance
- ✅ Head motion compensated
- ✅ Outliers removed
- ✅ Smoothed
- ✅ RMS distance (dễ hiểu hơn)

---

## 🔧 Cấu hình

### Trong `config.py`:

```python
# Bật/tắt improved calculation
GAZE_STABILITY_USE_IMPROVED: bool = True

# Window size (milliseconds)
GAZE_STABILITY_WINDOW_MS: float = 200.0  # 100-300ms recommended

# RMS threshold (normalized by interocular distance)
GAZE_STABILITY_RMS_THRESHOLD: float = 0.02

# Head motion compensation
GAZE_STABILITY_USE_HEAD_COMPENSATION: bool = True

# Outlier removal
GAZE_STABILITY_USE_OUTLIER_REMOVAL: bool = True
GAZE_STABILITY_Z_THRESHOLD: float = 2.5

# Smoothing
GAZE_STABILITY_USE_SMOOTHING: bool = True
GAZE_STABILITY_SMOOTHING_WINDOW: int = 3

# Adaptive threshold
GAZE_STABILITY_ADAPTIVE_THRESHOLD: bool = False
```

---

## 💻 Cách sử dụng

### 1. Khởi tạo calculator:

```python
from gaze_tracking.gaze_stability import ImprovedGazeStabilityCalculator

calculator = ImprovedGazeStabilityCalculator(
    window_size_ms=200.0,      # 200ms window
    rms_threshold=0.02,        # RMS threshold
    z_threshold=2.5,           # Z-score threshold
    smoothing_window=3,        # Smoothing window
    use_head_compensation=True,
    use_outlier_removal=True,
    use_smoothing=True,
    adaptive_threshold=False
)
```

### 2. Tính stability mỗi frame:

```python
# Tính interocular distance
iod = calculate_interocular_distance(face_landmarks, w, h)

# Tính stability
result = calculator.calculate_stability(
    gaze_x=eye_offset_x,
    gaze_y=eye_offset_y,
    interocular_distance=iod,
    head_pose=head_pose_result,  # (yaw, pitch, roll) nếu có
    fps=fps,
    timestamp=current_time
)

# Kết quả
is_stable = result['is_stable']
rms_distance = result['rms_distance']
stability_score = result['stability_score']  # 0-1, cao hơn = ổn định hơn
```

---

## 📊 Kết quả trả về

```python
{
    'is_stable': bool,              # True nếu gaze ổn định
    'rms_distance': float,          # RMS distance (normalized)
    'variance': float,              # Variance (legacy, để so sánh)
    'stability_score': float,       # 0-1, cao hơn = ổn định hơn
    'threshold': float,             # Threshold được dùng
    'details': {
        'window_size': int,         # Số frames trong window
        'mean_iod': float,          # Mean interocular distance
        'head_compensation': bool,  # Có bù trừ head motion không
        'outliers_removed': bool,   # Có loại bỏ outliers không
        'smoothed': bool            # Có làm mượt không
    }
}
```

---

## 🎯 Metrics

### RMS Distance (Root Mean Square)
- **Ý nghĩa**: "Bán kính" dispersion - dễ hiểu hơn variance
- **Công thức**: `RMS = sqrt(mean(distances_from_center²))`
- **Đơn vị**: Normalized by interocular distance
- **Ví dụ**:
  - `RMS = 0.01` → Rất ổn định
  - `RMS = 0.02` → Ổn định (threshold)
  - `RMS = 0.05` → Không ổn định

### Stability Score
- **Ý nghĩa**: 0-1, cao hơn = ổn định hơn
- **Công thức**: `score = 1.0 - (rms_distance / max_rms)`
- **Ví dụ**:
  - `score = 1.0` → Hoàn toàn ổn định
  - `score = 0.5` → Vừa phải
  - `score = 0.0` → Không ổn định

---

## 🔄 Tích hợp vào code hiện tại

### Trong `routes_screening_gaze.py`:

```python
# Khởi tạo calculator (một lần)
if config and config.GAZE_STABILITY_USE_IMPROVED:
    stability_calculator = ImprovedGazeStabilityCalculator(
        window_size_ms=config.GAZE_STABILITY_WINDOW_MS,
        rms_threshold=config.GAZE_STABILITY_RMS_THRESHOLD,
        z_threshold=config.GAZE_STABILITY_Z_THRESHOLD,
        smoothing_window=config.GAZE_STABILITY_SMOOTHING_WINDOW,
        use_head_compensation=config.GAZE_STABILITY_USE_HEAD_COMPENSATION,
        use_outlier_removal=config.GAZE_STABILITY_USE_OUTLIER_REMOVAL,
        use_smoothing=config.GAZE_STABILITY_USE_SMOOTHING,
        adaptive_threshold=config.GAZE_STABILITY_ADAPTIVE_THRESHOLD
    )
else:
    stability_calculator = None

# Mỗi frame
if stability_calculator:
    # Tính interocular distance
    iod = calculate_interocular_distance(face_landmarks, w, h)
    
    # Tính stability
    stability_result = stability_calculator.calculate_stability(
        gaze_x=eye_offset_x,
        gaze_y=eye_offset_y,
        interocular_distance=iod,
        head_pose=head_pose_result,
        fps=fps,
        timestamp=current_time
    )
    
    is_stable = stability_result['is_stable']
    rms_distance = stability_result['rms_distance']
    stability_score = stability_result['stability_score']
else:
    # Fallback: dùng công thức cũ
    positions_x = [pos[0] for pos in gaze_positions_window]
    positions_y = [pos[1] for pos in gaze_positions_window]
    variance_x = np.var(positions_x)
    variance_y = np.var(positions_y)
    total_variance = variance_x + variance_y
    is_stable = total_variance < GAZE_STABILITY_THRESHOLD
```

---

## 📝 Lưu ý

1. **Interocular distance**: Cần face landmarks từ MediaPipe
2. **Head pose**: Cần 3D gaze estimation hoặc head pose estimation
3. **Window size**: 100-300ms được khuyến nghị (3-10 frames tại 30fps)
4. **RMS threshold**: Có thể cần calibrate cho từng camera/môi trường
5. **Adaptive threshold**: Chỉ bật nếu môi trường/camera thay đổi nhiều

---

## ✅ Kết luận

Improved gaze stability calculation cung cấp:
- ✅ Chính xác hơn (normalized, head compensated)
- ✅ Ổn định hơn (outlier removal, smoothing)
- ✅ Dễ hiểu hơn (RMS distance thay vì variance)
- ✅ Linh hoạt hơn (adaptive threshold, configurable)

Khuyến nghị: **Bật** `GAZE_STABILITY_USE_IMPROVED = True` để có kết quả tốt nhất.

