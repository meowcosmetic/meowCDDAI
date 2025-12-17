# Giải thích: Config xác định "điểm dừng của mắt"

## 🎯 Improved Gaze Stability Calculator (MỚI)

### Tổng quan:
Hệ thống hiện tại sử dụng **Improved Gaze Stability Calculator** với các cải thiện:
- ✅ Normalization by interocular distance
- ✅ Head motion compensation
- ✅ Outlier removal
- ✅ Smoothing
- ✅ RMS distance metric (thay vì variance)

### Config chính: GAZE_STABILITY_USE_IMPROVED

```python
# text_embeding/gaze_tracking/config.py
GAZE_STABILITY_USE_IMPROVED: bool = True  # Bật/tắt improved calculator
```

**Mặc định**: `True` - Sử dụng improved calculator
**Nếu `False`**: Sẽ dùng công thức cũ (variance-based) - **KHÔNG KHUYẾN NGHỊ**

---

## 📊 Config Improved Calculator

### 1. GAZE_STABILITY_RMS_THRESHOLD

```python
GAZE_STABILITY_RMS_THRESHOLD: float = 0.02
```

**Cách hoạt động:**
- Ngưỡng **RMS distance** (normalized by interocular distance) để xác định mắt có "dừng" không
- Nếu `rms_distance < GAZE_STABILITY_RMS_THRESHOLD` → **Mắt đang "dừng"** (ổn định, đang focus)
- Nếu `rms_distance >= GAZE_STABILITY_RMS_THRESHOLD` → **Mắt đang di chuyển** (không ổn định)

**Giá trị:**
- **0.02** (mặc định) - Đã được normalize theo interocular distance
- Đơn vị: normalized (0-1)
  - 0 = hoàn toàn ổn định (mắt hoàn toàn dừng)
  - 1 = rất không ổn định (mắt di chuyển nhiều)

**Ví dụ:**
- **0.01**: Rất nghiêm ngặt, chỉ tính khi mắt cực kỳ ổn định
- **0.02**: Vừa phải (mặc định)
- **0.05**: Dễ dãi hơn, chấp nhận dao động lớn hơn

### 2. GAZE_STABILITY_WINDOW_MS

```python
GAZE_STABILITY_WINDOW_MS: float = 200.0  # milliseconds
```

**Cách hoạt động:**
- Kích thước **sliding window** tính bằng milliseconds
- **Biến config được sử dụng:** `config.GAZE_STABILITY_WINDOW_MS`
- Window càng lớn → tính toán ổn định hơn nhưng phản ứng chậm hơn
- Window càng nhỏ → phản ứng nhanh hơn nhưng dễ bị nhiễu

**Công thức chuyển đổi:**
```python
# Trong code (gaze_stability.py):
window_size_frames = max(3, int(self.window_size_ms * fps / 1000.0))

# Ví dụ với GAZE_STABILITY_WINDOW_MS = 200.0 và fps = 30:
window_size_frames = max(3, int(200.0 * 30 / 1000.0))
window_size_frames = max(3, int(6.0))
window_size_frames = 6 frames
```

**Giá trị khuyến nghị:**
- **100-300ms** (3-10 frames tại 30fps)
- **200ms** (mặc định) = ~6 frames tại 30fps

**Ví dụ với FPS = 30:**
- 100ms = 3 frames (phản ứng nhanh)
- 200ms = 6 frames (cân bằng - mặc định)
- 300ms = 9 frames (ổn định hơn)

**Nơi sử dụng trong code:**
```python
# text_embeding/routes_screening_gaze.py (dòng ~638)
stability_calculator = ImprovedGazeStabilityCalculator(
    window_size_ms=config.GAZE_STABILITY_WINDOW_MS,  # ← Biến config này
    rms_threshold=config.GAZE_STABILITY_RMS_THRESHOLD,
    z_threshold=config.GAZE_STABILITY_Z_THRESHOLD,
    smoothing_window=config.GAZE_STABILITY_SMOOTHING_WINDOW,
    use_head_compensation=config.GAZE_STABILITY_USE_HEAD_COMPENSATION,
    use_outlier_removal=config.GAZE_STABILITY_USE_OUTLIER_REMOVAL,
    use_smoothing=config.GAZE_STABILITY_USE_SMOOTHING,
    adaptive_threshold=config.GAZE_STABILITY_ADAPTIVE_THRESHOLD
)
```

### 3. GAZE_STABILITY_USE_HEAD_COMPENSATION

```python
GAZE_STABILITY_USE_HEAD_COMPENSATION: bool = True
```

**Cách hoạt động:**
- Bù trừ chuyển động đầu khi tính gaze stability
- Nếu `True`: Loại bỏ ảnh hưởng của head motion (đầu xoay/nghiêng)
- Nếu `False`: Không bù trừ (có thể tăng false positive khi đầu di chuyển)

**Khuyến nghị:** `True` (mặc định)

### 4. GAZE_STABILITY_USE_OUTLIER_REMOVAL

```python
GAZE_STABILITY_USE_OUTLIER_REMOVAL: bool = True
GAZE_STABILITY_Z_THRESHOLD: float = 2.0
```

**Cách hoạt động:**
- Loại bỏ outliers (giật mắt, blink, missing data) bằng Z-score
- `GAZE_STABILITY_Z_THRESHOLD`: Ngưỡng Z-score để coi là outlier
  - Giá trị càng nhỏ → loại bỏ nhiều hơn (nghiêm ngặt hơn)
  - Giá trị càng lớn → loại bỏ ít hơn (dễ dãi hơn)

**Giá trị:**
- **2.0** (mặc định) - Loại bỏ ~5% outliers
- **2.5** - Loại bỏ ~1% outliers (ít hơn)
- **1.5** - Loại bỏ ~13% outliers (nhiều hơn)

**Khuyến nghị:** `True` với `Z_THRESHOLD = 2.0`

### 5. GAZE_STABILITY_USE_SMOOTHING

```python
GAZE_STABILITY_USE_SMOOTHING: bool = True
GAZE_STABILITY_SMOOTHING_WINDOW: int = 3
```

**Cách hoạt động:**
- Làm mượt dữ liệu bằng moving average
- `GAZE_STABILITY_SMOOTHING_WINDOW`: Kích thước window cho smoothing
  - Giá trị càng lớn → mượt hơn nhưng phản ứng chậm hơn
  - Giá trị càng nhỏ → phản ứng nhanh hơn nhưng ít mượt hơn

**Giá trị:**
- **3** (mặc định) - Cân bằng tốt
- **5** - Mượt hơn
- **1** - Không smoothing (tắt smoothing)

**Khuyến nghị:** `True` với `SMOOTHING_WINDOW = 3`

### 6. GAZE_STABILITY_ADAPTIVE_THRESHOLD

```python
GAZE_STABILITY_ADAPTIVE_THRESHOLD: bool = False
```

**Cách hoạt động:**
- Tự động điều chỉnh threshold dựa trên lịch sử RMS values
- Nếu `True`: Threshold sẽ tự điều chỉnh theo môi trường/camera
- Nếu `False`: Dùng threshold cố định (`GAZE_STABILITY_RMS_THRESHOLD`)

**Khuyến nghị:** `False` (mặc định) - Chỉ bật nếu môi trường/camera thay đổi nhiều

---

## 📋 Công thức Improved Calculator

### ⚠️ Quan trọng: RMS Distance được tính trên SLIDING WINDOW

**RMS distance KHÔNG được tính trên toàn bộ video**, mà được tính trên **sliding window** (khoảng 200ms = ~6 frames tại 30fps).

**Cách hoạt động:**
1. Mỗi frame, calculator lấy `window_size_frames` frames gần nhất
2. Tính RMS distance trên các frames trong window đó
3. Window di chuyển theo từng frame (sliding window)

**Ví dụ với window = 6 frames:**
```
Frame 10: Tính RMS trên frames [5, 6, 7, 8, 9, 10]
Frame 11: Tính RMS trên frames [6, 7, 8, 9, 10, 11]
Frame 12: Tính RMS trên frames [7, 8, 9, 10, 11, 12]
...
```

### Công thức tính RMS trong window:

```python
# Mỗi frame, lấy dữ liệu trong window
window_data = gaze_window[-window_size_frames:]  # Ví dụ: 6 frames gần nhất

# 1. Normalize by interocular distance
normalized_x = [gaze_x * interocular_distance for gaze_x in window_data]
normalized_y = [gaze_y * interocular_distance for gaze_y in window_data]

# 2. Head motion compensation (nếu bật)
compensated = compensate_head_motion(normalized_positions, head_poses)

# 3. Outlier removal (nếu bật)
filtered = remove_outliers_zscore(compensated, z_threshold=2.0)

# 4. Smoothing (nếu bật)
smoothed = apply_smoothing_moving_average(filtered, window_size=3)

# 5. Calculate RMS distance trong window
center_x = mean(smoothed_x)
center_y = mean(smoothed_y)
distances = [sqrt((x - center_x)^2 + (y - center_y)^2) for x, y in zip(smoothed_x, smoothed_y)]
rms_distance = sqrt(mean(distances^2))

# 6. Normalize by interocular distance
rms_distance_normalized = rms_distance / mean_interocular_distance

# 7. Check stability
is_stable = rms_distance_normalized < GAZE_STABILITY_RMS_THRESHOLD
```

**Kết quả:**
- Mỗi frame có một giá trị `rms_distance` riêng (tính trên window của frame đó)
- `rms_distance` thay đổi theo từng frame (vì window di chuyển)
- Giá trị `rms_distance` trong API response là giá trị **cuối cùng** (frame cuối cùng của video)

---

## 🔄 Legacy Config (Công thức cũ - KHÔNG KHUYẾN NGHỊ)

### GAZE_STABILITY_THRESHOLD (Legacy)

```python
# text_embeding/gaze_tracking/config.py
GAZE_STABILITY_THRESHOLD: float = 0.05  # CHỈ DÙNG KHI GAZE_STABILITY_USE_IMPROVED = False
```

**Lưu ý:** Config này chỉ được dùng khi `GAZE_STABILITY_USE_IMPROVED = False` (không khuyến nghị)

**Công thức cũ:**
```python
# Tính variance của gaze positions
positions_x = [gaze_x_1, gaze_x_2, ..., gaze_x_n]
positions_y = [gaze_y_1, gaze_y_2, ..., gaze_y_n]

variance_x = var(positions_x)
variance_y = var(positions_y)
total_variance = variance_x + variance_y

# Kiểm tra "điểm dừng"
is_stable = total_variance < GAZE_STABILITY_THRESHOLD
```

**Vấn đề với công thức cũ:**
- ❌ Đơn vị pixel - khác camera/độ phân giải
- ❌ Không chuẩn hóa theo kích thước khuôn mặt
- ❌ Không loại bỏ chuyển động đầu
- ❌ Không loại bỏ outliers
- ❌ Variance khó diễn giải

---

## ⏱️ Config thời gian: MIN_FOCUSING_DURATION

### Vị trí:
```python
MIN_FOCUSING_DURATION: float = 0.5  # giây
```

### Cách hoạt động:
- Thời gian tối thiểu để coi là trẻ đang "focusing" vào một đối tượng
- Mắt phải "dừng" (stable) trong thời gian >= giá trị này mới được tính là focusing
- **Lưu ý:** Giá trị này được tính bằng số frames trong `FOCUSING_WINDOW_SIZE`

### Ví dụ:
- Nếu `MIN_FOCUSING_DURATION = 0.5` và `FOCUSING_WINDOW_SIZE = 15` (frames):
  - Mắt phải dừng ít nhất 0.5 giây (15 frames tại 30fps) mới được tính là focusing
  - Nếu chỉ dừng 0.3 giây → không tính là focusing

### FOCUSING_WINDOW_SIZE

```python
FOCUSING_WINDOW_SIZE: int = 15  # frames
```

**Cách hoạt động:**
- Số frames tối thiểu trong window để tính focusing
- Tương ứng với `MIN_FOCUSING_DURATION` tại FPS hiện tại
- Ví dụ: 15 frames tại 30fps = 0.5 giây


---

## 🔍 Config khoảng cách: LOOKING_AT_OBJECT_THRESHOLD

### Vị trí:
```python
LOOKING_AT_OBJECT_THRESHOLD: float = 0.6
```

### Cách hoạt động:
- Ngưỡng để xác định trẻ có đang nhìn vào một object không
- Tính khoảng cách giữa gaze position và object center
- Nếu `distance < threshold` → đang nhìn vào object

### Công thức:
```python
distance = sqrt((gaze_x - object_center_x)^2 + (gaze_y - object_center_y)^2)
is_looking_at_object = distance < LOOKING_AT_OBJECT_THRESHOLD
```

---

## 📝 Tóm tắt: Các config xác định "điểm dừng của mắt"

### Improved Calculator (KHUYẾN NGHỊ)

| Config | Giá trị | Mục đích |
|--------|---------|----------|
| **GAZE_STABILITY_USE_IMPROVED** | `True` | Bật/tắt improved calculator |
| **GAZE_STABILITY_RMS_THRESHOLD** | 0.02 | Ngưỡng RMS distance (normalized) để xác định mắt có "dừng" không |
| **GAZE_STABILITY_WINDOW_MS** | 200.0 ms | Kích thước window tính bằng milliseconds |
| **GAZE_STABILITY_USE_HEAD_COMPENSATION** | `True` | Bù trừ chuyển động đầu |
| **GAZE_STABILITY_USE_OUTLIER_REMOVAL** | `True` | Loại bỏ outliers |
| **GAZE_STABILITY_Z_THRESHOLD** | 2.0 | Ngưỡng Z-score cho outlier removal |
| **GAZE_STABILITY_USE_SMOOTHING** | `True` | Làm mượt dữ liệu |
| **GAZE_STABILITY_SMOOTHING_WINDOW** | 3 | Kích thước window cho smoothing |
| **GAZE_STABILITY_ADAPTIVE_THRESHOLD** | `False` | Tự động điều chỉnh threshold |

### Legacy Config (KHÔNG KHUYẾN NGHỊ)

| Config | Giá trị | Mục đích |
|--------|---------|----------|
| **GAZE_STABILITY_THRESHOLD** | 0.05 | Ngưỡng variance (chỉ dùng khi `USE_IMPROVED = False`) |

### Config khác

| Config | Giá trị | Mục đích |
|--------|---------|----------|
| **FOCUSING_WINDOW_SIZE** | 15 frames | Số frames tối thiểu để tính focusing |
| **MIN_FOCUSING_DURATION** | 0.5 giây | Thời gian tối thiểu để coi là focusing |
| **LOOKING_AT_OBJECT_THRESHOLD** | 0.6 | Ngưỡng khoảng cách để xác định nhìn vào object |

---

## 🛠️ Cách điều chỉnh

### Để mắt "dừng" dễ hơn (ít nghiêm ngặt hơn):
```python
GAZE_STABILITY_RMS_THRESHOLD: float = 0.05  # Tăng từ 0.02 lên 0.05
```

### Để mắt "dừng" khó hơn (nghiêm ngặt hơn):
```python
GAZE_STABILITY_RMS_THRESHOLD: float = 0.01  # Giảm từ 0.02 xuống 0.01
```

### Để phản ứng nhanh hơn:
```python
GAZE_STABILITY_WINDOW_MS: float = 100.0  # Giảm từ 200ms xuống 100ms (3 frames tại 30fps)
```

### Để tính toán ổn định hơn:
```python
GAZE_STABILITY_WINDOW_MS: float = 300.0  # Tăng từ 200ms lên 300ms (9 frames tại 30fps)
```

### Để loại bỏ nhiều outliers hơn (nghiêm ngặt hơn):
```python
GAZE_STABILITY_Z_THRESHOLD: float = 1.5  # Giảm từ 2.0 xuống 1.5
```

### Để làm mượt nhiều hơn:
```python
GAZE_STABILITY_SMOOTHING_WINDOW: int = 5  # Tăng từ 3 lên 5
```

### Để tắt head compensation (nếu không có head pose):
```python
GAZE_STABILITY_USE_HEAD_COMPENSATION: bool = False
```

---

## 💡 Lưu ý

### Improved Calculator (KHUYẾN NGHỊ)

1. **GAZE_STABILITY_RMS_THRESHOLD** là config quan trọng nhất để xác định "điểm dừng"
2. Giá trị nhỏ hơn → nghiêm ngặt hơn (chỉ tính khi mắt rất ổn định)
3. Giá trị lớn hơn → dễ dãi hơn (chấp nhận dao động lớn hơn)
4. Nên điều chỉnh cùng với **GAZE_STABILITY_WINDOW_MS** để có kết quả tốt nhất
5. **Head compensation** cần head pose từ 3D gaze estimation - nếu không có, nên tắt
6. **Outlier removal** giúp loại bỏ giật mắt, blink - nên bật
7. **Smoothing** giúp làm mượt dữ liệu - nên bật với window size = 3
8. **Adaptive threshold** chỉ nên bật nếu môi trường/camera thay đổi nhiều

### Dependencies

- **Interocular distance**: Cần face landmarks từ MediaPipe
- **Head pose**: Cần 3D gaze estimation hoặc head pose estimation (cho head compensation)
- **FPS**: Cần biết FPS của video để tính window size chính xác

### Calibration

- **RMS threshold** có thể cần calibrate cho từng camera/môi trường
- Nên test với video mẫu để tìm giá trị phù hợp
- Có thể dùng **adaptive threshold** nếu môi trường thay đổi nhiều

