# Giải thích: gaze_x là gì?

## 📚 Định nghĩa đơn giản

**gaze_x** là **tọa độ X** (theo chiều ngang) của vị trí mà mắt đang nhìn vào trong frame.

- **gaze_x** = Vị trí ngang của điểm nhìn
- **gaze_y** = Vị trí dọc của điểm nhìn

Cùng nhau, `(gaze_x, gaze_y)` tạo thành **gaze position** - vị trí mà mắt đang nhìn vào.

---

## 🎯 Hai cách biểu diễn gaze_x

### 1. **Normalized Offset** (Offset chuẩn hóa) - Phổ biến nhất

**Đơn vị**: Normalized (-1.0 đến 1.0 hoặc 0.0 đến 1.0)

**Ý nghĩa**: 
- `gaze_x = 0.0` → Nhìn vào **giữa frame** (center)
- `gaze_x < 0.0` → Nhìn sang **trái**
- `gaze_x > 0.0` → Nhìn sang **phải**
- Giá trị tuyệt đối càng lớn → Nhìn càng xa center

**Ví dụ:**
```
gaze_x = -0.5  → Nhìn sang trái, cách center 50%
gaze_x = 0.0   → Nhìn vào center
gaze_x = 0.5   → Nhìn sang phải, cách center 50%
gaze_x = 1.0   → Nhìn sang phải, ở rìa frame
```

### 2. **Absolute Position** (Vị trí tuyệt đối)

**Đơn vị**: Pixels

**Ý nghĩa**: 
- `gaze_x = 0` → Rìa trái của frame
- `gaze_x = width/2` → Giữa frame
- `gaze_x = width` → Rìa phải của frame

**Ví dụ với frame width = 640:**
```
gaze_x = 0     → Rìa trái
gaze_x = 320   → Giữa frame
gaze_x = 640   → Rìa phải
```

---

## 💻 Cách tính gaze_x trong code

### Phương pháp 1: MediaPipe (Chính xác)

```python
# 1. Lấy vị trí iris (con ngươi)
LEFT_IRIS = [474, 475, 476, 477]
left_iris_x = sum(landmark[i].x for i in LEFT_IRIS) / len(LEFT_IRIS)

# 2. Lấy vị trí center của mắt
LEFT_EYE_CENTER = [33, 7, 163, ...]
left_eye_center_x = sum(landmark[i].x for i in LEFT_EYE_CENTER) / len(LEFT_EYE_CENTER)

# 3. Tính offset (iris so với eye center)
left_gaze_x = (left_iris_x - left_eye_center_x) * frame_width

# 4. Làm tương tự cho mắt phải và lấy trung bình
gaze_x = (left_gaze_x + right_gaze_x) / 2

# 5. Normalize (chuẩn hóa)
gaze_magnitude = sqrt(gaze_x² + gaze_y²)
gaze_x = gaze_x / gaze_magnitude  # Normalized
```

**Giải thích:**
- Iris (con ngươi) di chuyển trong mắt → cho biết hướng nhìn
- So sánh vị trí iris với center của mắt → tính offset
- Normalize để không phụ thuộc vào kích thước frame

### Phương pháp 2: OpenCV Fallback (Đơn giản)

```python
# 1. Lấy vị trí center của face
face_center_x = x + face_width / 2

# 2. Tính offset so với center frame
frame_center_x = frame_width / 2
offset_x = (face_center_x - frame_center_x) / (frame_width / 2)

# 3. gaze_x = offset_x (normalized)
gaze_x = offset_x
```

**Giải thích:**
- Giả định: Nếu face ở giữa frame → nhìn vào center
- Nếu face lệch trái → nhìn sang trái
- Đơn giản nhưng kém chính xác hơn MediaPipe

---

## 📊 Ví dụ cụ thể

### Ví dụ 1: Nhìn vào center

```
Frame width = 640 pixels
Face ở giữa frame

MediaPipe:
- left_iris_x = 0.5 (normalized)
- left_eye_center_x = 0.5 (normalized)
- left_gaze_x = (0.5 - 0.5) * 640 = 0 pixels
- gaze_x (normalized) = 0.0

→ gaze_x = 0.0 → Nhìn vào center ✅
```

### Ví dụ 2: Nhìn sang trái

```
Frame width = 640 pixels
Iris lệch sang trái so với eye center

MediaPipe:
- left_iris_x = 0.4 (normalized)
- left_eye_center_x = 0.5 (normalized)
- left_gaze_x = (0.4 - 0.5) * 640 = -64 pixels
- gaze_x (normalized) = -0.2

→ gaze_x = -0.2 → Nhìn sang trái ✅
```

### Ví dụ 3: Nhìn sang phải

```
Frame width = 640 pixels
Iris lệch sang phải so với eye center

MediaPipe:
- left_iris_x = 0.6 (normalized)
- left_eye_center_x = 0.5 (normalized)
- left_gaze_x = (0.6 - 0.5) * 640 = +64 pixels
- gaze_x (normalized) = +0.2

→ gaze_x = +0.2 → Nhìn sang phải ✅
```

---

## 🎨 Minh họa bằng hình ảnh

### gaze_x trong frame:

```
Frame (width = 640):

gaze_x = -1.0    gaze_x = 0.0    gaze_x = +1.0
   |                  |                  |
   ▼                  ▼                  ▼
┌─────────────────────────────────────────┐
│  ← Trái      Center      Phải →         │
│                                         │
│    ●                    ●              │
│  (Nhìn trái)        (Nhìn phải)         │
│                                         │
│              ●                          │
│         (Nhìn center)                   │
└─────────────────────────────────────────┘
```

### gaze_x trong mắt:

```
Mắt trái (nhìn từ trên xuống):

Eye Center    Iris Position    gaze_x
     |              |            |
     ▼              ▼            ▼
    ┌─────────────┐
    │      ●      │  ← Iris ở center → gaze_x = 0.0
    │   (center)  │
    └─────────────┘

    ┌─────────────┐
    │  ●          │  ← Iris lệch trái → gaze_x < 0.0
    │ (trái)      │
    └─────────────┘

    ┌─────────────┐
    │          ●  │  ← Iris lệch phải → gaze_x > 0.0
    │    (phải)   │
    └─────────────┘
```

---

## 🔢 Phạm vi giá trị

### Normalized Offset (Phổ biến):
- **-1.0 đến +1.0**: Offset từ center
  - `-1.0` = Rìa trái
  - `0.0` = Center
  - `+1.0` = Rìa phải

### Absolute Position:
- **0 đến frame_width**: Vị trí pixel
  - `0` = Rìa trái
  - `width/2` = Center
  - `width` = Rìa phải

---

## 💻 Sử dụng trong code

### Lưu vào window để tính stability:

```python
# Mỗi frame, lưu gaze position
gaze_positions_window.append((offset_x, offset_y, ...))

# Lấy tất cả gaze_x trong window
positions_x = [pos[0] for pos in gaze_positions_window]

# Tính variance để kiểm tra stability
variance_x = np.var(positions_x)
```

### Kiểm tra nhìn vào object:

```python
# Lấy gaze position
gaze_x, gaze_y = child_gaze_abs_pos

# Lấy object center
obj_center_x = (bbox[0] + bbox[2]) / 2

# Tính khoảng cách
distance_x = abs(gaze_x - obj_center_x)

# Kiểm tra có nhìn vào object không
if distance_x < threshold:
    is_looking_at_object = True
```

---

## 📝 Tóm tắt

| Khái niệm | Ý nghĩa | Ví dụ |
|-----------|---------|-------|
| **gaze_x** | Tọa độ X của vị trí nhìn | `0.0` = center, `-0.5` = trái, `+0.5` = phải |
| **gaze_y** | Tọa độ Y của vị trí nhìn | `0.0` = center, `-0.5` = trên, `+0.5` = dưới |
| **gaze position** | `(gaze_x, gaze_y)` | Vị trí 2D mà mắt đang nhìn |
| **Normalized** | Giá trị từ -1.0 đến +1.0 | Không phụ thuộc kích thước frame |
| **Absolute** | Giá trị pixel (0 đến width) | Phụ thuộc kích thước frame |

---

## 🎯 Mối quan hệ với variance

Khi tính **variance của gaze_x**:

```python
# Lấy các gaze_x trong window (ví dụ: 30 frames)
positions_x = [0.50, 0.51, 0.49, 0.50, 0.51, ...]

# Tính variance
variance_x = np.var(positions_x)

# Nếu variance thấp → gaze_x ổn định → Mắt đang "dừng"
# Nếu variance cao → gaze_x thay đổi nhiều → Mắt đang di chuyển
```

**Ví dụ:**
- `gaze_x = [0.50, 0.51, 0.49, 0.50]` → Variance thấp → Mắt dừng
- `gaze_x = [0.20, 0.50, 0.80, 0.30]` → Variance cao → Mắt di chuyển

---

## 💡 Lưu ý quan trọng

1. **gaze_x** thường được **normalize** để không phụ thuộc kích thước frame
2. **gaze_x = 0.0** không có nghĩa là "không nhìn", mà là "nhìn vào center"
3. **gaze_x** được tính từ **iris position** (MediaPipe) hoặc **face position** (OpenCV fallback)
4. **gaze_x** được lưu trong **sliding window** để tính stability
5. **gaze_x** và **gaze_y** cùng nhau tạo thành **gaze position** (2D)

---

## 🔧 Các biến liên quan trong code

- `offset_x` = gaze_x (normalized offset)
- `eye_offset_x` = gaze_x (từ eye landmarks)
- `positions_x` = List các gaze_x trong window
- `variance_x` = Variance của positions_x
- `gaze_positions_window` = List các `(gaze_x, gaze_y, ...)` tuples

