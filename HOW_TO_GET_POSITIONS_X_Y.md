# Cách lấy positions_x và positions_y từ code

## 📋 Tóm tắt

`positions_x` và `positions_y` được **trích xuất** từ `gaze_positions_window` bằng **list comprehension**.

---

## 🔍 Cấu trúc dữ liệu: gaze_positions_window

### Định nghĩa:
```python
gaze_positions_window = []  # List các tuples
```

### Mỗi phần tử trong window là một tuple:
```python
(eye_offset_x, eye_offset_y, is_looking_at_adult, is_looking_at_object)
```

**Ví dụ:**
```python
gaze_positions_window = [
    (0.50, 0.51, True, False),   # Frame 1: gaze_x=0.50, gaze_y=0.51, nhìn adult, không nhìn object
    (0.49, 0.52, True, False),   # Frame 2: gaze_x=0.49, gaze_y=0.52, nhìn adult, không nhìn object
    (0.51, 0.50, True, True),    # Frame 3: gaze_x=0.51, gaze_y=0.50, nhìn adult, nhìn object
    # ... nhiều frames khác
]
```

---

## 💻 Cách trích xuất positions_x và positions_y

### Code trong routes_screening_gaze.py:

```python
# Kiểm tra độ ổn định của gaze (focusing detection)
if len(gaze_positions_window) >= FOCUSING_WINDOW_SIZE:
    # ✅ TRÍCH XUẤT positions_x và positions_y
    positions_x = [pos[0] for pos in gaze_positions_window]
    positions_y = [pos[1] for pos in gaze_positions_window]
    
    # Tính variance
    variance_x = np.var(positions_x) if len(positions_x) > 1 else 0
    variance_y = np.var(positions_y) if len(positions_y) > 1 else 0
    total_variance = variance_x + variance_y
```

---

## 🔧 Giải thích chi tiết

### 1. List Comprehension

```python
positions_x = [pos[0] for pos in gaze_positions_window]
```

**Giải thích từng phần:**
- `for pos in gaze_positions_window` → Duyệt qua từng tuple trong window
- `pos[0]` → Lấy phần tử đầu tiên (gaze_x) của mỗi tuple
- `[...]` → Tạo list mới chứa tất cả các giá trị gaze_x

**Tương tự:**
```python
positions_y = [pos[1] for pos in gaze_positions_window]
```
- `pos[1]` → Lấy phần tử thứ hai (gaze_y) của mỗi tuple

---

## 📊 Ví dụ cụ thể

### Input: gaze_positions_window
```python
gaze_positions_window = [
    (0.50, 0.51, True, False),   # pos[0]=0.50, pos[1]=0.51
    (0.49, 0.52, True, False),   # pos[0]=0.49, pos[1]=0.52
    (0.51, 0.50, True, True),    # pos[0]=0.51, pos[1]=0.50
    (0.50, 0.51, False, True),   # pos[0]=0.50, pos[1]=0.51
    (0.49, 0.52, False, False),  # pos[0]=0.49, pos[1]=0.52
]
```

### Output: positions_x
```python
positions_x = [pos[0] for pos in gaze_positions_window]
# Kết quả:
positions_x = [0.50, 0.49, 0.51, 0.50, 0.49]
```

### Output: positions_y
```python
positions_y = [pos[1] for pos in gaze_positions_window]
# Kết quả:
positions_y = [0.51, 0.52, 0.50, 0.51, 0.52]
```

---

## 🎯 Cách hoạt động từng bước

### Bước 1: Tạo gaze_positions_window
```python
# Khởi tạo window rỗng
gaze_positions_window = []

# Mỗi frame, thêm gaze position vào window
for frame in video:
    # Tính gaze position
    eye_offset_x = calculate_gaze_x(...)  # Ví dụ: 0.50
    eye_offset_y = calculate_gaze_y(...)  # Ví dụ: 0.51
    
    # Thêm vào window
    gaze_positions_window.append((eye_offset_x, eye_offset_y, is_looking_at_adult, is_looking_at_object))
    
    # Giữ window size cố định (sliding window)
    if len(gaze_positions_window) > FOCUSING_WINDOW_SIZE:
        gaze_positions_window.pop(0)  # Xóa phần tử cũ nhất
```

### Bước 2: Trích xuất positions_x và positions_y
```python
# Khi window đủ lớn
if len(gaze_positions_window) >= FOCUSING_WINDOW_SIZE:
    # Trích xuất tất cả gaze_x
    positions_x = [pos[0] for pos in gaze_positions_window]
    # → [0.50, 0.49, 0.51, 0.50, 0.49, ...]
    
    # Trích xuất tất cả gaze_y
    positions_y = [pos[1] for pos in gaze_positions_window]
    # → [0.51, 0.52, 0.50, 0.51, 0.52, ...]
```

### Bước 3: Tính variance
```python
# Tính variance để kiểm tra stability
variance_x = np.var(positions_x)  # Ví dụ: 0.0008
variance_y = np.var(positions_y)   # Ví dụ: 0.0012
total_variance = variance_x + variance_y  # 0.002

# Kiểm tra "điểm dừng"
is_stable = total_variance < GAZE_STABILITY_THRESHOLD  # 0.05
```

---

## 🔄 Luồng dữ liệu hoàn chỉnh

```
Video Frame
    ↓
Tính gaze position (eye_offset_x, eye_offset_y)
    ↓
Thêm vào gaze_positions_window
    ↓
[Tuple 1: (0.50, 0.51, True, False)]
[Tuple 2: (0.49, 0.52, True, False)]
[Tuple 3: (0.51, 0.50, True, True)]
...
    ↓
Trích xuất bằng list comprehension
    ↓
positions_x = [0.50, 0.49, 0.51, ...]
positions_y = [0.51, 0.52, 0.50, ...]
    ↓
Tính variance
    ↓
Kiểm tra stability
```

---

## 💡 Các cách khác để trích xuất (tương đương)

### Cách 1: List Comprehension (đang dùng)
```python
positions_x = [pos[0] for pos in gaze_positions_window]
positions_y = [pos[1] for pos in gaze_positions_window]
```

### Cách 2: Vòng lặp for
```python
positions_x = []
positions_y = []
for pos in gaze_positions_window:
    positions_x.append(pos[0])
    positions_y.append(pos[1])
```

### Cách 3: Unpacking
```python
positions_x, positions_y = zip(*[(pos[0], pos[1]) for pos in gaze_positions_window])
```

### Cách 4: NumPy (nếu cần array)
```python
import numpy as np
positions_array = np.array(gaze_positions_window)
positions_x = positions_array[:, 0]  # Cột đầu tiên
positions_y = positions_array[:, 1]  # Cột thứ hai
```

---

## 📝 Tóm tắt

| Bước | Mô tả | Code |
|------|-------|------|
| **1. Tạo window** | Lưu gaze positions mỗi frame | `gaze_positions_window.append((x, y, ...))` |
| **2. Trích xuất X** | Lấy tất cả gaze_x | `positions_x = [pos[0] for pos in window]` |
| **3. Trích xuất Y** | Lấy tất cả gaze_y | `positions_y = [pos[1] for pos in window]` |
| **4. Tính variance** | Tính độ phân tán | `variance_x = np.var(positions_x)` |
| **5. Kiểm tra** | So sánh với threshold | `is_stable = variance < threshold` |

---

## 🎯 Vị trí trong code

**File:** `text_embeding/routes_screening_gaze.py`

**Dòng ~774 (OpenCV fallback):**
```python
positions_x = [pos[0] for pos in gaze_positions_window]
positions_y = [pos[1] for pos in gaze_positions_window]
```

**Dòng ~1234 (MediaPipe):**
```python
positions_x = [pos[0] for pos in gaze_positions_window]
positions_y = [pos[1] for pos in gaze_positions_window]
```

---

## ✅ Kết luận

Để có `positions_x` và `positions_y`:
1. Cần có `gaze_positions_window` (list các tuples)
2. Dùng **list comprehension** để trích xuất:
   - `positions_x = [pos[0] for pos in gaze_positions_window]`
   - `positions_y = [pos[1] for pos in gaze_positions_window]`
3. Sau đó tính variance để kiểm tra stability




