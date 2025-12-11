# Giải thích: Variance là gì?

## 📚 Định nghĩa đơn giản

**Variance (Phương sai)** là một số đo **độ phân tán** của dữ liệu.

- **Variance thấp** → Dữ liệu **gần nhau** (ít thay đổi) → Mắt **ổn định** (đang dừng)
- **Variance cao** → Dữ liệu **xa nhau** (thay đổi nhiều) → Mắt **không ổn định** (đang di chuyển)

---

## 🎯 Ví dụ cụ thể: Vị trí mắt nhìn

### Trường hợp 1: Mắt đang "dừng" (variance thấp)

Giả sử trong 5 frames, vị trí X của mắt là:
```
Frame 1: gaze_x = 0.50
Frame 2: gaze_x = 0.51
Frame 3: gaze_x = 0.49
Frame 4: gaze_x = 0.50
Frame 5: gaze_x = 0.51
```

**Nhận xét:**
- Tất cả giá trị đều **gần nhau** (0.49 - 0.51)
- Mắt **hầu như không di chuyển**
- **Variance thấp** → Mắt đang **"dừng"**

### Trường hợp 2: Mắt đang di chuyển (variance cao)

Giả sử trong 5 frames, vị trí X của mắt là:
```
Frame 1: gaze_x = 0.20
Frame 2: gaze_x = 0.50
Frame 3: gaze_x = 0.80
Frame 4: gaze_x = 0.30
Frame 5: gaze_x = 0.70
```

**Nhận xét:**
- Giá trị **thay đổi nhiều** (0.20 - 0.80)
- Mắt **đang di chuyển** qua lại
- **Variance cao** → Mắt **không ổn định**

---

## 📐 Công thức tính Variance

### Bước 1: Tính giá trị trung bình (mean)
```python
mean = (x1 + x2 + x3 + ... + xn) / n
```

### Bước 2: Tính variance
```python
variance = [(x1 - mean)² + (x2 - mean)² + ... + (xn - mean)²] / n
```

### Ví dụ tính toán:

**Trường hợp 1: Mắt dừng**
```
Vị trí X: [0.50, 0.51, 0.49, 0.50, 0.51]

Bước 1: mean = (0.50 + 0.51 + 0.49 + 0.50 + 0.51) / 5 = 0.502

Bước 2: variance = [
    (0.50 - 0.502)² + 
    (0.51 - 0.502)² + 
    (0.49 - 0.502)² + 
    (0.50 - 0.502)² + 
    (0.51 - 0.502)²
] / 5

variance = [0.000004 + 0.000064 + 0.000144 + 0.000004 + 0.000064] / 5
variance = 0.000056  # Rất thấp! → Mắt đang dừng
```

**Trường hợp 2: Mắt di chuyển**
```
Vị trí X: [0.20, 0.50, 0.80, 0.30, 0.70]

Bước 1: mean = (0.20 + 0.50 + 0.80 + 0.30 + 0.70) / 5 = 0.50

Bước 2: variance = [
    (0.20 - 0.50)² + 
    (0.50 - 0.50)² + 
    (0.80 - 0.50)² + 
    (0.30 - 0.50)² + 
    (0.70 - 0.50)²
] / 5

variance = [0.09 + 0.00 + 0.09 + 0.04 + 0.04] / 5
variance = 0.052  # Cao hơn nhiều! → Mắt đang di chuyển
```

---

## 💻 Cách tính trong code

### Trong Python (numpy):
```python
import numpy as np

# Vị trí gaze theo trục X
positions_x = [0.50, 0.51, 0.49, 0.50, 0.51]

# Tính variance
variance_x = np.var(positions_x)
print(variance_x)  # 0.000056 (rất thấp)
```

### Trong code gaze tracking:
```python
# Lấy các vị trí gaze trong window (ví dụ: 30 frames gần nhất)
positions_x = [pos[0] for pos in gaze_positions_window]  # Vị trí X
positions_y = [pos[1] for pos in gaze_positions_window]  # Vị trí Y

# Tính variance
variance_x = np.var(positions_x)  # Phương sai theo trục X
variance_y = np.var(positions_y)  # Phương sai theo trục Y
total_variance = variance_x + variance_y  # Tổng phương sai

# Kiểm tra "điểm dừng"
is_stable = total_variance < GAZE_STABILITY_THRESHOLD  # 0.05
```

---

## 🎨 Minh họa bằng hình ảnh

### Variance thấp (mắt dừng):
```
Vị trí gaze:
  |
  |     ●
  |   ● ● ●
  |     ●
  |________________
  0.0  0.5  1.0

→ Tất cả điểm gần nhau → Variance thấp → Mắt đang dừng
```

### Variance cao (mắt di chuyển):
```
Vị trí gaze:
  |
  |●              ●
  |      ●
  |            ●
  |  ●
  |________________
  0.0  0.5  1.0

→ Các điểm xa nhau → Variance cao → Mắt đang di chuyển
```

---

## 🔢 Đơn vị và phạm vi

### Trong gaze tracking:
- **Đơn vị**: Normalized (0-1)
  - 0.0 = Hoàn toàn ổn định (mắt hoàn toàn dừng)
  - 1.0 = Rất không ổn định (mắt di chuyển nhiều)

### So sánh với threshold:
```python
GAZE_STABILITY_THRESHOLD = 0.05

if total_variance < 0.05:
    # Mắt đang "dừng" (ổn định)
    is_stable = True
else:
    # Mắt đang di chuyển (không ổn định)
    is_stable = False
```

---

## 📊 Ví dụ thực tế

### Scenario 1: Trẻ đang nhìn chăm chú vào sách
```
Frame 1-30: gaze_x = [0.48, 0.49, 0.47, 0.48, 0.49, ...]
            gaze_y = [0.52, 0.51, 0.53, 0.52, 0.51, ...]

variance_x = 0.0008  (rất thấp)
variance_y = 0.0012  (rất thấp)
total_variance = 0.002  < 0.05

→ is_stable = True → Mắt đang "dừng" → Đang focus vào sách ✅
```

### Scenario 2: Trẻ đang nhìn xung quanh
```
Frame 1-30: gaze_x = [0.20, 0.50, 0.80, 0.30, 0.70, ...]
            gaze_y = [0.40, 0.60, 0.20, 0.80, 0.50, ...]

variance_x = 0.045  (cao)
variance_y = 0.038  (cao)
total_variance = 0.083  > 0.05

→ is_stable = False → Mắt đang di chuyển → Không focus ❌
```

---

## 🎯 Tóm tắt

| Khái niệm | Ý nghĩa | Ví dụ |
|-----------|---------|-------|
| **Variance** | Độ phân tán của dữ liệu | 0.0001 = rất ổn định, 0.1 = không ổn định |
| **Variance thấp** | Dữ liệu gần nhau | Mắt đang "dừng" (ổn định) |
| **Variance cao** | Dữ liệu xa nhau | Mắt đang di chuyển (không ổn định) |
| **GAZE_STABILITY_THRESHOLD** | Ngưỡng để phân biệt | 0.05 (nghiêm ngặt) |

---

## 💡 Lưu ý quan trọng

1. **Variance = 0** → Hoàn toàn không thay đổi (rất hiếm trong thực tế)
2. **Variance < 0.05** → Rất ổn định (mắt đang dừng)
3. **Variance > 0.1** → Không ổn định (mắt đang di chuyển)
4. **Variance được tính trên cả 2 trục** (X và Y) → `total_variance = variance_x + variance_y`

---

## 🔧 Điều chỉnh threshold

### Nếu muốn dễ dàng hơn (chấp nhận dao động nhỏ):
```python
GAZE_STABILITY_THRESHOLD = 0.1  # Tăng từ 0.05 lên 0.1
```

### Nếu muốn nghiêm ngặt hơn (chỉ tính khi rất ổn định):
```python
GAZE_STABILITY_THRESHOLD = 0.02  # Giảm từ 0.05 xuống 0.02
```




