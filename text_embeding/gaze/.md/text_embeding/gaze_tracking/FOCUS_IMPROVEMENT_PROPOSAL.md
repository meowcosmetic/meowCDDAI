# Đề xuất cải thiện Focus Tracking

## 🔍 Vấn đề hiện tại

### Logic hiện tại (routes_screening_gaze.py, line 816-821):

```python
is_valid_focusing = is_stable and (
    (adult_face is not None and looking_at_adult_ratio > 0.5) or
    (looking_at_object_ratio > 0.5) or  # Nhìn vào objects
    (adult_face is None and looking_at_object_ratio < 0.3 and abs(offset_x) < 0.2 and abs(offset_y) < 0.2)  # ❌ VẤN ĐỀ Ở ĐÂY
)
```

**Vấn đề**: Điều kiện cuối cùng cho phép tính là "focusing" khi:
- Không có adult face
- Không nhìn vào objects (< 30%)
- Gaze stable và ở center (offset < 0.2)

→ **Kết quả**: Trẻ nhìn về camera nhưng không focus vào vật thể cụ thể vẫn được tính là "focusing"

## ✅ Giải pháp đề xuất

### Option 1: Smart Mode - Xử lý trường hợp adult ngồi kế camera (KHUYẾN NGHỊ)

**Vấn đề**: Nếu người lớn ngồi kế bên camera, trẻ nhìn về camera = có thể đang nhìn vào người lớn.

**Giải pháp**: Chỉ tính focus khi:
1. Có adult face trong frame VÀ gaze về phía camera → Coi là nhìn vào adult
2. Có tracked objects VÀ gaze vào objects → Coi là nhìn vào objects
3. Không có gì → KHÔNG tính focus

```python
# Smart logic: Xử lý trường hợp adult ngồi kế camera
is_valid_focusing = is_stable and (
    # Case 1: Có adult face trong frame
    (adult_face is not None and (
        looking_at_adult_ratio > 0.5 or  # Nhìn trực tiếp vào adult
        (looking_at_object_ratio < 0.3 and abs(offset_x) < 0.2 and abs(offset_y) < 0.2)  # Nhìn về camera (có thể đang nhìn adult kế camera)
    )) or
    # Case 2: Nhìn vào tracked objects
    (looking_at_object_ratio > 0.5)  # Nhìn vào objects (book, cup, etc.)
)
# ❌ KHÔNG tính focus nếu: không có adult face VÀ không nhìn vào objects
```

**Lợi ích**:
- ✅ Xử lý trường hợp adult ngồi kế camera
- ✅ Chỉ tính focus khi có object/adult cụ thể
- ✅ Phân biệt rõ "nhìn về camera (không có gì)" vs "nhìn về camera (có adult kế đó)"

**Nhược điểm**:
- Cần detect adult face tốt
- Có thể cần điều chỉnh threshold

### Option 1B: Strict Mode - Chỉ tính focus khi có object cụ thể

```python
# CHỈ tính focus khi thực sự nhìn vào tracked object
is_valid_focusing = is_stable and (
    (adult_face is not None and looking_at_adult_ratio > 0.5) or  # Nhìn vào người lớn
    (looking_at_object_ratio > 0.5)  # Nhìn vào objects (book, cup, person, etc.)
)
# ❌ LOẠI BỎ điều kiện "nhìn vào camera" khi không có object
```

**Lợi ích**:
- ✅ Chỉ tính focus khi có object cụ thể
- ✅ Phân biệt rõ "nhìn về camera" vs "focus vào object"
- ✅ Kết quả chính xác hơn

**Nhược điểm**:
- Có thể giảm eye_contact_percentage nếu video không có objects
- Không xử lý trường hợp adult ngồi kế camera

### Option 2: Hybrid Mode - Có flag để chọn strict/lenient

```python
# Thêm config flag
REQUIRE_OBJECT_FOCUS = True  # True = strict, False = lenient (giữ logic cũ)

if REQUIRE_OBJECT_FOCUS:
    # Strict: chỉ tính khi có object
    is_valid_focusing = is_stable and (
        (adult_face is not None and looking_at_adult_ratio > 0.5) or
        (looking_at_object_ratio > 0.5)
    )
else:
    # Lenient: giữ logic cũ (cho backward compatibility)
    is_valid_focusing = is_stable and (
        (adult_face is not None and looking_at_adult_ratio > 0.5) or
        (looking_at_object_ratio > 0.5) or
        (adult_face is None and looking_at_object_ratio < 0.3 and abs(offset_x) < 0.2 and abs(offset_y) < 0.2)
    )
```

### Option 3: Confidence-based - Dùng confidence score từ 3D gaze

```python
# Sử dụng 3D gaze estimation với confidence threshold
if gaze_3d_result:
    object_id, confidence = gaze_3d_result
    
    # Chỉ tính focus nếu confidence đủ cao
    if confidence > 0.5:  # Threshold
        is_valid_focusing = is_stable
    else:
        is_valid_focusing = False
else:
    # Fallback: chỉ tính khi có object (strict mode)
    is_valid_focusing = is_stable and (
        (adult_face is not None and looking_at_adult_ratio > 0.5) or
        (looking_at_object_ratio > 0.5)
    )
```

## 📊 So sánh các options

| Option | Độ chính xác | Xử lý adult kế camera | Backward Compatible | Phức tạp | Khuyến nghị |
|--------|--------------|----------------------|---------------------|----------|-------------|
| Option 1 (Smart) | ⭐⭐⭐⭐⭐ | ✅ | ⚠️ | ⭐⭐ | ✅ **KHUYẾN NGHỊ** |
| Option 1B (Strict) | ⭐⭐⭐⭐ | ❌ | ❌ | ⭐ | ⚠️ Không xử lý adult kế camera |
| Option 2 (Hybrid) | ⭐⭐⭐⭐ | ⚠️ | ✅ | ⭐⭐ | ✅ Tốt |
| Option 3 (Confidence) | ⭐⭐⭐⭐⭐ | ✅ | ⚠️ | ⭐⭐⭐ | ✅ Tốt nhất (cần 3D gaze) |

## 🎯 Đề xuất Implementation

### Bước 1: Cập nhật Config

```python
# text_embeding/gaze_tracking/config.py
@dataclass
class GazeConfig:
    # ... existing configs ...
    
    # Focus detection mode
    REQUIRE_OBJECT_FOCUS: bool = True  # True = chỉ tính focus khi có object cụ thể
    MIN_OBJECT_FOCUS_RATIO: float = 0.5  # Tỷ lệ tối thiểu để coi là focus vào object
    USE_3D_GAZE_CONFIDENCE: bool = True  # Dùng confidence từ 3D gaze nếu có
    MIN_3D_GAZE_CONFIDENCE: float = 0.5  # Confidence threshold cho 3D gaze
```

### Bước 2: Cập nhật Logic trong routes_screening_gaze.py

**Vị trí 1: Fallback mode (OpenCV) - line ~817**
```python
# TRƯỚC:
is_valid_focusing = is_stable and (
    (adult_face is not None and looking_at_adult_ratio > 0.5) or
    (looking_at_object_ratio > 0.5) or
    (adult_face is None and looking_at_object_ratio < 0.3 and abs(offset_x) < 0.2 and abs(offset_y) < 0.2)  # ❌
)

# SAU (Option 1 - Smart Mode):
is_valid_focusing = is_stable and (
    # Case 1: Có adult face trong frame
    (adult_face is not None and (
        looking_at_adult_ratio > config.MIN_OBJECT_FOCUS_RATIO or  # Nhìn trực tiếp vào adult
        (looking_at_object_ratio < 0.3 and abs(offset_x) < 0.2 and abs(offset_y) < 0.2)  # Nhìn về camera (có thể đang nhìn adult kế camera)
    )) or
    # Case 2: Nhìn vào tracked objects
    (looking_at_object_ratio > config.MIN_OBJECT_FOCUS_RATIO)  # Nhìn vào objects
)
# ✅ Xử lý trường hợp adult ngồi kế camera
# ❌ KHÔNG tính focus nếu: không có adult face VÀ không nhìn vào objects
```

**Vị trí 2: MediaPipe mode - line ~1163**
```python
# Tương tự
is_valid_focusing = is_stable and (
    # Case 1: Có adult face trong frame
    (adult_face_info and (
        looking_at_adult_ratio > config.MIN_OBJECT_FOCUS_RATIO or
        (looking_at_object_ratio < 0.3 and abs(eye_offset_x) < 0.2 and abs(eye_offset_y) < 0.2)
    )) or
    # Case 2: Nhìn vào tracked objects
    (looking_at_object_ratio > config.MIN_OBJECT_FOCUS_RATIO)
)
```

### Bước 3: Cập nhật FocusTimeline (đã tốt, chỉ cần verify)

FocusTimeline đã đúng - chỉ tính focus khi `looking_at_object` không None.

### Bước 4: Thêm logging để debug

```python
if is_stable and not is_valid_focusing:
    logger.debug(f"[Gaze] Gaze stable nhưng không focus: "
                 f"adult_ratio={looking_at_adult_ratio:.2f}, "
                 f"object_ratio={looking_at_object_ratio:.2f}, "
                 f"offset=({offset_x:.2f}, {offset_y:.2f})")
```

## 🔄 Migration Path

1. **Phase 1**: Implement Option 1 (Strict mode) - Loại bỏ điều kiện "nhìn vào camera"
2. **Phase 2**: Thêm config flag `REQUIRE_OBJECT_FOCUS` để có thể toggle
3. **Phase 3**: Tích hợp 3D gaze confidence (nếu cần)

## 📝 Expected Behavior After Fix

### Trước (Current):
- Trẻ nhìn về camera, gaze stable → ✅ Tính là focusing (dù không có object)
- Trẻ nhìn về camera, không có objects → ✅ Tính là focusing
- Trẻ nhìn về camera, có adult kế camera → ✅ Tính là focusing (nhưng không phân biệt được)

### Sau (Fixed - Option 1 Smart Mode):
- Trẻ nhìn về camera, KHÔNG có adult face → ❌ KHÔNG tính là focusing
- Trẻ nhìn về camera, CÓ adult face trong frame → ✅ Tính là focusing (coi là nhìn vào adult kế camera)
- Trẻ nhìn vào book_1, gaze stable → ✅ Tính là focusing
- Trẻ nhìn vào adult (trực tiếp), gaze stable → ✅ Tính là focusing
- Trẻ nhìn về camera, CÓ objects nhưng không nhìn vào → ❌ KHÔNG tính là focusing (nếu không có adult)

### Sau (Fixed - Option 1B Strict Mode):
- Trẻ nhìn về camera, gaze stable → ❌ KHÔNG tính là focusing (vì không có object)
- Trẻ nhìn vào book_1, gaze stable → ✅ Tính là focusing
- Trẻ nhìn vào adult, gaze stable → ✅ Tính là focusing
- Trẻ nhìn về camera, có adult kế camera → ❌ KHÔNG tính (nhược điểm)

## ⚠️ Breaking Changes

- `eye_contact_percentage` có thể giảm nếu video không có objects
- Cần đảm bảo object detection hoạt động tốt
- Có thể cần điều chỉnh thresholds

## 🧪 Testing

Test cases cần verify:
1. Video có objects → focus được detect đúng
2. Video không có objects → KHÔNG tính focus (dù gaze stable)
3. Video có adult face → focus vào adult được detect
4. Video có books → focus vào books được detect

