# Open Images Dataset V7 (OID) Detector

## Tổng quan

OID Detector sử dụng YOLOv8 từ Ultralytics, được train trên Open Images Dataset V7 với **600 classes** (so với COCO chỉ có 80 classes).

## ✅ Ưu điểm

1. **Có "pen" và "pencil"** - COCO không có!
2. **600 classes** - nhiều hơn COCO rất nhiều
3. **YOLOv8** - nhanh hơn và chính xác hơn YOLOv3/v4
4. **Tự động download** - weights được download tự động lần đầu

## Cài đặt

```bash
# Cài ultralytics
pip install ultralytics>=8.0.0

# Hoặc chạy script
install_oid_detector.bat
```

## Cấu hình

Trong `gaze_tracking/config.py`:

```python
USE_OID_DATASET: bool = True  # True = dùng OID, False = dùng COCO
OID_MODEL_SIZE: str = 'n'  # 'n' (nano), 's', 'm', 'l', 'x'
```

### Model Sizes

- **'n' (nano)**: Nhanh nhất, ít chính xác nhất (~6MB)
- **'s' (small)**: Cân bằng (~22MB)
- **'m' (medium)**: Chính xác hơn (~52MB)
- **'l' (large)**: Rất chính xác (~87MB)
- **'x' (xlarge)**: Chính xác nhất (~136MB)

## Sử dụng

OID detector sẽ tự động được sử dụng nếu:
1. `USE_OID_DATASET = True` trong config
2. `ultralytics` đã được cài đặt
3. Model weights được download thành công

## Classes có sẵn

OID có nhiều classes hơn COCO, bao gồm:
- ✅ **Pen, Pencil, Marker, Crayon** (COCO không có!)
- Person, Man, Woman, Boy, Girl, Child, Baby
- Book, Cup, Bottle, Glass
- Cell phone, Laptop, Computer, Tablet
- Mouse, Keyboard, Remote control
- Toy, Doll, Teddy bear, Ball
- Và nhiều classes khác...

## Fallback

Nếu OID không available, hệ thống sẽ tự động fallback về COCO YOLO.

## Logs

Khi OID được sử dụng, bạn sẽ thấy:
```
[ObjectDetector] ✅ Sử dụng OID detector (có pen/pencil)
```

Khi fallback về COCO:
```
[ObjectDetector] OID detector không available, fallback về COCO
```




