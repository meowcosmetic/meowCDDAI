# 📥 Hướng dẫn download YOLO weights cho Object Detection

## Tại sao cần YOLO?

YOLO (You Only Look Once) là model object detection để phát hiện:
- Người (person)
- Đồ vật: sách (book), bút, đồ chơi, etc.

## Cách download

### Option 1: YOLOv3 Tiny (Khuyến nghị - nhẹ, nhanh)

```bash
# Download weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights

# Download config
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
```

Hoặc download thủ công:
- Weights: https://pjreddie.com/media/files/yolov3-tiny.weights
- Config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg

### Option 2: YOLOv3 (Chính xác hơn, nhưng chậm hơn)

```bash
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

### Option 3: YOLOv4 Tiny

```bash
wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
```

## Đặt files

Đặt cả 2 files (`.weights` và `.cfg`) vào thư mục project root:
```
meowCDDAI/
  ├── yolov3-tiny.weights
  ├── yolov3-tiny.cfg
  ├── main.py
  └── ...
```

## Kiểm tra

Sau khi download, chạy lại API. Bạn sẽ thấy log:
```
[Gaze] ✅ Đã load YOLO model: yolov3-tiny.cfg
[Gaze] Object detection enabled với YOLO
```

## Lưu ý

- YOLOv3-tiny.weights: ~33 MB
- YOLOv3.weights: ~248 MB
- YOLOv4-tiny.weights: ~23 MB

Khuyến nghị dùng YOLOv3-tiny cho tốc độ tốt và đủ chính xác.

## Không có YOLO?

Nếu không có YOLO weights, API vẫn chạy được nhưng:
- Object detection sẽ bị tắt
- Chỉ detect được faces (người lớn/trẻ)
- Không detect được đồ vật (sách, bút, etc.)

