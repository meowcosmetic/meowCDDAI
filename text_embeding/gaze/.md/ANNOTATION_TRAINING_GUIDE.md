# Annotate một số frame và train lại YOLO (dùng cho gaze/object)

Mục tiêu: chọn **một số frame nhất định** trong video, **gán nhãn thủ công** (bbox + class), rồi **train/fine-tune** YOLOv8 với Ultralytics.

## 1) Chuẩn bị dataset folder

Tạo folder ví dụ:

```
dataset_myvideo/
  images/            # ảnh trích xuất từ video
  labels/            # YOLO labels (tool sẽ tạo)
  classes.txt        # danh sách class bạn muốn train
```

Tạo `classes.txt` (mỗi dòng 1 class), ví dụ:

```
book
pen
pencil
```

## 2) Trích xuất đúng frame bạn muốn annotate

### Lấy mỗi N frame

```bash
python tools/extract_frames_from_video.py --video C:\path\video.mp4 --out dataset_myvideo --every 30
```

### Lấy danh sách/range frame cụ thể

```bash
python tools/extract_frames_from_video.py --video C:\path\video.mp4 --out dataset_myvideo --frames 10,20,21,300,500-550
```

Kết quả ảnh nằm ở:

```
dataset_myvideo/images/
```

## 3) Annotate bbox và lưu YOLO label

```bash
python tools/annotate_yolo.py --images dataset_myvideo/images --labels dataset_myvideo/labels --classes dataset_myvideo/classes.txt
```

Phím tắt chính:
- **Chuột trái kéo-thả**: vẽ bbox
- **1..9,0**: chọn class nhanh
- **s**: lưu label
- **n/b**: ảnh tiếp/trước
- **d**: xoá bbox cuối
- **q/ESC**: thoát

## 4) Chuyển dataset sang chuẩn Ultralytics (train/val)

Ultralytics mong structure:

```
dataset_ultra/
  train/images
  train/labels
  val/images
  val/labels
  classes.txt
```

Nếu bạn muốn nhanh nhất: copy ảnh/label sang `train/` (và tự tạo `val/` sau), ví dụ:
- Copy toàn bộ `dataset_myvideo/images/*` → `dataset_ultra/train/images/`
- Copy toàn bộ `dataset_myvideo/labels/*` → `dataset_ultra/train/labels/`
- Copy `classes.txt` → `dataset_ultra/classes.txt`

Sau đó tạo `dataset.yaml`:

```bash
python tools/create_yolo_dataset_yaml.py --dataset dataset_ultra --out dataset_ultra/dataset.yaml
```

## 5) Train / Fine-tune YOLOv8

```bash
python tools/train_yolo_from_dataset.py --data dataset_ultra/dataset.yaml --weights yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --name myvideo-finetune
```

Output nằm ở: `runs/detect/`

---

## Bạn cần mình “gắn” model mới vào gaze pipeline luôn không?

Nếu bạn muốn hệ thống gaze dùng model mới để detect (ví dụ book/pen/pencil), mình có thể:
- thêm config để chọn weights mới
- load weights mới trong `text_embeding/gaze_tracking/oid_detector.py` hoặc tạo detector riêng cho dataset custom.

### ĐÃ HỖ TRỢ: dùng weights custom trong gaze

Bạn chỉ cần mở `text_embeding/gaze_tracking/config.py` và set:

- `CUSTOM_YOLO_WEIGHTS`: đường dẫn tới `best.pt` bạn train ra

Sau đó chạy gaze như bình thường (API hoặc test script). Nếu `CUSTOM_YOLO_WEIGHTS` hợp lệ, pipeline sẽ ưu tiên dùng model này để detect objects.


