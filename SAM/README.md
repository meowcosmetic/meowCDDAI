# SAM + CLIP Object Detection Test

Test script để detect objects bằng SAM (Segment Anything Model) + CLIP (Contrastive Language-Image Pre-training).

## Cách hoạt động

1. **Load ảnh mẫu** (reference image) của object cần tìm (ví dụ: cây bút)
2. **Tính CLIP embedding** của ảnh mẫu
3. **Segment ảnh target** với SAM để tìm tất cả các objects
4. **Tính CLIP embedding** của mỗi segment
5. **So sánh similarity** giữa reference embedding và segment embeddings
6. **Return các segments** có similarity cao (match với object mẫu)

## Ưu điểm

- ✅ Detect được objects ngay cả khi bị che khuất một phần (ví dụ: bút bị tay che 70%)
- ✅ Không cần dataset training
- ✅ Chỉ cần ảnh mẫu của object cần tìm
- ✅ Có thể detect bất kỳ object nào, không giới hạn classes

## Cài đặt

```bash
# Cài đặt dependencies
pip install ultralytics>=8.0.0
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python pillow torch torchvision
```

## Sử dụng

### 1. Sử dụng như một hàm

```python
from test_sam_clip import test_sam_clip_detection

# Detect objects
detections = test_sam_clip_detection(
    reference_image_path="pen_sample.jpg",  # Ảnh mẫu cây bút
    target_image_path="test_image.jpg",      # Ảnh cần detect
    output_path="result.jpg",                # Lưu kết quả
    similarity_threshold=0.25                 # Ngưỡng similarity
)

print(f"Found {len(detections)} matches")
for det in detections:
    print(f"Similarity: {det['similarity']:.3f}")
```

### 2. Sử dụng class SAMCLIPDetector

```python
from test_sam_clip import SAMCLIPDetector

# Khởi tạo detector
detector = SAMCLIPDetector(
    sam_model="sam_b.pt",           # SAM model (sam_b.pt, sam_l.pt, sam_x.pt)
    clip_model="ViT-B/32",          # CLIP model
    use_fastsam=False,              # Sử dụng FastSAM (nhanh hơn) hoặc SAM (chính xác hơn)
    similarity_threshold=0.25
)

# Register reference image (có thể register nhiều objects)
detector.register_reference_image("pen_sample.jpg", object_name="pen")
detector.register_reference_image("book_sample.jpg", object_name="book")

# Detect trong target image
detections = detector.detect_objects(
    target_image="test_image.jpg",
    reference_name="pen"  # Sử dụng reference đã register
)

# Visualize
detector.visualize_detections(
    image="test_image.jpg",
    detections=detections,
    output_path="result.jpg",
    show=True
)
```

### 3. Chạy từ command line

```bash
python test_sam_clip.py \
    --reference pen_sample.jpg \
    --target test_image.jpg \
    --output result.jpg \
    --sam-model sam_b.pt \
    --threshold 0.25
```

## Parameters

### SAM Models
- `sam_b.pt`: Base model (nhanh nhất, ít chính xác nhất)
- `sam_l.pt`: Large model (cân bằng)
- `sam_x.pt`: XLarge model (chậm nhất, chính xác nhất)

### FastSAM
- `FastSAM-x.pt`: FastSAM model (nhanh hơn SAM nhưng kém chính xác hơn)
- Sử dụng `--fastsam` flag hoặc `use_fastsam=True`

### CLIP Models
- `ViT-B/32`: Base model (nhanh nhất)
- `ViT-L/14`: Large model (chính xác hơn)
- `RN50`: ResNet-50 (cân bằng)

### Similarity Threshold
- `0.15-0.20`: Dễ dãi, detect nhiều objects (có thể có false positives)
- `0.25`: Cân bằng (mặc định)
- `0.30-0.35`: Nghiêm ngặt, chỉ detect objects rất giống

## Ví dụ Use Cases

### 1. Detect cây bút trong ảnh (kể cả khi bị che)

```python
detections = test_sam_clip_detection(
    reference_image_path="pen_sample.jpg",
    target_image_path="child_holding_pen.jpg",  # Bút có thể bị tay che
    output_path="pen_detected.jpg",
    similarity_threshold=0.25
)
```

### 2. Detect nhiều objects khác nhau

```python
detector = SAMCLIPDetector()

# Register nhiều reference images
detector.register_reference_image("pen.jpg", "pen")
detector.register_reference_image("book.jpg", "book")
detector.register_reference_image("cup.jpg", "cup")

# Detect từng loại
pen_detections = detector.detect_objects("test.jpg", reference_name="pen")
book_detections = detector.detect_objects("test.jpg", reference_name="book")
cup_detections = detector.detect_objects("test.jpg", reference_name="cup")
```

### 3. Xử lý video (detect tất cả objects)

```python
from test_sam_clip import process_video_sam_clip

# Xử lý video và detect tất cả objects
result = process_video_sam_clip(
    video_path="test_video.mp4",
    output_path="result_video.mp4",
    use_fastsam=True,  # Dùng FastSAM để nhanh hơn
    classify_objects=True,  # Phân loại và đặt tên objects
    frame_skip=5,  # Xử lý mỗi 5 frames (tăng tốc độ)
    show_video=True,  # Hiển thị video trong quá trình xử lý
    save_video=True  # Save video output
)

print(f"Processed {result['processed_frames']} frames")
print(f"Object counts: {result['object_counts']}")
```

### 4. Xử lý video với reference image (tìm object cụ thể)

```python
import cv2
from test_sam_clip import SAMCLIPDetector

detector = SAMCLIPDetector(use_fastsam=True)
detector.register_reference_image("pen_sample.jpg", "pen")

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects giống với reference
    detections = detector.detect_objects(frame, reference_name="pen")
    
    # Visualize
    annotated = detector.visualize_detections(frame, detections, show=False)
    cv2.imshow("Video", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Video Processing

### Xử lý video với command line:

```bash
# Xử lý video, detect và classify tất cả objects
python "test SAM/test_sam_clip.py" --target video.mp4 --all --video --output result.mp4

# Xử lý mỗi 10 frames để tăng tốc độ
python "test SAM/test_sam_clip.py" --target video.mp4 --all --video --frame-skip 10 --output result.mp4

# Không hiển thị video (chỉ save)
python "test SAM/test_sam_clip.py" --target video.mp4 --all --video --no-show --output result.mp4
```

### Parameters cho video:

- `--frame-skip`: Xử lý mỗi N frames (1 = tất cả, 5 = mỗi 5 frames)
  - Giá trị nhỏ hơn → chính xác hơn nhưng chậm hơn
  - Giá trị lớn hơn → nhanh hơn nhưng có thể miss objects
- `--no-show`: Không hiển thị video (tăng tốc độ)
- `--no-save`: Không save video output

## Notes

- **Performance**: SAM chậm hơn YOLO nhưng chính xác hơn và có thể detect objects bị che
- **Memory**: SAM models lớn, cần GPU để chạy nhanh
- **Video Processing**: Nên dùng `frame_skip` để tăng tốc độ (ví dụ: 5-10 frames)
- **Accuracy**: CLIP embedding matching có thể có false positives, cần điều chỉnh threshold
- **Use Cases**: Phù hợp khi cần detect một object cụ thể mà không có trong dataset training
- **Video**: Có thể xử lý video để detect và classify objects trong từng frame

## Troubleshooting

### Lỗi: "SAM model not found"
- Download SAM model: Ultralytics sẽ tự động download khi chạy lần đầu
- Hoặc download thủ công: `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/sam_b.pt`

### Lỗi: "CLIP not available"
- Cài đặt: `pip install git+https://github.com/openai/CLIP.git`
- Hoặc sử dụng Ultralytics CLIP (tự động nếu có ultralytics)

### Chạy chậm
- Sử dụng FastSAM thay vì SAM (`use_fastsam=True`)
- Sử dụng GPU (`device="cuda"`)
- Giảm image size trước khi process

