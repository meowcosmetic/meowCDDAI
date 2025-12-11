"""
Script hướng dẫn train YOLOv8 OID model từ Open Images Dataset V7
"""
import sys
import os

print("=" * 70)
print("HƯỚNG DẪN TRAIN YOLOv8 OID MODEL")
print("=" * 70)
print()

print("⚠️  Ultralytics không có OID model sẵn!")
print("   Để sử dụng OID detector, bạn cần train model từ Open Images Dataset V7")
print()

print("=" * 70)
print("CÁCH 1: Train từ đầu với Open Images Dataset V7")
print("=" * 70)
print()
print("1. Download Open Images Dataset V7:")
print("   - Truy cập: https://storage.googleapis.com/openimages/web/index.html")
print("   - Download images và annotations")
print()
print("2. Chuẩn bị dataset theo format YOLO:")
print("   - Tổ chức dataset theo cấu trúc:")
print("     dataset/")
print("       train/")
print("         images/")
print("         labels/")
print("       val/")
print("         images/")
print("         labels/")
print()
print("3. Tạo file dataset.yaml:")
print("""
path: ./dataset
train: train/images
val: val/images

names:
  0: Person
  1: Book
  2: Pen
  3: Pencil
  # ... và các classes khác từ OID
""")
print()
print("4. Train model:")
print("""
from ultralytics import YOLO

# Load pretrained COCO model làm starting point
model = YOLO('yolov8n.pt')

# Train trên OID dataset
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='yolov8n-oidv7'
)

# Export model
model.export(format='onnx')
""")
print()

print("=" * 70)
print("CÁCH 2: Download model đã train sẵn từ cộng đồng")
print("=" * 70)
print()
print("Tìm kiếm trên:")
print("  - Hugging Face: https://huggingface.co/models")
print("  - GitHub: Tìm 'yolov8 oid' hoặc 'yolov8 open images'")
print("  - Roboflow: https://roboflow.com/models")
print()
print("Sau khi có model, đặt vào:")
print(f"  {os.path.expanduser('~/.ultralytics/weights/yolov8n-oidv7.pt')}")
print()

print("=" * 70)
print("CÁCH 3: Sử dụng model từ Roboflow (nếu có)")
print("=" * 70)
print()
print("Roboflow có thể có OID models:")
print("  - Truy cập: https://roboflow.com/models")
print("  - Tìm 'Open Images Dataset' hoặc 'OID'")
print("  - Download và convert sang YOLO format")
print()

print("=" * 70)
print("LƯU Ý")
print("=" * 70)
print()
print("⚠️  Training OID model từ đầu cần:")
print("  - Dataset lớn (~9TB cho full OID)")
print("  - GPU mạnh (recommended)")
print("  - Thời gian train lâu (nhiều ngày)")
print()
print("💡 Khuyến nghị:")
print("  - Sử dụng subset của OID (chỉ các classes cần thiết)")
print("  - Hoặc tìm model đã train sẵn từ cộng đồng")
print("  - Hoặc fine-tune từ COCO model trên subset OID")
print()

print("=" * 70)
print("THAM KHẢO")
print("=" * 70)
print()
print("  - Ultralytics Docs: https://docs.ultralytics.com/")
print("  - Open Images Dataset: https://storage.googleapis.com/openimages/web/index.html")
print("  - YOLOv8 Training Guide: https://docs.ultralytics.com/modes/train/")
print()

