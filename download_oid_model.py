"""
Script để download YOLOv8 OID model
Chạy script này để download model trước khi sử dụng
"""
import sys
import os

print("=" * 60)
print("Download YOLOv8 OID Model")
print("=" * 60)
print()

# Check ultralytics
try:
    from ultralytics import YOLO
    print("✅ Ultralytics đã được cài đặt")
except ImportError:
    print("❌ Ultralytics chưa được cài đặt!")
    print()
    print("Vui lòng cài đặt ultralytics trước:")
    print("  pip install ultralytics>=8.0.0")
    print()
    print("Hoặc chạy: install_oid_detector.bat")
    sys.exit(1)

# Model size từ config hoặc default
model_size = 'm'  # Default: medium
if len(sys.argv) > 1:
    model_size = sys.argv[1]

model_name = f"yolov8{model_size}-oidv7.pt"

print(f"📦 Model: {model_name}")
print(f"📊 Size: {model_size.upper()}")
print()
print("Đang download model...")
print("(Lần đầu tiên có thể mất vài phút, tùy vào tốc độ internet)")
print()

try:
    # Load model (sẽ tự động download nếu chưa có)
    model = YOLO(model_name)
    
    print("=" * 60)
    print("✅ Model đã được download và load thành công!")
    print("=" * 60)
    print()
    print(f"Model location: {os.path.expanduser('~/.ultralytics/weights/')}")
    print(f"Model file: {model_name}")
    print()
    print("Bây giờ bạn có thể sử dụng OID detector!")
    print()
    
except Exception as e:
    print("=" * 60)
    print("❌ Lỗi khi download model!")
    print("=" * 60)
    print()
    print(f"Lỗi: {str(e)}")
    print()
    print("Có thể do:")
    print("  - Không có kết nối internet")
    print("  - Firewall chặn download")
    print("  - Ultralytics chưa được cài đặt đúng")
    print()
    print("Vui lòng thử lại hoặc kiểm tra kết nối internet.")
    sys.exit(1)



