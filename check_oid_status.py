"""
Script để kiểm tra trạng thái OID detector
"""
import sys

print("=" * 60)
print("KIỂM TRA OID DETECTOR STATUS")
print("=" * 60)
print()

# 1. Kiểm tra ultralytics
print("1. Kiểm tra ultralytics:")
try:
    from ultralytics import YOLO
    print("   ✅ ultralytics đã được cài đặt")
    try:
        version = YOLO.__version__ if hasattr(YOLO, '__version__') else "unknown"
        print(f"   Version: {version}")
    except:
        pass
except ImportError as e:
    print("   ❌ ultralytics CHƯA được cài đặt")
    print(f"   Error: {e}")
    print("   👉 Cần cài: pip install ultralytics>=8.0.0")
    print()
    sys.exit(1)

print()

# 2. Kiểm tra OID detector module
print("2. Kiểm tra OID detector module:")
try:
    from text_embeding.gaze_tracking.oid_detector import ULTRALYTICS_AVAILABLE, create_oid_detector
    if ULTRALYTICS_AVAILABLE:
        print("   ✅ OID detector module available")
    else:
        print("   ❌ OID detector module không available (ultralytics chưa cài)")
        sys.exit(1)
except ImportError as e:
    print(f"   ❌ Không thể import OID detector: {e}")
    sys.exit(1)

print()

# 3. Kiểm tra ObjectDetector
print("3. Kiểm tra ObjectDetector:")
try:
    from text_embeding.gaze_tracking.object_detector import ObjectDetector
    from text_embeding.gaze_tracking.config import GazeConfig
    from text_embeding.gaze_tracking.gpu_utils import GPUManager
    
    config = GazeConfig()
    gpu = GPUManager()
    print(f"   OID_MODEL_SIZE: {config.OID_MODEL_SIZE}")
    print(f"   GPU available: {gpu.is_available}")
    
    detector = ObjectDetector(config, gpu)
    if detector.is_available():
        print("   ✅ ObjectDetector initialized và OID available")
    else:
        print("   ❌ ObjectDetector không available (có thể ultralytics chưa cài hoặc lỗi khởi tạo)")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Lỗi khởi tạo ObjectDetector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("✅ TẤT CẢ ĐỀU OK - OID DETECTOR SẴN SÀNG!")
print("=" * 60)
print()
print("Lưu ý:")
print("  - Model YOLOv8 OID sẽ được download tự động lần đầu tiên sử dụng")
print(f"  - Model size: {config.OID_MODEL_SIZE} (nano/small/medium/large/xlarge)")
print("  - Model có 600 classes, bao gồm 'pen' và 'pencil' ✅")




