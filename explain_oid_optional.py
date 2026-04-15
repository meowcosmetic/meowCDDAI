"""
Giải thích tại sao Gaze API vẫn chạy được khi chưa cài ultralytics
"""
print("=" * 70)
print("TẠI SAO GAZE API VẪN CHẠY ĐƯỢC KHI CHƯA CÀI ULTRALYTICS?")
print("=" * 70)
print()

print("1. OBJECT DETECTION KHÔNG BẮT BUỘC:")
print("   - Gaze tracking (hướng nhìn) hoạt động độc lập với object detection")
print("   - API vẫn có thể:")
print("     ✅ Track hướng nhìn (left/right/up/down)")
print("     ✅ Tính eye contact percentage")
print("     ✅ Detect nhìn vào camera")
print("     ✅ Tính gaze stability")
print("     ✅ Detect nhìn vào adult (face detection)")
print()
print("   - Object detection chỉ là BỔ SUNG để:")
print("     📦 Biết trẻ đang nhìn vào object nào cụ thể (pen, book, etc.)")
print("     📦 Track objects qua frames")
print("     📦 Tính attention_to_objects_percentage")
print()

print("2. CODE CÓ ERROR HANDLING TỐT:")
print("   - ObjectDetector được khởi tạo nhưng nếu OID không available:")
print("     → Chỉ log error, không crash")
print("     → object_detector_new.is_available() = False")
print("   - Code check trước khi dùng:")
print("     if object_detector_new and object_detector_new.is_available():")
print("         # Chỉ chạy object detection nếu available")
print()

print("3. KẾT QUẢ KHI CHƯA CÀI ULTRALYTICS:")
print("   ✅ Gaze API vẫn chạy được")
print("   ✅ Vẫn track được hướng nhìn")
print("   ✅ Vẫn tính được eye contact")
print("   ❌ KHÔNG detect được objects (pen, book, etc.)")
print("   ❌ attention_to_objects_percentage = 0")
print("   ❌ detected_objects = []")
print()

print("4. ĐỂ CÓ OBJECT DETECTION:")
print("   👉 Cài ultralytics: pip install ultralytics>=8.0.0")
print("   👉 Model sẽ được download tự động lần đầu")
print("   👉 Sau đó sẽ detect được pen, pencil, book, etc.")
print()

print("=" * 70)
print("KIỂM TRA TRẠNG THÁI HIỆN TẠI:")
print("=" * 70)

try:
    from ultralytics import YOLO
    print("✅ ultralytics: ĐÃ CÀI")
except ImportError:
    print("❌ ultralytics: CHƯA CÀI")
    print("   → Gaze API chạy được nhưng không có object detection")

try:
    from text_embeding.gaze_tracking.object_detector import ObjectDetector
    from text_embeding.gaze_tracking.config import GazeConfig
    from text_embeding.gaze_tracking.gpu_utils import GPUManager
    
    config = GazeConfig()
    gpu = GPUManager()
    detector = ObjectDetector(config, gpu)
    
    if detector.is_available():
        print("✅ ObjectDetector: AVAILABLE")
        print("   → Có thể detect objects (pen, book, etc.)")
    else:
        print("❌ ObjectDetector: KHÔNG AVAILABLE")
        print("   → Gaze API vẫn chạy nhưng không detect objects")
except Exception as e:
    print(f"⚠️  Lỗi kiểm tra: {e}")

print()
print("=" * 70)







