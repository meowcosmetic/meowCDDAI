"""
Script test để kiểm tra config có được apply đúng không
"""
from text_embeding.gaze_tracking.config import GazeConfig

print("=" * 70)
print("KIỂM TRA CONFIG VALUES")
print("=" * 70)
print()

config = GazeConfig()

print("📋 Các config quan trọng:")
print(f"   MAX_FRAME_WIDTH: {config.MAX_FRAME_WIDTH} pixels")
print(f"   MIN_FOCUSING_DURATION: {config.MIN_FOCUSING_DURATION} giây")
print(f"   GAZE_STABILITY_THRESHOLD: {config.GAZE_STABILITY_THRESHOLD}")
print(f"   OBJECT_DETECTION_INTERVAL: {config.OBJECT_DETECTION_INTERVAL} frames")
print(f"   OBJECT_CONFIDENCE_THRESHOLD: {config.OBJECT_CONFIDENCE_THRESHOLD}")
print(f"   OID_MODEL_SIZE: {config.OID_MODEL_SIZE}")
print(f"   FPS_DEFAULT: {config.FPS_DEFAULT}")
print()

print("=" * 70)
print("KIỂM TRA TRONG CODE:")
print("=" * 70)
print()

# Đọc routes_screening_gaze.py
with open("text_embeding/routes_screening_gaze.py", "r", encoding="utf-8") as f:
    content = f.read()

# Kiểm tra MAX_FRAME_WIDTH
if "config.MAX_FRAME_WIDTH" in content or "MAX_FRAME_WIDTH" in content:
    print("✅ MAX_FRAME_WIDTH: Được sử dụng trong code")
    if "config.MAX_FRAME_WIDTH" in content:
        print("   ✅ Dùng config.MAX_FRAME_WIDTH (đúng)")
    else:
        print("   ⚠️  Có thể dùng biến local thay vì config")
else:
    print("❌ MAX_FRAME_WIDTH: KHÔNG được sử dụng")

# Kiểm tra FPS_DEFAULT
if "config.FPS_DEFAULT" in content:
    print("✅ FPS_DEFAULT: Được sử dụng trong code")
elif "fps = fps if fps > 0 else" in content:
    print("⚠️  FPS_DEFAULT: Có hardcoded value, nên dùng config.FPS_DEFAULT")
else:
    print("❌ FPS_DEFAULT: KHÔNG được sử dụng")

# Kiểm tra resize
if "resize" in content.lower() and "max_width" in content.lower():
    print("✅ Frame resize: Có trong code")
    if "config.MAX_FRAME_WIDTH" in content:
        print("   ✅ Dùng config.MAX_FRAME_WIDTH (đúng)")
    else:
        print("   ⚠️  Có thể dùng hardcoded value")
else:
    print("❌ Frame resize: KHÔNG tìm thấy")

print()
print("=" * 70)
print("KẾT LUẬN:")
print("=" * 70)
print(f"MAX_FRAME_WIDTH hiện tại: {config.MAX_FRAME_WIDTH} pixels")
print("→ Video sẽ được resize nếu width > giá trị này")
print()
print("💡 Để thay đổi kích thước hiển thị:")
print(f"   Sửa MAX_FRAME_WIDTH trong config.py (hiện tại: {config.MAX_FRAME_WIDTH})")
print("   Ví dụ: MAX_FRAME_WIDTH = 1280  # Cho màn hình lớn hơn")





