"""
Test script để test Gaze Analysis với camera
Chạy script này để test camera real-time

Cách chạy:
    # Với venv312 (khuyến nghị):
    .\\venv312\\Scripts\\python.exe test_camera_gaze.py
    
    # Hoặc activate venv312 trước:
    .\\venv312\\Scripts\\Activate.ps1
    python test_camera_gaze.py
"""
import cv2
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import cần thiết
try:
    from text_embeding.gaze.processor import process_gaze_analysis
except ImportError as e:
    print(f"❌ Lỗi import: {str(e)}")
    print("Vui lòng đảm bảo bạn đang chạy từ thư mục gốc của project")
    sys.exit(1)

def test_camera(camera_id=0):
    """Test camera với gaze analysis"""
    print("=" * 60)
    print("Test Gaze Analysis với Camera")
    print("=" * 60)
    print("\nHướng dẫn:")
    print("- Camera sẽ tự động mở và chạy LIÊN TỤC")
    print("- Nhấn 'q' trong cửa sổ video để dừng")
    print("- Hoặc nhấn Ctrl+C trong terminal để dừng")
    print("- Chương trình sẽ chạy cho đến khi bạn dừng")
    print("\nBắt đầu trong 3 giây...")
    
    import time
    time.sleep(3)
    
    # Mở camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở camera ID {camera_id}")
        print("Vui lòng kiểm tra:")
        print("1. Camera có được kết nối không?")
        print("2. Camera có đang được sử dụng bởi ứng dụng khác không?")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"✅ Camera đã mở: Camera ID {camera_id}")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"   FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
    print("\n" + "=" * 60)
    print("Bắt đầu phân tích...")
    print("Nhấn 'q' trong cửa sổ video để dừng")
    print("=" * 60 + "\n")
    
    try:
        # Gọi hàm process_gaze_analysis
        # max_duration=0.0 nghĩa là không giới hạn thời gian, chạy liên tục
        result = process_gaze_analysis(
            cap=cap,
            target_type="camera",
            show_video=True,
            max_duration=0.0,  # 0 = không giới hạn, chạy liên tục cho đến khi nhấn 'q' hoặc Ctrl+C
            is_camera=True
        )
        
        # Hiển thị kết quả
        print("\n" + "=" * 60)
        print("KẾT QUẢ PHÂN TÍCH")
        print("=" * 60)
        print(f"Tổng số frames: {result.total_frames}")
        print(f"Thời gian phân tích: {result.analyzed_duration:.2f} giây")
        print(f"Eye contact percentage: {result.eye_contact_percentage:.2f}%")
        print(f"Focusing duration: {result.focusing_duration:.2f} giây")
        print(f"\nGaze directions:")
        for direction, percentage in result.gaze_direction_stats.items():
            print(f"  {direction}: {percentage:.2f}%")
        print(f"\nAttention:")
        print(f"  To person: {result.attention_to_person_percentage:.2f}%")
        print(f"  To objects: {result.attention_to_objects_percentage:.2f}%")
        print(f"  To book: {result.attention_to_book_percentage:.2f}%")
        print(f"\nRisk score: {result.risk_score:.2f}")
        print(f"Gaze wandering: {result.gaze_wandering_percentage:.2f}%")
        print(f"Fatigue score: {result.fatigue_score:.2f} ({result.fatigue_level})")
        print(f"Focus level: {result.focus_level:.2f}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Người dùng dừng (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Camera đã được đóng")

if __name__ == "__main__":
    # Nhận camera_id từ command line argument nếu có
    camera_id = 0  # Default
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Camera ID không hợp lệ: {sys.argv[1]}, sử dụng camera mặc định (0)")
            camera_id = 0
    
    test_camera(camera_id=camera_id)

