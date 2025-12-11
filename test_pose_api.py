"""
Script test cho API Pose & Movement Detection
Phân tích hành vi và cử động từ video

Cách sử dụng:
    python test_pose_api.py <path_to_video_file>
    
Ví dụ:
    python test_pose_api.py test_video.mp4
"""
import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8102/screening/pose/analyze"

# ========================================
# CẤU HÌNH VIDEO PATH - SỬA ĐƯỜNG DẪN Ở ĐÂY
# ========================================
# Cách 1: Đặt đường dẫn trực tiếp (ưu tiên)
VIDEO_PATH = r"C:\Users\Admin\Desktop\mon.mp4"  # <-- SỬA ĐƯỜNG DẪN Ở ĐÂY

# Cách 2: Hoặc để None để dùng command line argument
# VIDEO_PATH = None

# ========================================
# CẤU HÌNH VIDEO DISPLAY
# ========================================
# Bật/tắt hiển thị video real-time trong quá trình xử lý
SHOW_VIDEO = True  # True = hiển thị video, False = không hiển thị

# ========================================

def test_pose_api(video_path: str):
    """
    Test API pose detection với video file
    
    Args:
        video_path: Đường dẫn đến file video
    """
    if not Path(video_path).exists():
        print(f"❌ File không tồn tại: {video_path}")
        return
    
    print("=" * 60)
    print("🏃 POSE & MOVEMENT DETECTION API TEST")
    print("=" * 60)
    print(f"📹 Video: {video_path}")
    print(f"🌐 API URL: {API_URL}")
    if SHOW_VIDEO:
        print(f"📺 Video Display: ENABLED (sẽ hiển thị video real-time)")
        print("   → Nhấn 'q' trong cửa sổ video để tắt hiển thị")
    else:
        print(f"📺 Video Display: DISABLED")
    print("-" * 60)
    
    try:
        # Gửi request với video file
        with open(video_path, 'rb') as video_file:
            files = {
                'video': (Path(video_path).name, video_file, 'video/mp4')
            }
            data = {
                'show_video': 'true' if SHOW_VIDEO else 'false'
            }
            
            print("⏳ Đang xử lý video (có thể mất vài phút)...")
            print("   - Detecting pose landmarks...")
            print("   - Analyzing movement patterns...")
            print("   - Classifying behaviors...")
            response = requests.post(API_URL, files=files, data=data, timeout=600)
        
        # Kiểm tra response
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 60)
            print("✅ PHÂN TÍCH THÀNH CÔNG!")
            print("=" * 60)
            
            # Hiển thị kết quả chính
            print("\n📊 KẾT QUẢ CHÍNH:")
            print("-" * 60)
            print(f"  🏃 Activity Score: {result['activity_score']:.2f}/100")
            if result['activity_score'] > 70:
                print(f"     → ⚠️  Hoạt động cao (có thể là hyperactivity)")
            elif result['activity_score'] > 40:
                print(f"     → ⚠️  Hoạt động trung bình")
            else:
                print(f"     → ✅ Hoạt động bình thường")
            
            print(f"  💨 Movement Intensity: {result['movement_intensity']:.2f}/100")
            print(f"  📈 Risk Score: {result['risk_score']:.2f}/100")
            if result['risk_score'] < 30:
                print(f"     → ✅ Rủi ro thấp (hành vi bình thường)")
            elif result['risk_score'] < 60:
                print(f"     → ⚠️  Rủi ro trung bình")
            else:
                print(f"     → ❌ Rủi ro cao (nhiều hành vi bất thường)")
            
            # Thông tin video
            print("\n📹 THÔNG TIN VIDEO:")
            print("-" * 60)
            print(f"  🎬 Tổng frames: {result['total_frames']:,}")
            print(f"  ⏱️  Thời gian phân tích: {result['analyzed_duration']:.2f}s")
            
            # Detected behaviors
            print("\n🎭 HÀNH VI ĐƯỢC PHÁT HIỆN:")
            print("-" * 60)
            behaviors = result['detected_behaviors']
            
            # Emoji mapping
            behavior_emojis = {
                'hand_flapping': '👋',
                'rocking': '🔄',
                'toe_walking': '👣',
                'spinning': '🌀',
                'hyperactivity': '⚡',
                'normal': '✅'
            }
            
            # Sắp xếp theo percentage
            sorted_behaviors = sorted(behaviors.items(), key=lambda x: x[1], reverse=True)
            for behavior, percentage in sorted_behaviors:
                if percentage > 0:
                    bar_length = int(percentage / 2)  # Scale to 50 chars max
                    bar = "█" * bar_length
                    emoji = behavior_emojis.get(behavior, "•")
                    behavior_name = behavior.replace('_', ' ').title()
                    print(f"  {emoji} {behavior_name:15s}: {percentage:6.2f}% {bar}")
            
            # Thống kê chi tiết
            print("\n📈 THỐNG KÊ CHI TIẾT:")
            print("-" * 60)
            abnormal_behaviors = (
                behaviors.get('hand_flapping', 0) +
                behaviors.get('rocking', 0) +
                behaviors.get('toe_walking', 0) +
                behaviors.get('spinning', 0) +
                behaviors.get('hyperactivity', 0)
            )
            normal_percentage = behaviors.get('normal', 0)
            
            print(f"  ⚠️  Hành vi bất thường: {abnormal_behaviors:.2f}%")
            print(f"  ✅ Hành vi bình thường: {normal_percentage:.2f}%")
            
            if abnormal_behaviors > 30:
                print(f"     → ❌ Tỷ lệ hành vi bất thường cao")
            elif abnormal_behaviors > 15:
                print(f"     → ⚠️  Tỷ lệ hành vi bất thường trung bình")
            else:
                print(f"     → ✅ Tỷ lệ hành vi bất thường thấp")
            
            # Chi tiết từng hành vi
            print("\n🔍 CHI TIẾT HÀNH VI:")
            print("-" * 60)
            if behaviors.get('hand_flapping', 0) > 5:
                print(f"  👋 Hand Flapping: {behaviors['hand_flapping']:.2f}%")
                print(f"     → Tay vẫy nhanh lên xuống")
            if behaviors.get('rocking', 0) > 5:
                print(f"  🔄 Rocking: {behaviors['rocking']:.2f}%")
                print(f"     → Đung đưa cơ thể qua lại")
            if behaviors.get('toe_walking', 0) > 5:
                print(f"  👣 Toe Walking: {behaviors['toe_walking']:.2f}%")
                print(f"     → Đi nhón chân")
            if behaviors.get('spinning', 0) > 5:
                print(f"  🌀 Spinning: {behaviors['spinning']:.2f}%")
                print(f"     → Quay vòng")
            if behaviors.get('hyperactivity', 0) > 5:
                print(f"  ⚡ Hyperactivity: {behaviors['hyperactivity']:.2f}%")
                print(f"     → Di chuyển liên tục")
            
            print("\n" + "=" * 60)
            print("✅ HOÀN TẤT!")
            print("=" * 60)
            
        else:
            print(f"\n❌ Lỗi: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Chi tiết: {error_detail}")
            except:
                print(f"Response: {response.text}")
    
    except requests.exceptions.Timeout:
        print("\n❌ Timeout: Video quá dài hoặc server xử lý chậm")
    except requests.exceptions.ConnectionError:
        print("\n❌ Không thể kết nối đến server")
        print("   Hãy đảm bảo server đang chạy tại:", API_URL)
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Xác định video path
    video_path = VIDEO_PATH
    
    # Nếu VIDEO_PATH là None, dùng command line argument
    if video_path is None or not Path(video_path).exists():
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
        else:
            print("=" * 60)
            print("❌ CẦN CUNG CẤP ĐƯỜNG DẪN VIDEO")
            print("=" * 60)
            print("\nCách sử dụng:")
            print("  1. Sửa VIDEO_PATH trong file này")
            print("  2. Hoặc chạy: python test_pose_api.py <path_to_video>")
            print("\nVí dụ:")
            print("  python test_pose_api.py C:\\Users\\Admin\\Desktop\\video.mp4")
            sys.exit(1)
    
    test_pose_api(video_path)



