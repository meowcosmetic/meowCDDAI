"""
Script test cho API Interaction Detection
Phân tích tương tác xã hội từ video

Cách sử dụng:
    python test_interaction_api.py <path_to_video_file>
    
Ví dụ:
    python test_interaction_api.py test_video.mp4
"""
import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8102/screening/interaction/analyze"

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

def test_interaction_api(video_path: str):
    """
    Test API interaction detection với video file
    
    Args:
        video_path: Đường dẫn đến file video
    """
    if not Path(video_path).exists():
        print(f"❌ File không tồn tại: {video_path}")
        return
    
    print("=" * 60)
    print("🤝 INTERACTION DETECTION API TEST")
    print("=" * 60)
    print(f"📹 Video: {video_path}")
    print(f"🌐 API URL: {API_URL}")
    if SHOW_VIDEO:
        print(f"📺 Video Display: ENABLED (sẽ hiển thị video real-time)")
        print("   → Nhấn 'q' hoặc ESC trong cửa sổ video để tắt hiển thị")
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
            print("   - Detecting objects and people...")
            print("   - Tracking objects...")
            print("   - Detecting hand gestures...")
            print("   - Analyzing interactions...")
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
            print(f"  🤝 Interaction Score: {result['interaction_score']:.2f}/100")
            if result['interaction_score'] > 70:
                print(f"     → ✅ Tương tác tốt")
            elif result['interaction_score'] > 40:
                print(f"     → ⚠️  Tương tác trung bình")
            else:
                print(f"     → ❌ Tương tác thấp")
            
            print(f"  📈 Response Rate: {result['response_rate']:.2f}%")
            print(f"     → Tỷ lệ trẻ phản hồi khi người lớn đưa đồ vật")
            
            print(f"  👆 Pointing Gestures: {result['pointing_gestures']}")
            print(f"  🔄 Object Exchanges: {result['object_exchanges']}")
            
            print(f"  📈 Risk Score: {result['risk_score']:.2f}/100")
            if result['risk_score'] < 30:
                print(f"     → ✅ Rủi ro thấp (tương tác tốt)")
            elif result['risk_score'] < 60:
                print(f"     → ⚠️  Rủi ro trung bình")
            else:
                print(f"     → ❌ Rủi ro cao (tương tác kém)")
            
            # Thông tin video
            print("\n📹 THÔNG TIN VIDEO:")
            print("-" * 60)
            print(f"  🎬 Tổng frames: {result['total_frames']:,}")
            print(f"  ⏱️  Thời gian phân tích: {result['analyzed_duration']:.2f}s")
            
            # Interaction events
            if result.get('interaction_events'):
                print("\n🎯 SỰ KIỆN TƯƠNG TÁC:")
                print("-" * 60)
                
                # Nhóm events theo type
                events_by_type = {}
                for event in result['interaction_events']:
                    event_type = event.get('type', 'unknown')
                    if event_type not in events_by_type:
                        events_by_type[event_type] = []
                    events_by_type[event_type].append(event)
                
                # Emoji mapping
                event_emojis = {
                    'pointing': '👆',
                    'object_offer': '📤',
                    'following': '👀',
                    'object_exchange': '🔄',
                    'pointing_at_object': '👉'
                }
                
                for event_type, events in events_by_type.items():
                    emoji = event_emojis.get(event_type, '•')
                    print(f"  {emoji} {event_type.replace('_', ' ').title()}: {len(events)} lần")
                    
                    # Hiển thị một số events đầu tiên
                    for event in events[:3]:
                        start_time = event.get('start_time', event.get('timestamp', 0))
                        duration = event.get('duration', None)
                        description = event.get('description', '')
                        if duration is not None:
                            print(f"     • {start_time:.1f}s (+{duration:.1f}s): {description}")
                        else:
                            print(f"     • {start_time:.1f}s: {description}")
                    
                    if len(events) > 3:
                        print(f"     ... và {len(events) - 3} events khác")
                
                # Thống kê chi tiết
                print("\n📈 THỐNG KÊ CHI TIẾT:")
                print("-" * 60)
                total_events = len(result['interaction_events'])
                events_per_second = total_events / result['analyzed_duration'] if result['analyzed_duration'] > 0 else 0
                print(f"  📊 Tổng số events: {total_events}")
                print(f"  ⚡ Tần suất: {events_per_second:.2f} events/giây")
                
                if events_per_second > 0.5:
                    print(f"     → ✅ Tần suất tương tác cao")
                elif events_per_second > 0.2:
                    print(f"     → ⚠️  Tần suất tương tác trung bình")
                else:
                    print(f"     → ❌ Tần suất tương tác thấp")
            
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
            print("  2. Hoặc chạy: python test_interaction_api.py <path_to_video>")
            print("\nVí dụ:")
            print("  python test_interaction_api.py C:\\Users\\Admin\\Desktop\\video.mp4")
            sys.exit(1)
    
    test_interaction_api(video_path)



