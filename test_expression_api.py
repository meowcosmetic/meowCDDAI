"""
Script test cho API Facial Expression Recognition
Phân tích biểu cảm khuôn mặt từ video

Cách sử dụng:
    python test_expression_api.py <path_to_video_file>
    
Ví dụ:
    python test_expression_api.py test_video.mp4
"""
import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8102/screening/expression/analyze"

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

def test_expression_api(video_path: str):
    """
    Test API expression recognition với video file
    
    Args:
        video_path: Đường dẫn đến file video
    """
    if not Path(video_path).exists():
        print(f"❌ File không tồn tại: {video_path}")
        return
    
    print("=" * 60)
    print("😊 FACIAL EXPRESSION RECOGNITION API TEST")
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
            print("   - Detecting faces...")
            print("   - Extracting facial features...")
            print("   - Classifying expressions...")
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
            print(f"  🎭 Expression Diversity Score: {result['expression_diversity_score']:.2f}/100")
            if result['expression_diversity_score'] > 70:
                print(f"     → ✅ Đa dạng biểu cảm tốt")
            elif result['expression_diversity_score'] > 40:
                print(f"     → ⚠️  Đa dạng biểu cảm trung bình")
            else:
                print(f"     → ❌ Đa dạng biểu cảm thấp")
            
            print(f"  😐 Neutral Percentage: {result['neutral_percentage']:.2f}%")
            print(f"  📈 Risk Score: {result['risk_score']:.2f}/100")
            if result['risk_score'] < 30:
                print(f"     → ✅ Rủi ro thấp (biểu cảm đa dạng)")
            elif result['risk_score'] < 60:
                print(f"     → ⚠️  Rủi ro trung bình")
            else:
                print(f"     → ❌ Rủi ro cao (ít biểu cảm)")
            
            # Thông tin video
            print("\n📹 THÔNG TIN VIDEO:")
            print("-" * 60)
            print(f"  🎬 Tổng frames: {result['total_frames']:,}")
            print(f"  ⏱️  Thời gian phân tích: {result['analyzed_duration']:.2f}s")
            
            # Phân bố biểu cảm
            print("\n😊 PHÂN BỐ BIỂU CẢM:")
            print("-" * 60)
            expression_dist = result['expression_distribution']
            
            # Emoji mapping
            expression_emojis = {
                "happy": "😊",
                "sad": "😢",
                "angry": "😠",
                "neutral": "😐",
                "surprised": "😲",
                "fearful": "😨",
                "disgusted": "🤢"
            }
            
            # Sắp xếp theo percentage
            sorted_expressions = sorted(expression_dist.items(), key=lambda x: x[1], reverse=True)
            for expression, percentage in sorted_expressions:
                if percentage > 0:
                    bar_length = int(percentage / 2)  # Scale to 50 chars max
                    bar = "█" * bar_length
                    emoji = expression_emojis.get(expression, "•")
                    print(f"  {emoji} {expression.capitalize():12s}: {percentage:6.2f}% {bar}")
            
            # Thống kê chi tiết
            print("\n📈 THỐNG KÊ CHI TIẾT:")
            print("-" * 60)
            total_expressions = sum(expression_dist.values())
            positive_expressions = expression_dist.get('happy', 0) + expression_dist.get('surprised', 0)
            negative_expressions = expression_dist.get('sad', 0) + expression_dist.get('angry', 0) + expression_dist.get('fearful', 0)
            
            print(f"  😊 Biểu cảm tích cực: {positive_expressions:.2f}%")
            print(f"  😢 Biểu cảm tiêu cực: {negative_expressions:.2f}%")
            print(f"  😐 Biểu cảm trung tính: {result['neutral_percentage']:.2f}%")
            
            if positive_expressions > negative_expressions:
                print(f"     → ✅ Tổng thể tích cực")
            elif negative_expressions > positive_expressions:
                print(f"     → ⚠️  Tổng thể tiêu cực")
            else:
                print(f"     → 😐 Tổng thể trung tính")
            
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
            print("  2. Hoặc chạy: python test_expression_api.py <path_to_video>")
            print("\nVí dụ:")
            print("  python test_expression_api.py C:\\Users\\Admin\\Desktop\\video.mp4")
            sys.exit(1)
    
    test_expression_api(video_path)



