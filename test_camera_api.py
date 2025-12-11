"""
Script test cho API Gaze Tracking với Camera Streaming
Phân tích eye contact và focusing time từ camera real-time

Cách sử dụng:
    python test_camera_api.py [camera_id] [max_duration] [show_video]
    
Ví dụ:
    python test_camera_api.py                    # Camera 0, 60s, show video
    python test_camera_api.py 0 60 true          # Camera 0, 60s, show video
    python test_camera_api.py 0 30 false         # Camera 0, 30s, no video
"""
import sys
import requests
import json
import time

API_URL = "http://localhost:8102/screening/gaze/analyze_camera"

# ========================================
# CẤU HÌNH MẶC ĐỊNH
# ========================================
DEFAULT_CAMERA_ID = 0  # Camera ID mặc định
DEFAULT_MAX_DURATION = 60.0  # Thời gian tối đa (giây), 0 = không giới hạn
DEFAULT_SHOW_VIDEO = True  # Hiển thị video real-time
DEFAULT_TARGET_TYPE = "camera"  # Loại target

# ========================================

def test_camera_api(camera_id: int = 0, max_duration: float = 60.0, show_video: bool = True, target_type: str = "camera"):
    """
    Test API gaze tracking với camera streaming
    
    Args:
        camera_id: ID của camera (0, 1, 2, ...)
        max_duration: Thời gian tối đa (giây), 0 = không giới hạn
        show_video: Có hiển thị video không
        target_type: Loại target ("camera", "parent", "face")
    """
    print("=" * 60)
    print("📹 GAZE TRACKING API TEST - CAMERA STREAMING")
    print("=" * 60)
    print(f"📷 Camera ID: {camera_id}")
    print(f"⏱️  Max Duration: {max_duration}s" if max_duration > 0 else "⏱️  Max Duration: Không giới hạn")
    print(f"🎯 Target: {target_type}")
    print(f"🌐 API URL: {API_URL}")
    if show_video:
        print(f"📺 Video Display: ENABLED (sẽ hiển thị video real-time)")
        print("   → Nhấn 'q' trong cửa sổ video để dừng phân tích")
    else:
        print(f"📺 Video Display: DISABLED")
    print("-" * 60)
    print()
    print("⚠️  LƯU Ý:")
    print("   - Camera sẽ tự động mở khi gửi request")
    print("   - Phân tích sẽ dừng khi:")
    print("     + Nhấn 'q' trong cửa sổ video (nếu show_video=true)")
    print("     + Đạt max_duration")
    print("     + Có lỗi xảy ra")
    print()
    
    try:
        # Gửi request với camera parameters
        data = {
            'camera_id': str(camera_id),
            'max_duration': str(max_duration),
            'show_video': 'true' if show_video else 'false',
            'target_type': target_type
        }
        
        print("⏳ Đang gửi request đến server...")
        print("   - Mở camera...")
        print("   - Bắt đầu phân tích...")
        print("   - Detecting faces (trẻ + người lớn)...")
        print("   - Tracking gaze direction...")
        print("   - Calculating focusing time...")
        print()
        print("⏳ Đang xử lý (có thể mất vài phút)...")
        print("   (Nhấn Ctrl+C để hủy)")
        print()
        
        start_time = time.time()
        response = requests.post(API_URL, data=data, timeout=max_duration + 30 if max_duration > 0 else 600)
        elapsed_time = time.time() - start_time
        
        # Kiểm tra response
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 60)
            print("✅ PHÂN TÍCH THÀNH CÔNG!")
            print("=" * 60)
            print(f"⏱️  Thời gian xử lý: {elapsed_time:.2f}s")
            print()
            
            # Hiển thị kết quả chính
            print("📊 KẾT QUẢ CHÍNH:")
            print("-" * 60)
            print(f"  👁️  Eye Contact (Focusing): {result['eye_contact_percentage']:.2f}%")
            print(f"     → Thời gian focusing vào item cố định (camera/người lớn/đồ vật)")
            print(f"  ⏱️  Focusing Duration: {result.get('focusing_duration', 0):.2f}s")
            print(f"  👤 Attention to Person: {result.get('attention_to_person_percentage', 0):.2f}%")
            print(f"     → Thời gian chú ý vào người tương tác")
            print(f"  📦 Attention to Objects: {result.get('attention_to_objects_percentage', 0):.2f}%")
            print(f"     → Thời gian chú ý vào đồ vật (sách, bút, etc.)")
            print(f"  📖 Attention to Book: {result.get('attention_to_book_percentage', 0):.2f}%")
            print(f"     → Thời gian chú ý vào sách")
            print(f"  🎯 Book Focusing Score: {result.get('book_focusing_score', 0):.2f}/100")
            if result.get('book_focusing_score', 0) > 70:
                print(f"     → ✅ Focusing tốt vào sách")
            elif result.get('book_focusing_score', 0) > 40:
                print(f"     → ⚠️  Focusing trung bình")
            else:
                print(f"     → ❌ Focusing kém vào sách")
            print(f"  📈 Risk Score: {result['risk_score']:.2f}/100")
            if result['risk_score'] < 30:
                print(f"     → ✅ Rủi ro thấp (eye contact tốt)")
            elif result['risk_score'] < 60:
                print(f"     → ⚠️  Rủi ro trung bình")
            else:
                print(f"     → ❌ Rủi ro cao (eye contact kém)")
            
            # Thông tin video
            print("\n📹 THÔNG TIN PHÂN TÍCH:")
            print("-" * 60)
            print(f"  🎬 Tổng frames: {result['total_frames']:,}")
            print(f"  ⏱️  Thời gian phân tích: {result['analyzed_duration']:.2f}s")
            if result['analyzed_duration'] > 0:
                focusing_ratio = result.get('focusing_duration', 0) / result['analyzed_duration']
                print(f"  📊 Tỷ lệ focusing: {focusing_ratio * 100:.1f}%")
            
            # Hướng nhìn
            print("\n📐 THỐNG KÊ HƯỚNG NHÌN:")
            print("-" * 60)
            gaze_stats = result['gaze_direction_stats']
            sorted_gaze = sorted(gaze_stats.items(), key=lambda x: x[1], reverse=True)
            for direction, percentage in sorted_gaze:
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                direction_emoji = {
                    "center": "👁️",
                    "left": "⬅️",
                    "right": "➡️",
                    "up": "⬆️",
                    "down": "⬇️"
                }.get(direction, "•")
                print(f"  {direction_emoji} {direction.capitalize():8s}: {percentage:6.2f}% {bar}")
            
            # Detected Objects
            if result.get('detected_objects'):
                print("\n📦 ĐỒ VẬT ĐƯỢC PHÁT HIỆN:")
                print("-" * 60)
                for obj in result['detected_objects'][:10]:
                    obj_name = obj.get('class', 'unknown')
                    count = obj.get('detection_count', 0)
                    percentage = obj.get('percentage', 0)
                    emoji = "📖" if obj_name == 'book' else "📦"
                    print(f"  {emoji} {obj_name.capitalize():15s}: {count:4d} lần ({percentage:.1f}%)")
                if len(result['detected_objects']) > 10:
                    print(f"  ... và {len(result['detected_objects']) - 10} objects khác")
            
            # Gaze Wandering
            if result.get('gaze_wandering_percentage') is not None:
                print("\n👀 GAZE WANDERING:")
                print("-" * 60)
                wandering_percentage = result.get('gaze_wandering_percentage', 0)
                wandering_score = result.get('gaze_wandering_score', 0)
                print(f"  📊 Wandering Percentage: {wandering_percentage:.2f}%")
                print(f"  📈 Wandering Score: {wandering_score:.2f}/100")
                if wandering_percentage > 30:
                    print(f"     → ⚠️  Nhìn vô định nhiều")
                elif wandering_percentage > 15:
                    print(f"     → ⚠️  Nhìn vô định trung bình")
                else:
                    print(f"     → ✅ Nhìn vô định ít")
            
            # Fatigue
            if result.get('fatigue_score') is not None:
                print("\n😴 FATIGUE DETECTION:")
                print("-" * 60)
                fatigue_score = result.get('fatigue_score', 0)
                fatigue_level = result.get('fatigue_level', 'low')
                print(f"  📊 Fatigue Score: {fatigue_score:.2f}/100")
                print(f"  📈 Fatigue Level: {fatigue_level}")
                if fatigue_score > 50:
                    print(f"     → ⚠️  Mệt mỏi cao")
                elif fatigue_score > 30:
                    print(f"     → ⚠️  Mệt mỏi trung bình")
                else:
                    print(f"     → ✅ Mệt mỏi thấp")
            
            # Focus Level
            if result.get('focus_level') is not None:
                print("\n🎯 FOCUS LEVEL:")
                print("-" * 60)
                focus_level = result.get('focus_level', 0)
                print(f"  📊 Focus Level: {focus_level:.2f}/100")
                if focus_level > 70:
                    print(f"     → ✅ Tập trung tốt")
                elif focus_level > 50:
                    print(f"     → ⚠️  Tập trung trung bình")
                else:
                    print(f"     → ❌ Tập trung kém")
            
            # Model Information
            if 'object_detection_model' in result or 'object_detection_available' in result:
                print("\n" + "=" * 60)
                print("🤖 MODEL INFORMATION:")
                print("-" * 60)
                model_name = result.get('object_detection_model', 'N/A')
                model_available = result.get('object_detection_available', False)
                
                if model_available:
                    print(f"  ✅ Object Detection: {model_name}")
                    print(f"  ✅ Status: Available")
                else:
                    print(f"  ❌ Object Detection: Not Available")
                    print(f"  💡 Để bật: pip install ultralytics>=8.0.0")
                print("=" * 60)
            
            # JSON output (optional)
            print("\n" + "=" * 60)
            print("📄 JSON Response:")
            print("-" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 60)
            
        else:
            print("\n" + "=" * 60)
            print(f"❌ LỖI: HTTP {response.status_code}")
            print("=" * 60)
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2, ensure_ascii=False))
            except:
                print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 60)
        print("❌ KHÔNG THỂ KẾT NỐI ĐẾN SERVER!")
        print("=" * 60)
        print("   Hãy đảm bảo server đang chạy tại http://localhost:8102")
        print("   Chạy lệnh: python main.py")
        print("=" * 60)
    except requests.exceptions.Timeout:
        print("\n" + "=" * 60)
        print("❌ REQUEST TIMEOUT!")
        print("=" * 60)
        print("   Phân tích mất quá nhiều thời gian.")
        print("   Hãy thử giảm max_duration hoặc kiểm tra camera.")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠️  Người dùng hủy (Ctrl+C)")
        print("   Camera đã được đóng")
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ LỖI: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    camera_id = DEFAULT_CAMERA_ID
    max_duration = DEFAULT_MAX_DURATION
    show_video = DEFAULT_SHOW_VIDEO
    target_type = DEFAULT_TARGET_TYPE
    
    if len(sys.argv) >= 2:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Warning: camera_id '{sys.argv[1]}' không hợp lệ, sử dụng {DEFAULT_CAMERA_ID}")
            camera_id = DEFAULT_CAMERA_ID
    
    if len(sys.argv) >= 3:
        try:
            max_duration = float(sys.argv[2])
        except ValueError:
            print(f"⚠️  Warning: max_duration '{sys.argv[2]}' không hợp lệ, sử dụng {DEFAULT_MAX_DURATION}")
            max_duration = DEFAULT_MAX_DURATION
    
    if len(sys.argv) >= 4:
        show_video_str = sys.argv[3].lower()
        show_video = show_video_str in ("true", "1", "yes", "on")
    
    if len(sys.argv) >= 5:
        target_type = sys.argv[4]
    
    # Validate target_type
    if target_type not in ["camera", "parent", "face"]:
        print(f"⚠️  Warning: target_type '{target_type}' không hợp lệ, sử dụng 'camera'")
        target_type = "camera"
    
    print("=" * 60)
    print("📹 CAMERA API TEST - CONFIGURATION")
    print("=" * 60)
    print(f"   Camera ID: {camera_id}")
    print(f"   Max Duration: {max_duration}s" if max_duration > 0 else "   Max Duration: Không giới hạn")
    print(f"   Show Video: {show_video}")
    print(f"   Target Type: {target_type}")
    print("=" * 60)
    print()
    
    test_camera_api(camera_id, max_duration, show_video, target_type)

