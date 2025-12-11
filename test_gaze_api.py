"""
Script test cho API Gaze Tracking
Phân tích eye contact và focusing time (vào camera hoặc người lớn)

Cách sử dụng:
    python test_gaze_api.py <path_to_video_file> [target_type]
    
Ví dụ:
    python test_gaze_api.py test_video.mp4
    python test_gaze_api.py test_video.mp4 camera
    python test_gaze_api.py test_video.mp4 parent
"""
import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8102/screening/gaze/analyze"

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

def test_gaze_api(video_path: str, target_type: str = "camera"):
    """
    Test API gaze tracking với video file
    
    Args:
        video_path: Đường dẫn đến file video
        target_type: Loại target ("camera", "parent", "face")
    """
    if not Path(video_path).exists():
        print(f"❌ File không tồn tại: {video_path}")
        return
    
    print("=" * 60)
    print("🔍 GAZE TRACKING API TEST")
    print("=" * 60)
    print(f"📹 Video: {video_path}")
    print(f"🎯 Target: {target_type}")
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
                'target_type': target_type,
                'show_video': 'true' if SHOW_VIDEO else 'false'  # Bật/tắt hiển thị video
            }
            
            print("⏳ Đang xử lý video (có thể mất vài phút)...")
            print("   - Detecting faces (trẻ + người lớn)...")
            print("   - Tracking gaze direction...")
            print("   - Calculating focusing time...")
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
            print("\n📹 THÔNG TIN VIDEO:")
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
            # Sắp xếp theo percentage
            sorted_gaze = sorted(gaze_stats.items(), key=lambda x: x[1], reverse=True)
            for direction, percentage in sorted_gaze:
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length
                direction_emoji = {
                    "center": "👁️",
                    "left": "⬅️",
                    "right": "➡️",
                    "up": "⬆️",
                    "down": "⬇️"
                }.get(direction, "•")
                print(f"  {direction_emoji} {direction.capitalize():8s}: {percentage:6.2f}% {bar}")
            
            # Detected Books (ưu tiên hiển thị)
            if result.get('detected_books'):
                print("\n📖 SÁCH ĐƯỢC PHÁT HIỆN:")
                print("-" * 60)
                for book in result['detected_books']:
                    count = book.get('detection_count', 0)
                    percentage = book.get('percentage', 0)
                    avg_conf = book.get('avg_confidence', 0)
                    first_frame = book.get('first_detection_frame', 0)
                    last_frame = book.get('last_detection_frame', 0)
                    print(f"  📖 Book:")
                    print(f"     • Số lần phát hiện: {count}")
                    print(f"     • Tỷ lệ: {percentage:.1f}%")
                    print(f"     • Độ tin cậy trung bình: {avg_conf:.2f}")
                    print(f"     • Frame: {first_frame} → {last_frame}")
            
            # Detected Objects
            if result.get('detected_objects'):
                print("\n📦 ĐỒ VẬT ĐƯỢC PHÁT HIỆN:")
                print("-" * 60)
                for obj in result['detected_objects'][:10]:  # Top 10
                    obj_name = obj.get('class', 'unknown')
                    count = obj.get('detection_count', 0)
                    percentage = obj.get('percentage', 0)
                    emoji = "📖" if obj_name == 'book' else "📦"
                    print(f"  {emoji} {obj_name.capitalize():15s}: {count:4d} lần ({percentage:.1f}%)")
                if len(result['detected_objects']) > 10:
                    print(f"  ... và {len(result['detected_objects']) - 10} objects khác")
            
            # Object Interaction Events
            if result.get('object_interaction_events'):
                print("\n🎯 SỰ KIỆN TƯƠNG TÁC:")
                print("-" * 60)
                for event in result['object_interaction_events'][:10]:  # Top 10
                    event_type = event.get('type', 'unknown')
                    obj_class = event.get('object_class', 'unknown')
                    duration = event.get('duration', 0)
                    start_time = event.get('start_time', 0)
                    focusing_score = event.get('focusing_score', None)
                    
                    if event_type == "book_attention":
                        event_emoji = "📖"
                        score_text = f" (Focusing: {focusing_score:.1f}/100)" if focusing_score is not None else ""
                    elif event_type == "person_attention":
                        event_emoji = "👤"
                        score_text = ""
                    else:
                        event_emoji = "📦"
                        score_text = ""
                    
                    print(f"  {event_emoji} {obj_class.capitalize():15s}: {duration:.1f}s (từ {start_time:.1f}s){score_text}")
                if len(result['object_interaction_events']) > 10:
                    print(f"  ... và {len(result['object_interaction_events']) - 10} events khác")
            
            # Focus Timeline (NEW - với Object Tracking)
            if result.get('focus_timeline'):
                print("\n📅 FOCUS TIMELINE (Chi tiết từng object):")
                print("-" * 60)
                timeline = result['focus_timeline']
                print(f"  Tổng số focus periods: {len(timeline)}")
                
                # Group by object
                objects_timeline = {}
                for period in timeline:
                    obj_id = period.get('object_id', 'unknown')
                    if obj_id not in objects_timeline:
                        objects_timeline[obj_id] = []
                    objects_timeline[obj_id].append(period)
                
                for obj_id, periods in objects_timeline.items():
                    total_duration = sum(p.get('duration', 0) for p in periods)
                    focus_count = len(periods)
                    class_name = periods[0].get('class_name', 'unknown')
                    track_id = periods[0].get('track_id')
                    
                    emoji = "📖" if class_name == 'book' else ("👤" if class_name == 'person' else "📦")
                    print(f"\n  {emoji} {obj_id}:")
                    print(f"     • Tổng thời gian: {total_duration:.2f}s")
                    print(f"     • Số lần focus: {focus_count}")
                    if track_id:
                        print(f"     • Track ID: {track_id}")
                    
                    # Hiển thị các periods
                    for i, period in enumerate(periods[:5]):  # Top 5 periods
                        start = period.get('start_time', 0)
                        end = period.get('end_time', 0)
                        duration = period.get('duration', 0)
                        print(f"     {i+1}. {start:.1f}s → {end:.1f}s ({duration:.1f}s)")
                    if len(periods) > 5:
                        print(f"     ... và {len(periods) - 5} periods khác")
            
            # Object Focus Stats (NEW)
            if result.get('object_focus_stats'):
                print("\n📊 THỐNG KÊ FOCUS THEO OBJECT:")
                print("-" * 60)
                stats = result['object_focus_stats']
                sorted_stats = sorted(stats.items(), key=lambda x: x[1].get('total_duration', 0), reverse=True)
                
                for obj_id, stat in sorted_stats[:10]:  # Top 10
                    total_duration = stat.get('total_duration', 0)
                    total_frames = stat.get('total_frames', 0)
                    focus_count = stat.get('focus_count', 0)
                    
                    class_name = obj_id.split('_')[0] if '_' in obj_id else obj_id
                    emoji = "📖" if class_name == 'book' else ("👤" if class_name == 'person' else "📦")
                    
                    print(f"  {emoji} {obj_id:20s}: {total_duration:6.2f}s ({focus_count} lần, {total_frames} frames)")
            
            # Pattern Analysis (NEW - Phát hiện quay lại nhìn object cũ)
            if result.get('pattern_analysis'):
                pattern = result['pattern_analysis']
                print("\n🔄 PHÂN TÍCH PATTERN:")
                print("-" * 60)
                
                revisit_count = pattern.get('revisit_count', 0)
                total_unique = pattern.get('total_unique_objects', 0)
                
                print(f"  Tổng số objects được nhìn: {total_unique}")
                print(f"  Số objects được quay lại nhìn: {revisit_count}")
                
                if pattern.get('revisited_objects'):
                    print("\n  🔁 Objects được quay lại nhìn:")
                    for obj_info in pattern['revisited_objects']:
                        obj_id = obj_info.get('object_id', 'unknown')
                        focus_count = obj_info.get('focus_count', 0)
                        total_duration = obj_info.get('total_duration', 0)
                        
                        class_name = obj_id.split('_')[0] if '_' in obj_id else obj_id
                        emoji = "📖" if class_name == 'book' else ("👤" if class_name == 'person' else "📦")
                        
                        print(f"    {emoji} {obj_id}: {focus_count} lần, tổng {total_duration:.1f}s")
                        
                        # Hiển thị periods
                        periods = obj_info.get('periods', [])
                        for i, p in enumerate(periods[:3]):  # Top 3
                            print(f"       {i+1}. {p.get('start', 0):.1f}s - {p.get('end', 0):.1f}s ({p.get('duration', 0):.1f}s)")
                        if len(periods) > 3:
                            print(f"       ... và {len(periods) - 3} periods khác")
                
                if pattern.get('single_focus_objects'):
                    single_count = len(pattern['single_focus_objects'])
                    print(f"\n  👁️  Objects chỉ nhìn 1 lần: {single_count}")
                    if single_count <= 5:
                        for obj_id in pattern['single_focus_objects']:
                            class_name = obj_id.split('_')[0] if '_' in obj_id else obj_id
                            emoji = "📖" if class_name == 'book' else ("👤" if class_name == 'person' else "📦")
                            print(f"    {emoji} {obj_id}")
            
            # Model Information (NEW)
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
        print("   Video có thể quá dài hoặc xử lý mất nhiều thời gian.")
        print("   Hãy thử với video ngắn hơn hoặc tăng timeout.")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ LỖI: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ưu tiên: Sử dụng VIDEO_PATH đã set trong code
    if VIDEO_PATH and Path(VIDEO_PATH).exists():
        video_path = VIDEO_PATH
        target_type = "camera"
        print("=" * 60)
        print("📹 Sử dụng VIDEO_PATH từ code")
        print(f"   Path: {video_path}")
        print("=" * 60)
        print()
    # Nếu VIDEO_PATH không set hoặc file không tồn tại, dùng command line argument
    elif len(sys.argv) >= 2:
        video_path = sys.argv[1]
        target_type = sys.argv[2] if len(sys.argv) > 2 else "camera"
    # Nếu không có gì, hiển thị hướng dẫn
    else:
        print("=" * 60)
        print("❌ CHƯA CẤU HÌNH VIDEO PATH")
        print("=" * 60)
        print("\n📝 CÁCH 1: Sửa VIDEO_PATH trong code (Khuyến nghị)")
        print("   Mở file test_gaze_api.py và sửa dòng:")
        print("   VIDEO_PATH = r\"C:\\Users\\Admin\\Desktop\\tiger.mp4\"")
        print("\n📝 CÁCH 2: Truyền qua command line")
        print(f"   python {sys.argv[0]} <path_to_video_file> [target_type]")
        print("\n📋 Ví dụ:")
        print(f"   python {sys.argv[0]} test_video.mp4")
        print(f"   python {sys.argv[0]} test_video.mp4 camera")
        print(f"   python {sys.argv[0]} C:/Users/Admin/Videos/kid_video.mp4")
        print("\n💡 Lưu ý:")
        print("   - API sẽ tự động detect face của trẻ và người lớn")
        print("   - Eye contact được tính khi focusing vào camera hoặc người lớn")
        print("   - Video nên có face rõ ràng để kết quả chính xác")
        print("=" * 60)
        sys.exit(1)
    
    # Validate target_type
    if target_type not in ["camera", "parent", "face"]:
        print(f"⚠️  Warning: target_type '{target_type}' không hợp lệ, sử dụng 'camera'")
        target_type = "camera"
    
    # Kiểm tra file có tồn tại không
    if not Path(video_path).exists():
        print("=" * 60)
        print("❌ FILE VIDEO KHÔNG TỒN TẠI")
        print("=" * 60)
        print(f"   Không tìm thấy: {video_path}")
        print("\n💡 Hãy kiểm tra lại đường dẫn hoặc sửa VIDEO_PATH trong code")
        print("=" * 60)
        sys.exit(1)
    
    test_gaze_api(video_path, target_type)

