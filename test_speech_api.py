"""
Script test cho API Speech & Audio Analysis
Phân tích tiếng nói và âm thanh từ file audio

Cách sử dụng:
    python test_speech_api.py <path_to_audio_file>
    
Ví dụ:
    python test_speech_api.py test_audio.wav
    python test_speech_api.py test_audio.mp3
"""
import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8102/screening/speech/analyze"

# ========================================
# CẤU HÌNH FILE PATH - SỬA ĐƯỜNG DẪN Ở ĐÂY
# ========================================
# Cách 1: Đặt đường dẫn trực tiếp (ưu tiên)
# Có thể là audio file (wav, mp3) hoặc video file (mp4, avi)
FILE_PATH = r"C:\Users\Admin\Desktop\mon.mp4"  # <-- SỬA ĐƯỜNG DẪN Ở ĐÂY

# Cách 2: Hoặc để None để dùng command line argument
# FILE_PATH = None

# ========================================
# CẤU HÌNH VIDEO DISPLAY
# ========================================
# Bật/tắt hiển thị video real-time trong quá trình xử lý (chỉ áp dụng cho video)
SHOW_VIDEO = True  # True = hiển thị video, False = không hiển thị

# Cách 2: Hoặc để None để dùng command line argument
# AUDIO_PATH = None

# ========================================

def test_speech_api(file_path: str):
    """
    Test API speech analysis với audio hoặc video file
    
    Args:
        file_path: Đường dẫn đến file audio hoặc video
    """
    if not Path(file_path).exists():
        print(f"❌ File không tồn tại: {file_path}")
        return
    
    # Kiểm tra loại file
    file_ext = Path(file_path).suffix.lower()
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    is_video = file_ext in video_extensions
    
    print("=" * 60)
    print("🎤 SPEECH & AUDIO ANALYSIS API TEST")
    print("=" * 60)
    if is_video:
        print(f"🎬 Video: {file_path}")
        print(f"   → Sẽ extract audio từ video và hiển thị kết quả trên video")
    else:
        print(f"🎵 Audio: {file_path}")
    print(f"🌐 API URL: {API_URL}")
    if is_video and SHOW_VIDEO:
        print(f"📺 Video Display: ENABLED (sẽ hiển thị video với annotations)")
        print("   → Nhấn 'q' trong cửa sổ video để tắt hiển thị")
    print("-" * 60)
    
    try:
        # Gửi request với file
        with open(file_path, 'rb') as file:
            content_type = 'video/mp4' if is_video else 'audio/wav'
            files = {
                'file': (Path(file_path).name, file, content_type)
            }
            data = {
                'show_video': 'true' if (is_video and SHOW_VIDEO) else 'false'
            }
            
            print("⏳ Đang xử lý (có thể mất vài phút)...")
            if is_video:
                print("   - Extracting audio from video...")
            print("   - Loading audio file...")
            print("   - Detecting voice activity...")
            print("   - Analyzing vocalizations...")
            print("   - Detecting babbling patterns...")
            if is_video:
                print("   - Displaying results on video...")
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
            print(f"  🎵 Audio Duration: {result['audio_duration']:.2f}s")
            print(f"  🗣️  Speech Duration: {result['speech_duration']:.2f}s")
            print(f"  📈 Speech Percentage: {result['speech_percentage']:.2f}%")
            
            if result['speech_percentage'] > 50:
                print(f"     → ✅ Tỷ lệ nói tốt")
            elif result['speech_percentage'] > 20:
                print(f"     → ⚠️  Tỷ lệ nói trung bình")
            else:
                print(f"     → ❌ Tỷ lệ nói thấp")
            
            print(f"  🔇 Silence Percentage: {result['silence_percentage']:.2f}%")
            print(f"  🎤 Vocalization Frequency: {result['vocalization_frequency']:.2f} vocalizations/s")
            
            if result['vocalization_frequency'] > 1.0:
                print(f"     → ✅ Tần suất phát âm tốt")
            elif result['vocalization_frequency'] > 0.5:
                print(f"     → ⚠️  Tần suất phát âm trung bình")
            else:
                print(f"     → ❌ Tần suất phát âm thấp")
            
            print(f"  👶 Babbling Detected: {'✅ Có' if result['babbling_detected'] else '❌ Không'}")
            if result['babbling_detected']:
                print(f"     → Có phát hiện bập bẹ (dấu hiệu tích cực)")
            else:
                print(f"     → Không phát hiện bập bẹ (có thể là dấu hiệu đáng lo)")
            
            # Vocalizations list
            if result.get('vocalizations'):
                print(f"\n  📋 Vocalizations: {len(result['vocalizations'])} events")
                for i, v in enumerate(result['vocalizations'][:5]):  # Hiển thị 5 đầu tiên
                    print(f"     {i+1}. {v['start_time']:.2f}s - {v['end_time']:.2f}s ({v['duration']:.2f}s)")
                if len(result['vocalizations']) > 5:
                    print(f"     ... và {len(result['vocalizations']) - 5} events khác")
            
            print(f"  📈 Risk Score: {result['risk_score']:.2f}/100")
            if result['risk_score'] < 30:
                print(f"     → ✅ Rủi ro thấp (tiếng nói tốt)")
            elif result['risk_score'] < 60:
                print(f"     → ⚠️  Rủi ro trung bình")
            else:
                print(f"     → ❌ Rủi ro cao (ít nói, ít bập bẹ)")
            
            # Phân loại giọng nói: Trẻ em vs Người lớn
            print("\n👥 PHÂN LOẠI GIỌNG NÓI:")
            print("-" * 60)
            child_duration = result.get('child_speech_duration', 0)
            adult_duration = result.get('adult_speech_duration', 0)
            child_percentage = result.get('child_speech_percentage', 0)
            adult_percentage = result.get('adult_speech_percentage', 0)
            
            print(f"  👶 Giọng trẻ em:")
            print(f"     - Thời lượng: {child_duration:.2f}s ({child_percentage:.1f}%)")
            print(f"     - Số segments: {len(result.get('child_speech_segments', []))}")
            
            print(f"  👨 Giọng người lớn:")
            print(f"     - Thời lượng: {adult_duration:.2f}s ({adult_percentage:.1f}%)")
            print(f"     - Số segments: {len(result.get('adult_speech_segments', []))}")
            
            # Hiển thị một vài segments mẫu
            if result.get('child_speech_segments'):
                print(f"\n  📋 Mẫu giọng trẻ em (5 đầu tiên):")
                for i, seg in enumerate(result['child_speech_segments'][:5]):
                    pitch_info = f", Pitch: {seg['pitch_mean']:.0f} Hz" if seg.get('pitch_mean') else ""
                    print(f"     {i+1}. {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s{pitch_info})")
            
            if result.get('adult_speech_segments'):
                print(f"\n  📋 Mẫu giọng người lớn (5 đầu tiên):")
                for i, seg in enumerate(result['adult_speech_segments'][:5]):
                    pitch_info = f", Pitch: {seg['pitch_mean']:.0f} Hz" if seg.get('pitch_mean') else ""
                    print(f"     {i+1}. {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s{pitch_info})")
            
            # Thống kê chi tiết
            print("\n📈 THỐNG KÊ CHI TIẾT:")
            print("-" * 60)
            speech_ratio = result['speech_duration'] / result['audio_duration'] if result['audio_duration'] > 0 else 0
            print(f"  📊 Tỷ lệ nói/silence:")
            print(f"     • Nói: {result['speech_percentage']:.1f}% ({result['speech_duration']:.1f}s)")
            print(f"     • Im lặng: {result['silence_percentage']:.1f}% ({result['audio_duration'] - result['speech_duration']:.1f}s)")
            
            print(f"\n  🎤 Vocalizations:")
            print(f"     • Số lượng: {result['vocalization_frequency'] * result['audio_duration']:.0f} events")
            print(f"     • Tần suất: {result['vocalization_frequency']:.2f} events/giây")
            
            # Đánh giá tổng thể
            print("\n💡 ĐÁNH GIÁ:")
            print("-" * 60)
            if result['speech_percentage'] > 40 and result['vocalization_frequency'] > 1.0 and result['babbling_detected']:
                print("  ✅ Tốt: Trẻ có tiếng nói tốt, tần suất phát âm cao, có bập bẹ")
            elif result['speech_percentage'] > 20 and result['vocalization_frequency'] > 0.5:
                print("  ⚠️  Trung bình: Trẻ có tiếng nói nhưng tần suất thấp")
            else:
                print("  ❌ Đáng lo: Trẻ ít nói, ít phát âm, không có bập bẹ")
                print("     → Có thể là dấu hiệu của ASD")
            
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
        print("\n❌ Timeout: Audio quá dài hoặc server xử lý chậm")
    except requests.exceptions.ConnectionError:
        print("\n❌ Không thể kết nối đến server")
        print("   Hãy đảm bảo server đang chạy tại:", API_URL)
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Xác định file path
    file_path = FILE_PATH
    
    # Nếu FILE_PATH là None, dùng command line argument
    if file_path is None or not Path(file_path).exists():
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            print("=" * 60)
            print("❌ CẦN CUNG CẤP ĐƯỜNG DẪN FILE")
            print("=" * 60)
            print("\nCách sử dụng:")
            print("  1. Sửa FILE_PATH trong file này")
            print("  2. Hoặc chạy: python test_speech_api.py <path_to_file>")
            print("\nVí dụ:")
            print("  python test_speech_api.py C:\\Users\\Admin\\Desktop\\audio.wav")
            print("  python test_speech_api.py C:\\Users\\Admin\\Desktop\\audio.mp3")
            print("  python test_speech_api.py C:\\Users\\Admin\\Desktop\\video.mp4")
            sys.exit(1)
    
    test_speech_api(file_path)

