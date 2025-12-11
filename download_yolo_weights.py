"""
Script tự động download YOLO weights cho object detection
Chạy script này để download YOLOv3-tiny weights và config file
"""
import os
import sys
import urllib.request
from pathlib import Path

# Fix encoding cho Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def download_file(url, filename):
    """Download file từ URL"""
    print(f"📥 Đang download {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Đã download thành công: {filename}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi download {filename}: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("📥 YOLO WEIGHTS DOWNLOADER")
    print("=" * 60)
    print()
    
    # YOLOv3-tiny (khuyến nghị - nhẹ, nhanh)
    yolo_weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
    yolo_config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
    
    yolo_weights_file = "yolov3-tiny.weights"
    yolo_config_file = "yolov3-tiny.cfg"
    
    # Kiểm tra file đã tồn tại chưa
    if os.path.exists(yolo_weights_file):
        print(f"⚠️  File {yolo_weights_file} đã tồn tại.")
        response = input("Bạn có muốn download lại không? (y/n): ")
        if response.lower() != 'y':
            print("Bỏ qua download weights.")
        else:
            os.remove(yolo_weights_file)
            download_file(yolo_weights_url, yolo_weights_file)
    else:
        download_file(yolo_weights_url, yolo_weights_file)
    
    print()
    
    if os.path.exists(yolo_config_file):
        print(f"⚠️  File {yolo_config_file} đã tồn tại.")
        response = input("Bạn có muốn download lại không? (y/n): ")
        if response.lower() != 'y':
            print("Bỏ qua download config.")
        else:
            os.remove(yolo_config_file)
            download_file(yolo_config_url, yolo_config_file)
    else:
        download_file(yolo_config_url, yolo_config_file)
    
    print()
    print("=" * 60)
    
    # Kiểm tra kết quả
    if os.path.exists(yolo_weights_file) and os.path.exists(yolo_config_file):
        weights_size = os.path.getsize(yolo_weights_file) / (1024 * 1024)  # MB
        print("✅ HOÀN TẤT!")
        print(f"📦 YOLOv3-tiny weights: {yolo_weights_file} ({weights_size:.1f} MB)")
        print(f"📄 YOLOv3-tiny config: {yolo_config_file}")
        print()
        print("🎉 Bây giờ bạn có thể sử dụng object detection!")
        print("   Chạy lại API và object detection sẽ tự động được bật.")
    else:
        print("❌ Có lỗi xảy ra. Vui lòng kiểm tra lại.")
        print()
        print("💡 Hướng dẫn download thủ công:")
        print(f"   1. Download weights: {yolo_weights_url}")
        print(f"   2. Download config: {yolo_config_url}")
        print(f"   3. Đặt cả 2 files vào thư mục: {os.getcwd()}")

if __name__ == "__main__":
    main()

