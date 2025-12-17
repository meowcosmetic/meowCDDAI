# 🚀 GPU Setup cho Gaze Tracking API

## Tổng quan

Gaze Tracking API hỗ trợ GPU acceleration để tăng tốc độ xử lý video. Có 2 cách sử dụng GPU:

1. **OpenCV với CUDA** - Tối ưu cho video processing
2. **MediaPipe với GPU delegate** - Tối ưu cho face detection

## Cấu hình

Thêm vào file `.env`:

```env
# GPU Configuration
USE_GPU=auto          # auto, true, false
GPU_DEVICE_ID=0       # 0, 1, 2... (GPU nào để sử dụng)
```

## Option 1: OpenCV với CUDA (Khuyến nghị)

### Yêu cầu:
- NVIDIA GPU với CUDA support
- CUDA Toolkit đã cài đặt
- OpenCV được build với CUDA

### Cài đặt:

#### Windows:
```powershell
# Cài opencv-contrib-python với CUDA (phức tạp, cần build từ source)
# Hoặc dùng pre-built wheel từ:
# https://github.com/opencv/opencv-python/issues/534

# Tạm thời, OpenCV standard version không hỗ trợ CUDA
# Cần build từ source hoặc dùng Docker image có CUDA
```

#### Linux/Docker:
```bash
# Sử dụng Docker image có CUDA
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

### Kiểm tra:
```python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())  # Số GPU devices
```

## Option 2: MediaPipe GPU (Đơn giản hơn)

MediaPipe tự động sử dụng GPU nếu có PyTorch với CUDA.

### Cài đặt:
```bash
# Đảm bảo PyTorch có CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# MediaPipe sẽ tự động detect GPU
pip install mediapipe
```

### Kiểm tra:
```python
import torch
print(torch.cuda.is_available())  # True nếu có GPU
print(torch.cuda.get_device_name(0))  # Tên GPU
```

## Cách hoạt động

### Auto mode (mặc định):
- Tự động detect GPU nếu có
- Fallback về CPU nếu không có GPU
- Log thông tin GPU khi khởi động

### Manual mode:
```env
USE_GPU=true   # Bắt buộc dùng GPU (sẽ lỗi nếu không có)
USE_GPU=false  # Bắt buộc dùng CPU
USE_GPU=auto   # Tự động (mặc định)
```

## Performance

### CPU vs GPU:
- **CPU**: ~5-10 FPS cho video 1080p
- **GPU**: ~30-60 FPS cho video 1080p (tùy GPU)

### Lưu ý:
- GPU acceleration chỉ tăng tốc video processing
- Face detection vẫn chủ yếu trên CPU (Haar Cascade)
- MediaPipe có thể tận dụng GPU tốt hơn

## Troubleshooting

### GPU không được detect:
1. Kiểm tra CUDA đã cài: `nvidia-smi`
2. Kiểm tra PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
3. Kiểm tra OpenCV: `python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"`

### Lỗi memory:
- Giảm batch size
- Xử lý video ngắn hơn
- Giảm resolution

## Logs

Khi chạy, bạn sẽ thấy:
```
[Gaze] ✅ OpenCV GPU detected: 1 device(s)
[Gaze] Using GPU device: 0
[Gaze] Sử dụng OpenCV fallback mode (GPU accelerated)
```

hoặc

```
[Gaze] OpenCV không có CUDA support, sử dụng CPU
[Gaze] Sử dụng OpenCV fallback mode (CPU)
```

