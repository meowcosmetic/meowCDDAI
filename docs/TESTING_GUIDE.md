# Hướng dẫn Chạy và Test Project

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường](#2-cài-đặt-môi-trường)
3. [Khởi động server](#3-khởi-động-server)
4. [Test từng tính năng](#4-test-từng-tính-năng)
5. [Chạy toàn bộ test suite](#5-chạy-toàn-bộ-test-suite)
6. [Test API qua HTTP](#6-test-api-qua-http)

---

## 1. Yêu cầu hệ thống

- Python **3.10** hoặc **3.12**
- Windows (có sẵn `venv312`) hoặc Linux/macOS
- Camera (cho test real-time) — tùy chọn
- GPU NVIDIA + CUDA 11.8+ — tùy chọn, tăng tốc AI inference

---

## 2. Cài đặt môi trường

### Bước 1 — Tạo và kích hoạt virtual environment

```cmd
REM Windows (đã có sẵn venv312)
venv312\Scripts\activate

REM Hoặc tạo mới
python -m venv venv312
venv312\Scripts\activate
```

### Bước 2 — Cài dependencies

```cmd
pip install -r requirements.txt
```

Nếu có GPU NVIDIA, cài PyTorch CUDA trước:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Bước 3 — Tải model weights (tùy chọn, cần cho object detection)

Project dùng **YOLOv8 OID** (Open Images Dataset V7, 600 classes). File `yolov8l-oiv7.pt` đã có sẵn trong thư mục gốc — không cần tải thêm.

Nếu cần tải lại hoặc muốn thử size khác:

```cmd
REM Large (mặc định, đang dùng — phù hợp RTX 3060 12GB)
download_oid_model.bat
REM Hoặc: python download_oid_model.py l

REM XLarge (lớn hơn, chính xác hơn một chút, ~136MB)
python download_oid_model.py x
```

> **Ghi chú RTX 3060:** `yolov8l-oiv7.pt` (~84MB) là lựa chọn tối ưu cho 12GB VRAM.
> Model `x` chỉ cải thiện accuracy ~2-3% nhưng chậm hơn đáng kể.
> Để tận dụng GPU tốt hơn, tăng `object_detection_interval` xuống `1` trong `config.yaml`.

> **Lưu ý:** Nếu không có model, object detection bị tắt nhưng gaze tracking vẫn hoạt động. Các test unit/property vẫn chạy bình thường.

### Bước 4 — Kiểm tra cài đặt

```cmd
python -c "import mediapipe; import cv2; import numpy; print('OK')"
python -c "from ai_enhanced_gaze_tracking.core.data_models import GazeEstimate; print('Module OK')"
```

---

## 3. Khởi động server

### Cách nhanh (Windows)

```cmd
run_server_python312.bat
```

### Cách thủ công

```cmd
venv312\Scripts\activate
python main.py
```

Server chạy tại: `http://localhost:8102`

Swagger UI (tài liệu API tương tác): `http://localhost:8102/docs`

---

## 4. Test từng tính năng

Kích hoạt môi trường trước khi chạy bất kỳ lệnh nào:

```cmd
venv312\Scripts\activate
```

---

### 4.1 Head Pose Compensation (Yêu cầu 1)

Bù trừ góc nghiêng đầu ±45° pitch, ±30° yaw, tính nhất quán ma trận 3D.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_head_pose_compensation.py ai_enhanced_gaze_tracking/tests/test_head_pose_accuracy.py ai_enhanced_gaze_tracking/tests/test_3d_transformations.py -v
```

---

### 4.2 Camera Angle Correction (Yêu cầu 2)

Tự động phát hiện và bù trừ góc camera lệch ±30°.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_camera_angle_correction.py ai_enhanced_gaze_tracking/tests/test_calibration_coordinate_system.py -v
```

---

### 4.3 AI Gaze Estimation (Yêu cầu 3)

Model AI dự đoán hướng nhìn, fallback khi confidence thấp, ensemble nhiều model.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_ai_model_accuracy.py ai_enhanced_gaze_tracking/tests/test_confidence_based_fallback.py ai_enhanced_gaze_tracking/tests/test_ensemble_improvement.py -v
```

---

### 4.4 Multi-Modal Sensor Fusion (Yêu cầu 4)

Kết hợp 2D landmarks + 3D pose + AI, adaptive weights, giải quyết xung đột.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_fusion_weighting.py ai_enhanced_gaze_tracking/tests/test_adaptive_weights.py ai_enhanced_gaze_tracking/tests/test_conflict_resolution.py -v
```

---

### 4.5 Automatic Calibration (Yêu cầu 5)

Tự động hiệu chỉnh theo từng người dùng, reference point calibration.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_parameter_adaptation.py ai_enhanced_gaze_tracking/tests/test_reference_point_calibration.py -v
```

---

### 4.6 Focus Detection (Yêu cầu 6)

Phát hiện focus, loại bỏ false positive, ray casting 3D, phân loại wandering.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_focus_stability.py ai_enhanced_gaze_tracking/tests/test_non_object_focus_rejection.py ai_enhanced_gaze_tracking/tests/test_3d_ray_casting.py ai_enhanced_gaze_tracking/tests/test_wandering_classification.py -v
```

---

### 4.7 Real-Time Performance (Yêu cầu 7)

Đảm bảo ≥25 FPS, tự động giảm độ phức tạp khi tải cao.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_real_time_performance.py ai_enhanced_gaze_tracking/tests/test_adaptive_performance.py -v
```

---

### 4.8 Error Handling & Graceful Degradation (Yêu cầu 8)

Hệ thống không crash khi thiếu tài nguyên hoặc component lỗi.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_graceful_degradation.py ai_enhanced_gaze_tracking/tests/test_face_detection_reliability.py -v
```

---

### 4.9 Data Quality Assessment (Yêu cầu 9)

Confidence score, đánh giá chất lượng dữ liệu, flagging segment không tin cậy.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_quality_assessment.py ai_enhanced_gaze_tracking/tests/test_confidence_completeness.py ai_enhanced_gaze_tracking/tests/test_data_validation_flagging.py -v
```

---

### 4.10 Backward API Compatibility (Yêu cầu 10)

Output tương thích với API cũ, config bridge legacy ↔ enhanced.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_api_compatibility.py -v
```

---

### 4.11 End-to-End Integration

Toàn bộ pipeline từ frame đầu vào đến gaze output + legacy response.

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/test_end_to_end_integration.py -v
```

---

## 5. Chạy toàn bộ test suite

### Chạy tất cả

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/ -v
```

### Chạy với báo cáo ngắn gọn

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/ -v --tb=short
```

### Chạy và xem coverage

```cmd
pip install pytest-cov
python -m pytest ai_enhanced_gaze_tracking/tests/ --cov=ai_enhanced_gaze_tracking --cov-report=term-missing
```

### Chạy chỉ property-based tests (Hypothesis)

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/ -v -k "property or Property"
```

### Chạy nhanh (bỏ qua slow tests)

```cmd
python -m pytest ai_enhanced_gaze_tracking/tests/ -v -m "not slow"
```

### Kết quả mong đợi

```
========================= X passed in Y.YYs =========================
```

Tất cả test phải **PASSED**. Nếu có **FAILED**, xem phần [Troubleshooting](#troubleshooting).

---

## 6. Test API qua HTTP

Sau khi server đang chạy tại `http://localhost:8102`:

### Kiểm tra server hoạt động

```cmd
curl http://localhost:8102/docs
```

Hoặc mở trình duyệt: `http://localhost:8102/docs`

### Phân tích gaze từ video file

```cmd
curl -X POST "http://localhost:8102/screening/gaze/analyze" ^
  -F "video=@path\to\video.mp4" ^
  -F "target_type=camera" ^
  -F "show_video=false"
```

### Phân tích gaze từ camera real-time

```cmd
curl -X POST "http://localhost:8102/screening/gaze/analyze_camera" ^
  -F "camera_id=0" ^
  -F "target_type=camera" ^
  -F "max_duration=30" ^
  -F "show_video=true"
```

> Nhấn `q` hoặc `ESC` trong cửa sổ video để dừng.

### Dùng file test có sẵn

```cmd
REM Test gaze API
python test_gaze_api.py

REM Test camera API
python test_camera_api.py

REM Test các API khác
python test_expression_api.py
python test_pose_api.py
python test_speech_api.py
python test_interaction_api.py
```

### Cấu trúc response (legacy-compatible)

```json
{
  "eye_contact_percentage": 65.3,
  "total_frames": 900,
  "analyzed_duration": 30.0,
  "focusing_duration": 12.5,
  "attention_to_person_percentage": 20.0,
  "attention_to_objects_percentage": 45.0,
  "gaze_wandering_score": 15.2,
  "focus_timeline": [...],
  "wandering_periods": [...],
  "risk_score": 22.1,

  "enhanced_version": "2.0.0",
  "overall_quality_score": 0.82,
  "head_pose_compensation_active": true,
  "camera_angle_correction_active": true,
  "confidence_scores": [...]
}
```

---

## Troubleshooting

**`ModuleNotFoundError: mediapipe`**
```cmd
pip install mediapipe>=0.10.0
```

**`ModuleNotFoundError: cv2`**
```cmd
pip install opencv-python>=4.8.0
```

**Test fail do thiếu model weights**
```cmd
python download_oid_model.py
```

**Camera không mở được**
- Kiểm tra camera_id (thử `0`, `1`, `2`)
- Đảm bảo không có app khác đang dùng camera

**Server không start**
```cmd
REM Kiểm tra port 8102 có bị chiếm không
netstat -ano | findstr 8102
```

Xem thêm: [`docs/troubleshooting.md`](troubleshooting.md)
