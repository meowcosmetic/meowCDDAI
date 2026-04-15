# Hướng Dẫn Chạy Local Server meowCDDAI (Windows)

Tài liệu này hướng dẫn cách cấu hình và khởi chạy ứng dụng meowAI (Gaze, Speech, Pose, OID Detection) tại môi trường phát triển máy tính cục bộ (Local Máy Windows).

## Yêu Cầu Hệ Thống
1. **Python 3.12** chuyên dụng (quan trọng vì các thư viện như Mediapipe, Librosa hoạt động ổn định nhất trên nền tảng này).
2. Tùy chọn: **GPU (NVIDIA CUDA)** nếu cần chạy các Model nhận diện hành vi, LLM với tốc độ cao.
3. Database Stack (PostgreSQL & Qdrant) đã cấu hình.

---

## Bước 1: Chuẩn bị Môi trường (Environment)

Hệ thống cung cấp sẵn các tập lệnh Batch để tự cập nhật và tạo môi trường ảo Python 3.12 tách biệt (`venv312`) giúp máy tính bạn không bị bẩn thư viện hệ thống.

1. Bật Command Prompt hoặc PowerShell tại thư mục `meowCDDAI`.
2. Chạy file khởi tạo môi trường:
   ```cmd
   setup_python312.bat
   ```
   *Quá trình này sẽ:*
   - Cảnh báo nếu máy tính chưa có Python 3.12 (có thể điền lệnh `winget install Python.Python.3.12` nếu chưa có).
   - Tạo thư mục ảo `venv312`.
   - Cài đặt toàn bộ pip dependencies từ file `requirements.txt`.
   - Cài đặt Mediapipe cho nhận diện Gaze, Face.

---

## Bước 2: Tải Cấp Mô Hình (Weights/Models)

Tùy vào các Endpoint cụ thể bạn sẽ Test, hãy tải thêm Weights cần thiết để module OpenCV/YOLO hoặc Mediapipe hoạt động bình thường:

- **Tải tệp trọng số cho Nhận diện Đồ vật YOLO:**
  ```cmd
  download_yolo_weights.bat
  ```
- **Tải OID Model (Tuỳ chọn cho các tác vụ OID):**
  ```cmd
  download_oid_model.bat
  ```

---

## Bước 3: Cấu hình Môi Trường Biến (.env)

Mặc định ứng dụng đã có fallback cho Local (như localhost cho Database và Qdrant) trong file `config.py`. 
Tuy nhiên, bạn có thể tạo một tệp `.env` tại thư mục gốc `meowCDDAI` với các giá trị override nếu thiết lập của bạn khác mặc định:

```env
# Database Settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=MeowCDD
POSTGRES_USER=cdd_app_admin
POSTGRES_PASSWORD=cdd_app_admin

# Vector DB Settings
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=books

# Local LLM Config (Ollama)
USE_LOCAL_LLM=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=hf.co/unsloth/gpt-oss-20b-GGUF:Q4_K_M
```

---

## Bước 4: Khởi Chạy Server FastAPI

Để khởi chạy server trên Localhost với Hot-Reload, hãy thực thi file:
```cmd
run_server_python312.bat
```
*(File này sẽ báo "Verifying librosa installation..." và cài đặt thiếu tự động, sau đó nó sẽ gọi `python main.py`).*

**Mặc định:** API sẽ lắng nghe tại `http://localhost:8003` 
*(Bạn có thể vào http://localhost:8003/docs để xem cấu trúc API Swagger)*.

---

## Bước 5: Kiểm tra / Testing System API

Để mô phỏng Frontend trên máy cục bộ, thư mục có cung cấp sẵn các tập lệnh test độc lập. **Giữ terminal chạy Server (Bước 4) luôn mở**, và mở một Terminal khác để chạy:

1. **Test Module Nhận Diện Khung Hình / Gaze:**
   ```cmd
   python test_gaze_api.py
   ```
2. **Test Âm Thanh & API Speech:**
   ```cmd
   python test_speech_api.py
   ```
3. Hoặc chạy tập lệnh tổng hợp có sẵn:
   ```cmd
   run_test_python312.bat
   ```

---

## Trợ giúp Gỡ lỗi (Troubleshoot)

- **Lỗi `ModuleNotFoundError`: librosa hoặc psycopg2** -> Xin cấp quyền bật Script trước: `call venv312\Scripts\activate.bat`, hoặc chạy `install_speech_modules.bat`.
- **Thiếu DB Collection** -> Hãy đảm bảo Qdrant Docker (Port 6333) được khởi chạy, có thể qua file docker-compose.yml chính của stack `meow`.
- **Lỗi venv script không chạy được** -> Bật quyền ExecutionPolicy trên Windows Powershell: `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`.
