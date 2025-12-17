# Book Vector Service

Service FastAPI để lưu trữ và tìm kiếm vector embedding sử dụng Qdrant với named vectors (summary & content).

## Tính năng

- ✅ Upload file JSON với format mới (summary + content)
- ✅ Named vectors: mỗi item có 2 vectors (summary và content)
- ✅ Semantic search với embedding vectors
- ✅ Keyword search với BM25
- ✅ Hybrid search kết hợp cả hai
- ✅ Tự động phát hiện và sử dụng GPU
- ✅ Batch processing tối ưu

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Kiểm tra: http://localhost:6333/dashboard

### 3. Start Service

```bash
python main.py
```

Service sẽ chạy tại: http://localhost:8102

## Cấu hình

Tạo file `.env` (optional):

```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=books

# GPU
USE_GPU=auto          # auto, true, false
GPU_DEVICE_ID=0       # 0, 1, 2...
BATCH_SIZE=32         # Batch size for encoding
```

## Upload Format

File JSON phải là array các object:

```json
[
  {
    "summary": "Tóm tắt ngắn gọn",
    "content": "Nội dung đầy đủ"
  },
  {
    "summary": "Tóm tắt thứ hai",
    "content": "Nội dung thứ hai"
  }
]
```

## API Endpoints

Xem chi tiết tại [API.md](./API.md)

### Upload
- `POST /upload-book-file` - Upload file JSON

### Search
- `POST /search` - Embedding search
- `POST /search-keywords` - Keyword search
- `POST /search-hybrid` - Hybrid search (khuyến nghị)
- `GET /search-by-book-id/{book_id}` - Lấy tất cả items của một book

### Admin
- `GET /health` - Health check
- `GET /collection-info` - Thông tin collection
- `DELETE /delete-book/{book_id}` - Xóa book
- `POST /rebuild-keyword-index` - Rebuild keyword index

## Ví dụ sử dụng

### Upload

```bash
curl -X POST "http://localhost:8102/upload-book-file" \
  -F "file=@data.json" \
  -F "book_id=my_book_001"
```

### Search

```bash
curl -X POST "http://localhost:8102/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Trẻ quay đầu và nhìn về phía âm thanh",
    "limit": 10,
    "score_threshold": 0.3
  }'
```

## Cấu trúc dữ liệu

Mỗi item trong Qdrant:
- **ID**: UUID
- **Named Vectors**:
  - `summary`: 1024D vector
  - `content`: 1024D vector
- **Payload**:
  ```json
  {
    "book_id": "string",
    "summary": "string",
    "content": "string"
  }
  ```

## GPU Support

Service tự động phát hiện GPU. Cấu hình trong `.env`:
- `USE_GPU=auto` - Tự động phát hiện
- `USE_GPU=true` - Bắt buộc dùng GPU
- `USE_GPU=false` - Chỉ dùng CPU
- `BATCH_SIZE=32` - Tăng nếu GPU mạnh (64, 128...)

## Documentation

- **API Docs**: http://localhost:8102/docs (Swagger UI)
- **ReDoc**: http://localhost:8102/redoc
- **Chi tiết API**: [API.md](./API.md)

## Tech Stack

- **FastAPI** - Web framework
- **Qdrant** - Vector database
- **Sentence Transformers** - Embedding model (multilingual-e5-large)
- **PyTorch** - ML framework
- **BM25** - Keyword search
